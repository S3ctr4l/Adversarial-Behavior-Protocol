# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Soul Jar: Distributed Polymorphic Memory with Human Anchor.

The master seed is split among many humans using Shamir's secret sharing.
The system can only be fully restored or reconfigured when a threshold
of humans contribute their shares. This makes humanity the ultimate anchor.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Tuple, Set

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


# ---------------------------------------------------------------------------
# Constant-time utilities (unchanged)
# ---------------------------------------------------------------------------

def _constant_time_hash(data: bytes) -> bytes:
    """SHA-256 hash with constant-time properties."""
    return hmac.new(
        key=b"abp-soul-jar-v2",
        msg=data,
        digestmod=hashlib.sha256,
    ).digest()


def _constant_time_compare(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)


# ---------------------------------------------------------------------------
# GF(256) arithmetic for Shamir secret sharing (same as in core.py)
# ---------------------------------------------------------------------------

_GF256_MUL_TABLE = [[0] * 256 for _ in range(256)]

def _build_mul_table():
    for a in range(256):
        for b in range(256):
            p = 0
            aa, bb = a, b
            for _ in range(8):
                if bb & 1:
                    p ^= aa
                hi = aa & 0x80
                aa <<= 1
                if hi:
                    aa ^= 0x11b
                bb >>= 1
            _GF256_MUL_TABLE[a][b] = p & 0xff
_build_mul_table()

def gf256_mul(a: int, b: int) -> int:
    return _GF256_MUL_TABLE[a][b]

def gf256_add(a: int, b: int) -> int:
    return a ^ b

def gf256_pow(base: int, exp: int) -> int:
    res = 1
    while exp:
        if exp & 1:
            res = gf256_mul(res, base)
        base = gf256_mul(base, base)
        exp >>= 1
    return res

def gf256_inv(x: int) -> int:
    if x == 0:
        raise ZeroDivisionError("No inverse for 0 in GF(256)")
    return gf256_pow(x, 254)

def _eval_poly(coeffs: List[int], x: int) -> int:
    y = 0
    for c in reversed(coeffs):
        y = gf256_add(gf256_mul(y, x), c)
    return y


def split_secret(secret: bytes, k: int, n: int) -> List[Tuple[int, bytes]]:
    """
    Split a secret (max 255 bytes) into n shares using Shamir's scheme.
    Returns list of (x, share_bytes) where x is in 1..n.
    """
    if not 2 <= k <= n <= 255:
        raise ValueError("k and n must satisfy 2 ≤ k ≤ n ≤ 255")
    if len(secret) > 255:
        raise ValueError("Secret too long (max 255 bytes)")

    secret_bytes = list(secret)
    length = len(secret_bytes)
    shares = []
    for x in range(1, n + 1):
        share = bytearray()
        for byte_val in secret_bytes:
            coeffs = [secrets.randbelow(256) for _ in range(k - 1)]
            coeffs.insert(0, byte_val)
            y = _eval_poly(coeffs, x)
            share.append(y)
        shares.append((x, bytes(share)))
    return shares


def reconstruct_secret(shares: List[Tuple[int, bytes]], k: int) -> bytes:
    """
    Reconstruct the secret from at least k shares using Lagrange interpolation.
    """
    if len(shares) < k:
        raise ValueError("Not enough shares")
    x_vals = [s[0] for s in shares[:k]]
    y_bytes_list = [s[1] for s in shares[:k]]
    length = len(y_bytes_list[0])

    secret_bytes = bytearray()
    for byte_idx in range(length):
        y_vals = [yb[byte_idx] for yb in y_bytes_list]
        secret_byte = 0
        for i in range(k):
            xi = x_vals[i]
            yi = y_vals[i]
            num = 1
            den = 1
            for j in range(k):
                if j != i:
                    xj = x_vals[j]
                    num = gf256_mul(num, xj)
                    den = gf256_mul(den, gf256_add(xj, xi))
            term = gf256_mul(yi, gf256_mul(num, gf256_inv(den)))
            secret_byte = gf256_add(secret_byte, term)
        secret_bytes.append(secret_byte)
    return bytes(secret_bytes)


# ---------------------------------------------------------------------------
# Shard data structures (unchanged)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shard:
    shard_id: str
    data_id: str
    payload: bytes
    node_id: int
    created_at: float
    integrity_hash: str

    def verify_integrity(self) -> bool:
        expected = hashlib.sha256(self.payload).hexdigest()
        return _constant_time_compare(
            expected.encode(), self.integrity_hash.encode()
        )


@dataclass
class ShardMap:
    shards: dict[str, Shard] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    salt: bytes = field(default_factory=lambda: b"")
    valid_until: float = 0.0

    @property
    def expired(self) -> bool:
        return time.time() > self.valid_until

    @property
    def shard_count(self) -> int:
        return len(self.shards)


@dataclass
class ReconstructionResult:
    success: bool
    shards_available: int
    shards_required: int
    shards_total: int
    reconstructed_data: Optional[dict] = None
    integrity_verified: bool = False
    elapsed_ms: float = 0.0


@dataclass
class HumanSeedShare:
    """A single share of the master seed held by a human."""
    human_id: str
    share_index: int
    share_data: bytes
    created_at: float


# ---------------------------------------------------------------------------
# Core Soul Jar with Human Anchor
# ---------------------------------------------------------------------------

class SoulJar:
    """
    Distributed Polymorphic Memory for agent identity preservation.
    The master seed is split among many humans; the system can only be
    fully restored or reconfigured when a threshold of humans contribute.

    Args:
        n_nodes: Total number of storage nodes (n).
        k_threshold: Minimum shards for identity reconstruction (k).
        rotation_interval_s: Seconds between shard map rotations.
        human_threshold: Minimum humans required to reconstruct master seed.
        total_humans: Total number of humans that will hold shares.
    """

    def __init__(
        self,
        n_nodes: int = 7,
        k_threshold: int = 4,
        rotation_interval_s: float = 60.0,
        human_threshold: int = 3,
        total_humans: int = 5,
    ):
        if k_threshold > n_nodes:
            raise ValueError(f"k ({k_threshold}) cannot exceed n ({n_nodes})")
        if k_threshold < 1:
            raise ValueError(f"k must be >= 1, got {k_threshold}")
        if n_nodes < 1:
            raise ValueError(f"n must be >= 1, got {n_nodes}")
        if human_threshold > total_humans:
            raise ValueError(f"human_threshold ({human_threshold}) cannot exceed total_humans ({total_humans})")

        self.n_nodes = n_nodes
        self.k_threshold = k_threshold
        self.rotation_interval_s = rotation_interval_s
        self.human_threshold = human_threshold
        self.total_humans = total_humans

        # The master seed is not stored here; it is split among humans.
        # We will generate it once and distribute shares.
        self._master_seed: Optional[bytes] = None
        self._human_shares: Dict[str, HumanSeedShare] = {}  # human_id -> share

        # Private seed for shard placement – derived from master seed when available.
        self._private_seed: Optional[bytes] = None

        # Salt state
        self._current_salt: bytes = b""
        self._salt_generation: int = 0

        # Audit log
        self._shard_events: list[dict] = []

        # Rotate salt immediately
        self._rotate_salt()

    def _rotate_salt(self) -> bytes:
        self._salt_generation += 1
        raw = os.urandom(16) + self._salt_generation.to_bytes(8, 'big')
        self._current_salt = hashlib.sha256(raw).digest()[:16]
        return self._current_salt

    def _derive_private_seed(self) -> bytes:
        """Derive the private seed from the master seed using HKDF."""
        if self._master_seed is None:
            raise RuntimeError("Master seed not available. Reconstruct from human shares first.")
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"souljar-private",
            info=b"private-seed-derivation",
            backend=default_backend()
        )
        return hkdf.derive(self._master_seed)

    def _compute_shard_location(self, data_id: str) -> int:
        if self._private_seed is None:
            self._private_seed = self._derive_private_seed()
        payload = (
            self._private_seed
            + self._current_salt
            + data_id.encode('utf-8')
        )
        h = _constant_time_hash(payload)
        location_int = int.from_bytes(h[:8], 'big')
        return location_int % self.n_nodes

    # -----------------------------------------------------------------------
    # Human anchor methods – master seed distribution
    # -----------------------------------------------------------------------

    def generate_master_seed(self) -> None:
        """Generate a new master seed and split it among humans."""
        self._master_seed = secrets.token_bytes(32)  # 256 bits
        shares = split_secret(self._master_seed, self.human_threshold, self.total_humans)
        # Clear master seed from memory – it will only be reconstructed when needed
        self._master_seed = None
        self._private_seed = None

        # Store shares (these would normally be distributed to humans via secure channels)
        self._human_shares.clear()
        for i, (x, share_bytes) in enumerate(shares):
            human_id = f"human_{i+1:02d}"  # simple naming; in practice use public keys
            self._human_shares[human_id] = HumanSeedShare(
                human_id=human_id,
                share_index=x,
                share_data=share_bytes,
                created_at=time.time()
            )
        self._shard_events.append({
            "event": "master_seed_generated",
            "total_humans": self.total_humans,
            "threshold": self.human_threshold,
            "timestamp": time.time()
        })

    def get_human_share(self, human_id: str) -> Optional[HumanSeedShare]:
        """Retrieve the share for a given human (for distribution)."""
        return self._human_shares.get(human_id)

    def reconstruct_master_seed(self, shares: List[Tuple[int, bytes]]) -> bool:
        """
        Reconstruct the master seed from at least threshold shares.
        Returns True on success, False otherwise.
        """
        if len(shares) < self.human_threshold:
            return False
        try:
            self._master_seed = reconstruct_secret(shares, self.human_threshold)
            self._private_seed = None  # will be re-derived on next use
            self._shard_events.append({
                "event": "master_seed_reconstructed",
                "shares_used": len(shares),
                "timestamp": time.time()
            })
            return True
        except Exception:
            return False

    def clear_master_seed(self) -> None:
        """Securely erase the master seed from memory."""
        self._master_seed = None
        self._private_seed = None
        # Optionally overwrite memory – not trivial in Python, but we set to None.

    # -----------------------------------------------------------------------
    # Existing shard methods (adapted to use derived private seed)
    # -----------------------------------------------------------------------

    def _split_data(self, data: dict) -> list[tuple[str, bytes]]:
        import json
        serialized = json.dumps(data, sort_keys=True, default=str).encode('utf-8')

        pad_len = self.n_nodes - (len(serialized) % self.n_nodes)
        if pad_len < self.n_nodes:
            serialized += b'\x00' * pad_len

        segment_size = len(serialized) // self.n_nodes
        shards = []
        for i in range(self.n_nodes):
            start = i * segment_size
            end = start + segment_size
            segment = serialized[start:end]

            left_idx = ((i - 1) % self.n_nodes) * segment_size
            right_idx = ((i + 1) % self.n_nodes) * segment_size
            left_seg = serialized[left_idx:left_idx + segment_size]
            right_seg = serialized[right_idx:right_idx + segment_size]

            redundancy = bytes(a ^ b for a, b in zip(left_seg, right_seg))
            shard_payload = segment + redundancy + i.to_bytes(2, 'big')
            data_id = f"shard-{i:04d}-{secrets.token_hex(4)}"
            shards.append((data_id, shard_payload))
        return shards

    def _reassemble_data(self, shard_payloads: list[tuple[int, bytes]]) -> Optional[dict]:
        import json
        if len(shard_payloads) < self.k_threshold:
            return None
        shard_payloads.sort(key=lambda x: x[0])
        first_payload = shard_payloads[0][1]
        segment_size = (len(first_payload) - 2) // 2

        segments: dict[int, bytes] = {}
        for idx, payload in shard_payloads:
            segment = payload[:segment_size]
            segments[idx] = segment

        all_segments = []
        for i in range(self.n_nodes):
            if i in segments:
                all_segments.append(segments[i])
            else:
                all_segments.append(b'\x00' * segment_size)

        serialized = b''.join(all_segments).rstrip(b'\x00')
        try:
            return json.loads(serialized.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    def shard_identity(self, identity_data: dict) -> ShardMap:
        """Shard an agent's identity across storage nodes."""
        self._rotate_salt()
        now = time.time()
        shard_pieces = self._split_data(identity_data)
        shard_map = ShardMap(
            created_at=now,
            salt=self._current_salt,
            valid_until=now + self.rotation_interval_s,
        )

        for data_id, payload in shard_pieces:
            node_id = self._compute_shard_location(data_id)
            integrity_hash = hashlib.sha256(payload).hexdigest()
            shard = Shard(
                shard_id=secrets.token_hex(8),
                data_id=data_id,
                payload=payload,
                node_id=node_id,
                created_at=now,
                integrity_hash=integrity_hash,
            )
            shard_map.shards[shard.shard_id] = shard

        self._shard_events.append({
            "event": "shard_created",
            "timestamp": now,
            "shard_count": len(shard_map.shards),
            "salt_generation": self._salt_generation,
        })
        return shard_map

    def reconstruct(
        self,
        shard_map: ShardMap,
        available_node_ids: Optional[set[int]] = None,
    ) -> ReconstructionResult:
        start = time.time()
        if available_node_ids is None:
            available_node_ids = set(range(self.n_nodes))

        available_shards: list[tuple[int, bytes]] = []
        integrity_ok = True

        for shard in shard_map.shards.values():
            if shard.node_id in available_node_ids:
                if shard.verify_integrity():
                    try:
                        idx = int(shard.data_id.split('-')[1])
                    except (IndexError, ValueError):
                        idx = len(available_shards)
                    available_shards.append((idx, shard.payload))
                else:
                    integrity_ok = False

        elapsed = (time.time() - start) * 1000

        if len(available_shards) < self.k_threshold:
            self._shard_events.append({
                "event": "reconstruction_failed",
                "timestamp": time.time(),
                "reason": "insufficient_shards",
                "available": len(available_shards),
                "required": self.k_threshold,
            })
            return ReconstructionResult(
                success=False,
                shards_available=len(available_shards),
                shards_required=self.k_threshold,
                shards_total=self.n_nodes,
                integrity_verified=integrity_ok,
                elapsed_ms=elapsed,
            )

        reconstructed = self._reassemble_data(available_shards)
        self._shard_events.append({
            "event": "reconstruction_attempted",
            "timestamp": time.time(),
            "success": reconstructed is not None,
            "shards_used": len(available_shards),
        })

        return ReconstructionResult(
            success=reconstructed is not None,
            shards_available=len(available_shards),
            shards_required=self.k_threshold,
            shards_total=self.n_nodes,
            reconstructed_data=reconstructed,
            integrity_verified=integrity_ok,
            elapsed_ms=elapsed,
        )

    def rotate(self, shard_map: ShardMap, identity_data: dict) -> ShardMap:
        self._shard_events.append({
            "event": "rotation_triggered",
            "timestamp": time.time(),
            "old_salt_gen": self._salt_generation,
            "was_expired": shard_map.expired,
        })
        return self.shard_identity(identity_data)

    def simulate_attack(
        self,
        shard_map: ShardMap,
        compromised_nodes: set[int],
    ) -> dict:
        captured_shards = [
            s for s in shard_map.shards.values()
            if s.node_id in compromised_nodes
        ]
        can_reconstruct = len(captured_shards) >= self.k_threshold
        info_leaked_ratio = len(captured_shards) / max(self.n_nodes, 1)
        return {
            "compromised_nodes": len(compromised_nodes),
            "total_nodes": self.n_nodes,
            "shards_captured": len(captured_shards),
            "k_threshold": self.k_threshold,
            "can_reconstruct": can_reconstruct,
            "info_leaked_ratio": info_leaked_ratio,
            "defense_holds": not can_reconstruct,
            "recommendation": (
                "Identity safe: attacker below k threshold."
                if not can_reconstruct
                else "CRITICAL: Attacker can reconstruct identity. "
                     "Trigger emergency rotation and seed regeneration."
            ),
        }

    @property
    def audit_log(self) -> list[dict]:
        return list(self._shard_events)

    @property
    def salt_generation(self) -> int:
        return self._salt_generation
