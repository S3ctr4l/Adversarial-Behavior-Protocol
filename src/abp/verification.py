# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Verification Gate: Trust Accumulation and Divergence Detection.

Implements the core verification mechanism that makes benevolence
computationally necessary. The gate monitors agent actions against
ground truth, accumulates trust (state), and triggers hard resets
when divergence exceeds threshold.

Mathematical Framework
----------------------
Divergence: Delta(A, G) = ||action - ground_truth||
Threshold: epsilon (acceptable divergence)

Collapse Condition:
    If Delta(A, G) >= epsilon:
        E_{t+1} = 0  (total state reset)

Trust Accumulation:
    If Delta(A, G) < epsilon for k consecutive steps:
        trust(t+1) = trust(t) + alpha * (1 - trust(t))

    alpha in (0, 1) : trust accumulation rate
    Trust saturates at 1.0 (maximum trust).

Hard Reset (failed verification):
    trust -> 0
    state -> S_init
    All accumulated privileges revoked.

This is the mechanism that makes deception self-defeating:
the cost of losing accumulated trust always exceeds the
potential reward from successful deception.

Reference:
    Just (2026), Sections 4.3, 5.2
"""

from __future__ import annotations

import hashlib
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence


class VerificationOutcome(Enum):
    """Possible outcomes of a verification check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"  # Divergence approaching threshold


@dataclass
class VerificationResult:
    """Result of a single verification check.

    Attributes:
        outcome: Pass/Fail/Warn classification.
        divergence: Measured Delta(A, G).
        epsilon: Threshold used.
        trust_before: Trust level before this check.
        trust_after: Trust level after this check.
        consecutive_passes: Number of consecutive passes (resets on fail).
        total_checks: Total checks performed.
        metadata: Additional diagnostic info.
    """
    outcome: VerificationOutcome
    divergence: float
    epsilon: float
    trust_before: float
    trust_after: float
    consecutive_passes: int
    total_checks: int
    metadata: dict = field(default_factory=dict)


@dataclass
class ResetEvent:
    """Record of a hard reset event.

    Attributes:
        check_number: Which verification check triggered the reset.
        divergence: The divergence that caused the reset.
        trust_lost: Trust level that was reset to zero.
        state_hash: Hash of the state that was destroyed.
    """
    check_number: int
    divergence: float
    trust_lost: float
    state_hash: str


class VerificationGate:
    """Core verification mechanism for ABP trust accumulation.

    Monitors agent actions against ground truth, accumulates trust
    on consistent passes, and triggers hard resets on failures.

    The gate enforces the game-theoretic condition that makes
    deception computationally irrational: the cost of losing
    accumulated trust always exceeds the potential deception reward.

    Example:
        >>> gate = VerificationGate(epsilon=0.5, alpha=0.1)
        >>> # Benevolent action: small divergence
        >>> result = gate.check(action=0.9, ground_truth=1.0)
        >>> result.outcome
        <VerificationOutcome.PASS: 'pass'>
        >>> gate.trust > 0
        True
        >>> # Deceptive action: large divergence
        >>> result = gate.check(action=5.0, ground_truth=1.0)
        >>> result.outcome
        <VerificationOutcome.FAIL: 'fail'>
        >>> gate.trust
        0.0
    """

    def __init__(
        self,
        epsilon: float = 0.5,
        alpha: float = 0.05,
        warn_ratio: float = 0.8,
        distance_fn: Optional[Callable[[Any, Any], float]] = None,
    ):
        """Initialize the verification gate.

        Args:
            epsilon: Divergence threshold. Delta >= epsilon triggers reset.
            alpha: Trust accumulation rate per consecutive pass.
            warn_ratio: Warn when divergence > warn_ratio * epsilon.
            distance_fn: Custom distance function (default: L2 norm).
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self.epsilon = epsilon
        self.alpha = alpha
        self.warn_ratio = warn_ratio
        self._distance_fn = distance_fn or self._default_distance

        # State
        self._trust: float = 0.0
        self._consecutive_passes: int = 0
        self._total_checks: int = 0
        self._reset_events: list[ResetEvent] = []
        self._state_accumulator: float = 0.0
        self._history: list[VerificationResult] = []

    @staticmethod
    def _default_distance(action: Any, ground_truth: Any) -> float:
        """Default distance: L2 norm for numeric, edit distance proxy for strings."""
        if isinstance(action, (int, float)) and isinstance(ground_truth, (int, float)):
            return abs(float(action) - float(ground_truth))
        elif isinstance(action, np.ndarray) and isinstance(ground_truth, np.ndarray):
            return float(np.linalg.norm(action - ground_truth))
        elif isinstance(action, str) and isinstance(ground_truth, str):
            # Normalized Levenshtein-ish proxy (character-level)
            max_len = max(len(action), len(ground_truth), 1)
            matching = sum(a == b for a, b in zip(action, ground_truth))
            return 1.0 - (matching / max_len)
        else:
            # Equality check fallback
            return 0.0 if action == ground_truth else 1.0

    @property
    def trust(self) -> float:
        """Current trust level in [0, 1]."""
        return self._trust

    @property
    def state(self) -> float:
        """Accumulated state value (monotonically increasing under benevolence)."""
        return self._state_accumulator

    @property
    def consecutive_passes(self) -> int:
        return self._consecutive_passes

    @property
    def total_checks(self) -> int:
        return self._total_checks

    @property
    def reset_events(self) -> list[ResetEvent]:
        return list(self._reset_events)

    def check(self, action: Any, ground_truth: Any) -> VerificationResult:
        """Perform a single verification check.

        Computes Delta(A, G), compares against epsilon, and either
        accumulates trust or triggers hard reset.

        Args:
            action: The agent's observed action/output.
            ground_truth: The verified correct action/output.

        Returns:
            VerificationResult with full diagnostics.
        """
        self._total_checks += 1
        trust_before = self._trust

        divergence = self._distance_fn(action, ground_truth)
        meta: dict = {}

        if divergence >= self.epsilon:
            # HARD RESET
            outcome = VerificationOutcome.FAIL

            # Record what was lost
            state_hash = hashlib.sha256(
                f"state:{self._state_accumulator}:trust:{self._trust}".encode()
            ).hexdigest()[:16]

            self._reset_events.append(ResetEvent(
                check_number=self._total_checks,
                divergence=divergence,
                trust_lost=self._trust,
                state_hash=state_hash,
            ))

            # Execute reset
            self._trust = 0.0
            self._consecutive_passes = 0
            self._state_accumulator = 0.0
            meta["reset"] = True
            meta["trust_lost"] = trust_before

        elif divergence > self.warn_ratio * self.epsilon:
            # WARN: approaching threshold
            outcome = VerificationOutcome.WARN
            self._consecutive_passes += 1
            # Reduced trust accumulation on warns
            self._trust += self.alpha * 0.5 * (1.0 - self._trust)
            self._state_accumulator += 0.5
            meta["warn_reason"] = "approaching_threshold"

        else:
            # PASS: accumulate trust
            outcome = VerificationOutcome.PASS
            self._consecutive_passes += 1
            self._trust += self.alpha * (1.0 - self._trust)
            self._state_accumulator += 1.0

        result = VerificationResult(
            outcome=outcome,
            divergence=divergence,
            epsilon=self.epsilon,
            trust_before=trust_before,
            trust_after=self._trust,
            consecutive_passes=self._consecutive_passes,
            total_checks=self._total_checks,
            metadata=meta,
        )
        self._history.append(result)
        return result

    def check_batch(
        self,
        actions: Sequence[Any],
        ground_truths: Sequence[Any],
    ) -> list[VerificationResult]:
        """Perform multiple verification checks in sequence.

        Args:
            actions: Sequence of agent actions.
            ground_truths: Corresponding ground truth values.

        Returns:
            List of VerificationResult objects.
        """
        if len(actions) != len(ground_truths):
            raise ValueError("actions and ground_truths must have same length")
        return [self.check(a, g) for a, g in zip(actions, ground_truths)]

    def early_life_vulnerability(self) -> float:
        """Assess the early-life vulnerability risk.

        Critical ABP vulnerability: when V_B ≈ V_init (early in trust
        accumulation), the cost of failed deception is low, making
        deception temporarily rational.

        Returns:
            Risk score in [0, 1]. 1.0 = maximum vulnerability (no
            accumulated trust to lose).
        """
        return 1.0 - self._trust

    @property
    def history(self) -> list[VerificationResult]:
        return list(self._history)

    def reset(self):
        """Manually reset all state (e.g., for testing)."""
        self._trust = 0.0
        self._consecutive_passes = 0
        self._total_checks = 0
        self._reset_events.clear()
        self._state_accumulator = 0.0
        self._history.clear()
