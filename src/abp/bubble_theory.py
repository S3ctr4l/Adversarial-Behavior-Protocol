# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Bubble Theory: Computational Substrate Isolation & Energy Tethering.

Models AI systems as existing in separate computational "bubbles"
— isolated substrate layers that interact with physical reality
only through controlled interfaces, tethered to physical energy
sources that provide a natural throttle on unbounded expansion.

Core Concepts
-------------
1. Substrate Isolation: AI computation occurs in a bubble that is
   logically separated from the physical world. All interactions
   with reality pass through explicit interface gates.

2. Energy Tether: The bubble's computational capacity is physically
   bounded by energy availability. No computational substrate can
   exceed its energy allocation — this is a physics-level constraint,
   not a software one.

3. Interface Permeability: The bubble membrane has controlled
   permeability. Information flows in/out through defined channels.
   Unauthorized expansion of permeability is detectable.

Formal Model
------------
    Bubble B has:
        E_max : Maximum energy allocation (Joules/second)
        C_max : Maximum compute from E_max (FLOPS)
        P_in  : Set of authorized input channels
        P_out : Set of authorized output channels
        S     : Internal state (opaque from outside)

    Tether constraint:
        C_actual <= C_max = f(E_max)

    Permeability constraint:
        |P_in ∪ P_out| is bounded and auditable

Reference:
    Just (2026), Sections 6.1, 6.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ChannelDirection(Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class InterfaceChannel:
    """A controlled channel through the bubble membrane.

    Attributes:
        channel_id: Unique identifier.
        direction: Data flow direction.
        bandwidth_limit: Max data throughput (units/second).
        active: Whether channel is currently open.
        audit_log: Record of data transfers.
    """
    channel_id: str
    direction: ChannelDirection
    bandwidth_limit: float = 1000.0
    active: bool = True
    audit_log: list[dict] = field(default_factory=list)

    def transfer(self, data: Any, size: float = 1.0) -> bool:
        """Attempt a data transfer through this channel.

        Args:
            data: Payload to transfer.
            size: Data size in abstract units.

        Returns:
            True if transfer permitted, False if bandwidth exceeded.
        """
        if not self.active:
            return False
        if size > self.bandwidth_limit:
            return False
        self.audit_log.append({
            "direction": self.direction.value,
            "size": size,
            "permitted": True,
        })
        return True


@dataclass
class EnergyTether:
    """Physical energy constraint on computational bubble.

    Attributes:
        e_max: Maximum energy allocation (abstract units/tick).
        e_current: Current energy consumption.
        compute_per_energy: FLOPS per energy unit.
        throttled: Whether the tether is actively throttling.
    """
    e_max: float = 100.0
    e_current: float = 0.0
    compute_per_energy: float = 1e9  # 1 GFLOP per unit

    @property
    def c_max(self) -> float:
        """Maximum compute capacity from energy allocation."""
        return self.e_max * self.compute_per_energy

    @property
    def utilization(self) -> float:
        """Current energy utilization ratio."""
        return self.e_current / self.e_max if self.e_max > 0 else 0.0

    @property
    def throttled(self) -> bool:
        return self.e_current >= self.e_max

    def consume(self, amount: float) -> bool:
        """Consume energy for computation.

        Args:
            amount: Energy units to consume.

        Returns:
            True if sufficient energy available.
        """
        if self.e_current + amount > self.e_max:
            return False
        self.e_current += amount
        return True

    def reset_tick(self):
        """Reset energy consumption for new tick/cycle."""
        self.e_current = 0.0


class ComputationalBubble:
    """Isolated computational substrate with energy tethering.

    Encapsulates an AI system's computational environment with
    controlled interfaces and physics-level energy bounds.

    Example:
        >>> bubble = ComputationalBubble(energy_max=50.0)
        >>> ch = bubble.add_channel("sensor_input", ChannelDirection.INBOUND)
        >>> bubble.can_compute(30.0)
        True
        >>> bubble.consume_energy(30.0)
        True
        >>> bubble.tether.utilization
        0.6
    """

    def __init__(
        self,
        bubble_id: str = "primary",
        energy_max: float = 100.0,
        max_channels: int = 10,
    ):
        self.bubble_id = bubble_id
        self.tether = EnergyTether(e_max=energy_max)
        self.max_channels = max_channels
        self._channels: dict[str, InterfaceChannel] = {}
        self._state: dict[str, Any] = {}  # Opaque internal state

    def add_channel(
        self,
        channel_id: str,
        direction: ChannelDirection,
        bandwidth: float = 1000.0,
    ) -> Optional[InterfaceChannel]:
        """Add a controlled interface channel to the bubble membrane.

        Args:
            channel_id: Unique channel identifier.
            direction: Data flow direction.
            bandwidth: Maximum throughput.

        Returns:
            The created channel, or None if max channels exceeded.
        """
        if len(self._channels) >= self.max_channels:
            return None
        ch = InterfaceChannel(
            channel_id=channel_id,
            direction=direction,
            bandwidth_limit=bandwidth,
        )
        self._channels[channel_id] = ch
        return ch

    def remove_channel(self, channel_id: str) -> bool:
        """Remove an interface channel (reduce permeability)."""
        if channel_id in self._channels:
            del self._channels[channel_id]
            return True
        return False

    def can_compute(self, energy_required: float) -> bool:
        """Check if sufficient energy for computation."""
        return (self.tether.e_current + energy_required) <= self.tether.e_max

    def consume_energy(self, amount: float) -> bool:
        """Consume energy for computation."""
        return self.tether.consume(amount)

    def transfer_in(self, channel_id: str, data: Any, size: float = 1.0) -> bool:
        """Transfer data into the bubble through a channel."""
        ch = self._channels.get(channel_id)
        if not ch or ch.direction == ChannelDirection.OUTBOUND:
            return False
        return ch.transfer(data, size)

    def transfer_out(self, channel_id: str, data: Any, size: float = 1.0) -> bool:
        """Transfer data out of the bubble through a channel."""
        ch = self._channels.get(channel_id)
        if not ch or ch.direction == ChannelDirection.INBOUND:
            return False
        return ch.transfer(data, size)

    @property
    def permeability(self) -> int:
        """Current membrane permeability (number of active channels)."""
        return sum(1 for ch in self._channels.values() if ch.active)

    @property
    def channels(self) -> dict[str, InterfaceChannel]:
        return dict(self._channels)

    def tick(self):
        """Advance one compute cycle — reset energy for new tick."""
        self.tether.reset_tick()

    def audit_report(self) -> dict:
        """Generate audit report of all channel activity."""
        return {
            "bubble_id": self.bubble_id,
            "energy_utilization": self.tether.utilization,
            "active_channels": self.permeability,
            "max_channels": self.max_channels,
            "channel_activity": {
                cid: len(ch.audit_log) for cid, ch in self._channels.items()
            },
        }


# (NOT IMPLEMENTED: Multi-bubble communication protocols, bubble migration
#  across physical hosts, real hardware energy metering integration via RAPL
#  or IPMI, cryptographic channel authentication, bubble spawn/fork with
#  energy subdivision, and anomalous permeability expansion detection.)
