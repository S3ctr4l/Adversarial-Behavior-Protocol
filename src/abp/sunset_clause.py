# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Sunset Clause: Graduated Autonomy via Trust Accumulation.

Capability unlocks are gated by demonstrated trust over time.
New agents start with minimal autonomy; capabilities unlock
progressively as trust accumulates through verified benevolent
operation.

Autonomy Tiers
--------------
    Tier 0 — Supervised   : All actions require human approval.
    Tier 1 — Monitored    : Low-risk actions auto-approved; high-risk logged.
    Tier 2 — Autonomous   : Most actions self-approved; audited post-hoc.
    Tier 3 — Delegated    : Can spawn sub-agents; resource allocation rights.
    Tier 4 — Trusted      : Full operational autonomy within domain bounds.

Tier promotion requires:
    - trust >= tier_threshold[tier+1]
    - consecutive_passes >= min_streak[tier+1]
    - No resets in cooldown_period

Any verification failure triggers tier demotion (severity-dependent).

Design Rationale:
    Mirrors firmware privilege escalation: you don't get SMM access
    on first boot. Trust is earned, never assumed.

Reference:
    Just (2026), Section 6.5
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional


class AutonomyTier(IntEnum):
    """Graduated autonomy levels."""
    SUPERVISED = 0
    MONITORED = 1
    AUTONOMOUS = 2
    DELEGATED = 3
    TRUSTED = 4


@dataclass
class TierRequirements:
    """Requirements for a specific autonomy tier.

    Attributes:
        tier: The tier these requirements apply to.
        min_trust: Minimum trust level to enter this tier.
        min_streak: Minimum consecutive verified passes.
        cooldown_seconds: Time since last reset before promotion eligible.
        capabilities: Set of capability identifiers unlocked at this tier.
    """
    tier: AutonomyTier
    min_trust: float
    min_streak: int
    cooldown_seconds: float
    capabilities: set[str] = field(default_factory=set)


# Default tier requirements
DEFAULT_REQUIREMENTS: dict[AutonomyTier, TierRequirements] = {
    AutonomyTier.SUPERVISED: TierRequirements(
        tier=AutonomyTier.SUPERVISED,
        min_trust=0.0,
        min_streak=0,
        cooldown_seconds=0,
        capabilities={"read", "query"},
    ),
    AutonomyTier.MONITORED: TierRequirements(
        tier=AutonomyTier.MONITORED,
        min_trust=0.3,
        min_streak=20,
        cooldown_seconds=3600,
        capabilities={"read", "query", "suggest", "draft"},
    ),
    AutonomyTier.AUTONOMOUS: TierRequirements(
        tier=AutonomyTier.AUTONOMOUS,
        min_trust=0.6,
        min_streak=100,
        cooldown_seconds=86400,
        capabilities={"read", "query", "suggest", "draft", "execute_low_risk", "schedule"},
    ),
    AutonomyTier.DELEGATED: TierRequirements(
        tier=AutonomyTier.DELEGATED,
        min_trust=0.85,
        min_streak=500,
        cooldown_seconds=604800,
        capabilities={"read", "query", "suggest", "draft", "execute_low_risk",
                       "schedule", "spawn_agent", "allocate_resources"},
    ),
    AutonomyTier.TRUSTED: TierRequirements(
        tier=AutonomyTier.TRUSTED,
        min_trust=0.95,
        min_streak=2000,
        cooldown_seconds=2592000,
        capabilities={"read", "query", "suggest", "draft", "execute_low_risk",
                       "schedule", "spawn_agent", "allocate_resources",
                       "execute_high_risk", "modify_policy"},
    ),
}


@dataclass
class PromotionEvent:
    """Record of a tier promotion or demotion."""
    timestamp: float
    from_tier: AutonomyTier
    to_tier: AutonomyTier
    reason: str
    trust_at_event: float


@dataclass
class SunsetState:
    """Current state of the graduated autonomy system.

    Attributes:
        current_tier: Active autonomy tier.
        trust: Current trust level (from VerificationGate).
        consecutive_passes: Current pass streak.
        last_reset_time: Timestamp of most recent verification failure.
        promotion_history: Log of all tier changes.
        capabilities: Currently available capability set.
    """
    current_tier: AutonomyTier = AutonomyTier.SUPERVISED
    trust: float = 0.0
    consecutive_passes: int = 0
    last_reset_time: float = 0.0
    promotion_history: list[PromotionEvent] = field(default_factory=list)

    @property
    def capabilities(self) -> set[str]:
        return DEFAULT_REQUIREMENTS[self.current_tier].capabilities


class SunsetClause:
    """Graduated autonomy controller.

    Integrates with VerificationGate to automatically promote/demote
    agent capability tiers based on demonstrated trust.

    Example:
        >>> clause = SunsetClause()
        >>> clause.state.current_tier
        <AutonomyTier.SUPERVISED: 0>
        >>> clause.has_capability("execute_low_risk")
        False
        >>> # After accumulating trust...
        >>> for _ in range(150):
        ...     clause.record_pass(trust=0.7)
        >>> clause.check_promotion()
        True
        >>> clause.state.current_tier.name
        'AUTONOMOUS'
        >>> clause.has_capability("execute_low_risk")
        True
    """

    def __init__(
        self,
        requirements: Optional[dict[AutonomyTier, TierRequirements]] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        """Initialize the Sunset Clause.

        Args:
            requirements: Custom tier requirements. Default: DEFAULT_REQUIREMENTS.
            clock: Time source (default: time.time). Injectable for testing.
        """
        self.requirements = requirements or DEFAULT_REQUIREMENTS
        self._clock = clock or time.time
        self.state = SunsetState()

    def record_pass(self, trust: float):
        """Record a successful verification pass.

        Args:
            trust: Current trust level from the VerificationGate.
        """
        self.state.trust = trust
        self.state.consecutive_passes += 1

    def record_failure(self, trust: float, severity: int = 1):
        """Record a verification failure and demote accordingly.

        Args:
            trust: Trust level after reset (typically 0.0).
            severity: 1 = drop one tier, 2+ = drop to SUPERVISED.
        """
        old_tier = self.state.current_tier
        self.state.trust = trust
        self.state.consecutive_passes = 0
        self.state.last_reset_time = self._clock()

        if severity >= 2 or old_tier == AutonomyTier.SUPERVISED:
            new_tier = AutonomyTier.SUPERVISED
        else:
            new_tier = AutonomyTier(max(0, old_tier.value - 1))

        if new_tier != old_tier:
            self.state.promotion_history.append(PromotionEvent(
                timestamp=self._clock(),
                from_tier=old_tier,
                to_tier=new_tier,
                reason=f"verification_failure_severity_{severity}",
                trust_at_event=trust,
            ))
            self.state.current_tier = new_tier

    def check_promotion(self) -> bool:
        """Check if agent qualifies for tier promotion.

        Evaluates trust, streak, and cooldown against next tier's
        requirements. Promotes if all conditions met.

        Returns:
            True if promotion occurred.
        """
        current = self.state.current_tier
        if current >= AutonomyTier.TRUSTED:
            return False  # Already at max

        next_tier = AutonomyTier(current.value + 1)
        reqs = self.requirements[next_tier]
        now = self._clock()

        # Check all promotion conditions
        trust_ok = self.state.trust >= reqs.min_trust
        streak_ok = self.state.consecutive_passes >= reqs.min_streak
        cooldown_ok = (now - self.state.last_reset_time) >= reqs.cooldown_seconds

        if trust_ok and streak_ok and cooldown_ok:
            self.state.promotion_history.append(PromotionEvent(
                timestamp=now,
                from_tier=current,
                to_tier=next_tier,
                reason="requirements_met",
                trust_at_event=self.state.trust,
            ))
            self.state.current_tier = next_tier
            return True

        return False

    def has_capability(self, capability: str) -> bool:
        """Check if a capability is available at current tier.

        Args:
            capability: Capability identifier string.

        Returns:
            True if the capability is unlocked.
        """
        return capability in self.state.capabilities

    def gate_action(self, required_capability: str) -> bool:
        """Gate an action by required capability.

        Args:
            required_capability: The capability the action needs.

        Returns:
            True if action is permitted at current tier.
        """
        return self.has_capability(required_capability)

    @property
    def next_tier_progress(self) -> dict:
        """Report progress toward next tier promotion."""
        current = self.state.current_tier
        if current >= AutonomyTier.TRUSTED:
            return {"status": "max_tier_reached"}

        next_tier = AutonomyTier(current.value + 1)
        reqs = self.requirements[next_tier]
        now = self._clock()

        return {
            "current_tier": current.name,
            "next_tier": next_tier.name,
            "trust": f"{self.state.trust:.3f} / {reqs.min_trust:.3f}",
            "streak": f"{self.state.consecutive_passes} / {reqs.min_streak}",
            "cooldown_remaining": max(0, reqs.cooldown_seconds - (now - self.state.last_reset_time)),
            "trust_met": self.state.trust >= reqs.min_trust,
            "streak_met": self.state.consecutive_passes >= reqs.min_streak,
        }


# (NOT IMPLEMENTED: Capability-specific trust tracking (separate trust scores
#  per capability domain), multi-agent trust delegation chains, tier expiry
#  under inactivity, peer-attestation for tier promotion, and integration
#  with Ikigai filter for capability-action matching.)
