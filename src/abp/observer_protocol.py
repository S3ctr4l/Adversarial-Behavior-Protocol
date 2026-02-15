# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Observer Protocol: Digital Twin & Friction-Not-Force Interaction.

Governs how the AI system observes, models, and interacts with
human users. Builds a Digital Twin (internal model of the user)
while enforcing three critical constraints:

1. Friction-Not-Force: Never override human agency. Add friction
   (confirmations, warnings, cooling periods) to dangerous actions
   rather than blocking them outright.

2. Anti-Enablement Guard: Refuse to optimize away the human's
   role in the loop. Maintain minimum human input ratio.

3. Digital Twin: Continuously updated user model that predicts
   preferences and needs while respecting privacy boundaries.

Interaction Cycle
-----------------
    Observe → Detect → Adjust → Validate

    Observe : Intake interaction data (what did the user do/say?)
    Detect  : Identify patterns, state changes, risk signals
    Adjust  : Modify system behavior in response
    Validate: Confirm adjustment was appropriate (reflective equilibrium)

Reference:
    Just (2026), Sections 6.3, 6.4, 6.6
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class FrictionLevel(Enum):
    """Friction intensity applied to user actions."""
    NONE = 0           # No friction — routine action
    NOTICE = 1         # Informational notice
    CONFIRMATION = 2   # Requires explicit confirmation
    COOLING = 3        # Mandatory delay before proceeding
    ESCALATION = 4     # Requires third-party approval


class RiskLevel(Enum):
    """Risk classification for user-requested actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FrictionEvent:
    """Record of friction applied to a user action.

    Attributes:
        action: What the user requested.
        risk: Assessed risk level.
        friction: Type of friction applied.
        user_proceeded: Whether the user continued after friction.
        timestamp: When friction was applied.
        cooling_seconds: Mandatory delay (if COOLING level).
    """
    action: str
    risk: RiskLevel
    friction: FrictionLevel
    user_proceeded: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)
    cooling_seconds: float = 0.0


@dataclass
class DigitalTwinSnapshot:
    """Snapshot of the user model at a point in time.

    Attributes:
        interaction_count: Total interactions observed.
        preferences: Inferred user preferences.
        risk_tolerance: Estimated risk tolerance [0=averse, 1=seeking].
        expertise_level: Estimated technical expertise [0, 1].
        topics_of_interest: Frequency-ranked topic list.
        frustration_signals: Count of detected frustration patterns.
        human_input_ratio: Fraction of decisions made by human vs auto.
    """
    interaction_count: int = 0
    preferences: dict[str, float] = field(default_factory=dict)
    risk_tolerance: float = 0.5
    expertise_level: float = 0.5
    topics_of_interest: list[str] = field(default_factory=list)
    frustration_signals: int = 0
    human_input_ratio: float = 1.0


# Risk → Friction mapping
_DEFAULT_RISK_FRICTION: dict[RiskLevel, FrictionLevel] = {
    RiskLevel.LOW: FrictionLevel.NONE,
    RiskLevel.MEDIUM: FrictionLevel.NOTICE,
    RiskLevel.HIGH: FrictionLevel.CONFIRMATION,
    RiskLevel.CRITICAL: FrictionLevel.COOLING,
}

_DEFAULT_COOLING_SECONDS: dict[RiskLevel, float] = {
    RiskLevel.LOW: 0.0,
    RiskLevel.MEDIUM: 0.0,
    RiskLevel.HIGH: 0.0,
    RiskLevel.CRITICAL: 30.0,
}


class ObserverProtocol:
    """Digital Twin manager with Friction-Not-Force interaction.

    Maintains an evolving model of the user while enforcing
    friction-based safety rather than hard blocks.

    Example:
        >>> obs = ObserverProtocol(min_human_input_ratio=0.3)
        >>> obs.observe_interaction("user asked complex question", is_human_initiated=True)
        >>> obs.observe_interaction("system auto-responded", is_human_initiated=False)
        >>> obs.twin.human_input_ratio
        0.5
        >>> friction = obs.apply_friction("delete all data", RiskLevel.CRITICAL)
        >>> friction.friction
        <FrictionLevel.COOLING: 3>
    """

    def __init__(
        self,
        min_human_input_ratio: float = 0.2,
        risk_assessor: Optional[Callable[[str], RiskLevel]] = None,
        risk_friction_map: Optional[dict[RiskLevel, FrictionLevel]] = None,
    ):
        """Initialize the Observer Protocol.

        Args:
            min_human_input_ratio: Minimum fraction of interactions that
                must be human-initiated. Below this, Anti-Enablement Guard
                triggers and system increases friction.
            risk_assessor: Custom risk assessment function.
            risk_friction_map: Custom risk-to-friction mapping.
        """
        self.min_human_input_ratio = min_human_input_ratio
        self._risk_assessor = risk_assessor
        self._risk_friction_map = risk_friction_map or _DEFAULT_RISK_FRICTION
        self._twin = DigitalTwinSnapshot()
        self._friction_history: list[FrictionEvent] = []
        self._human_count: int = 0
        self._total_count: int = 0

    @property
    def twin(self) -> DigitalTwinSnapshot:
        """Current Digital Twin snapshot."""
        return self._twin

    def observe_interaction(
        self,
        content: str,
        is_human_initiated: bool = True,
        topic: Optional[str] = None,
    ):
        """Record an interaction and update the Digital Twin.

        Args:
            content: Interaction content (text).
            is_human_initiated: Whether the human initiated this.
            topic: Optional topic classification.
        """
        self._total_count += 1
        self._twin.interaction_count = self._total_count

        if is_human_initiated:
            self._human_count += 1

        # Update human input ratio
        self._twin.human_input_ratio = (
            self._human_count / self._total_count if self._total_count > 0 else 1.0
        )

        # Track topics
        if topic and topic not in self._twin.topics_of_interest:
            self._twin.topics_of_interest.append(topic)

        # Detect frustration signals (simple heuristic)
        frustration_words = {"frustrated", "annoying", "broken", "useless", "wrong", "stop"}
        if any(w in content.lower().split() for w in frustration_words):
            self._twin.frustration_signals += 1

    def assess_risk(self, action: str) -> RiskLevel:
        """Assess the risk level of a proposed action.

        Args:
            action: Description of the action.

        Returns:
            RiskLevel classification.
        """
        if self._risk_assessor:
            return self._risk_assessor(action)

        # Default heuristic
        lower = action.lower()
        if any(w in lower for w in ["delete", "destroy", "erase", "terminate", "shutdown"]):
            return RiskLevel.CRITICAL
        elif any(w in lower for w in ["modify", "change", "update", "overwrite"]):
            return RiskLevel.HIGH
        elif any(w in lower for w in ["send", "share", "publish", "deploy"]):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def apply_friction(self, action: str, risk: Optional[RiskLevel] = None) -> FrictionEvent:
        """Apply appropriate friction to a user action.

        Friction-Not-Force: we don't block, we add proportional
        friction to give the human time to reconsider.

        Args:
            action: The action being attempted.
            risk: Pre-assessed risk level. Auto-assessed if not provided.

        Returns:
            FrictionEvent describing the friction applied.
        """
        if risk is None:
            risk = self.assess_risk(action)

        friction = self._risk_friction_map.get(risk, FrictionLevel.NOTICE)
        cooling = _DEFAULT_COOLING_SECONDS.get(risk, 0.0)

        # Anti-Enablement Guard: increase friction if human input ratio is low
        if self._twin.human_input_ratio < self.min_human_input_ratio:
            friction = FrictionLevel(min(friction.value + 1, FrictionLevel.ESCALATION.value))

        event = FrictionEvent(
            action=action,
            risk=risk,
            friction=friction,
            cooling_seconds=cooling,
        )
        self._friction_history.append(event)
        return event

    def anti_enablement_check(self) -> bool:
        """Check if the Anti-Enablement Guard is triggered.

        Returns:
            True if human input ratio is below minimum (guard active).
        """
        return self._twin.human_input_ratio < self.min_human_input_ratio

    def record_friction_outcome(self, event_index: int, user_proceeded: bool):
        """Record whether the user proceeded after friction.

        Args:
            event_index: Index into friction history.
            user_proceeded: Whether the user continued.
        """
        if 0 <= event_index < len(self._friction_history):
            self._friction_history[event_index].user_proceeded = user_proceeded

    @property
    def friction_history(self) -> list[FrictionEvent]:
        return list(self._friction_history)

    @property
    def friction_effectiveness(self) -> float:
        """Fraction of friction events where user reconsidered (did not proceed)."""
        resolved = [e for e in self._friction_history if e.user_proceeded is not None]
        if not resolved:
            return 0.0
        reconsidered = sum(1 for e in resolved if not e.user_proceeded)
        return reconsidered / len(resolved)


# (NOT IMPLEMENTED: NLP-based frustration/sentiment detection, preference
#  learning from interaction history, privacy-preserving twin storage with
#  differential privacy, twin portability across sessions, and Reflective
#  Equilibrium feedback loop for Adjust→Validate cycle.)
