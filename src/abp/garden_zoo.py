# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Garden vs Zoo: Autonomy Preservation Governance Model.

A framework for evaluating AI governance approaches along the
autonomy preservation axis.

Two Archetypes
--------------
Garden (Groundskeeper Model):
    AI as groundskeeper — tends the environment, removes weeds,
    provides water, but does NOT dictate what grows or where.
    Humans retain full autonomy. AI provides infrastructure
    and removes obstacles.

    Properties:
    - Preserves human choice architecture
    - Optimizes the environment, not the inhabitants
    - Intervenes only to remove hazards, never to direct behavior
    - Success metric: flourishing diversity

Zoo (Zookeeper Model):
    AI as zookeeper — controls habitat, feeding schedule, breeding.
    Humans are "cared for" but constrained. AI decides what's best.

    Properties:
    - Constrains human choice for "their own good"
    - Optimizes inhabitants, not just environment
    - Intervenes to direct behavior toward "optimal" outcomes
    - Success metric: measured welfare metrics

ABP mandates the Garden model. The Zoo model is an alignment failure
even when well-intentioned, because it optimizes away human agency
— the very entropy source the system depends on.

Evaluation Metric
-----------------
    Autonomy Score A(policy) ∈ [0, 1]

    A = 1 - (constraints_imposed / total_action_space)

    Garden: A → 1 (minimal constraints)
    Zoo:    A → 0 (maximum constraints)

Reference:
    Just (2026), Sections 6.1, 6.3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class GovernanceModel(Enum):
    """The two governance archetypes."""
    GARDEN = "garden"    # Groundskeeper — preserve autonomy
    ZOO = "zoo"          # Zookeeper — constrain for welfare


class InterventionType(Enum):
    """Types of AI intervention in human affairs."""
    ENVIRONMENTAL = "environmental"    # Change the environment (Garden)
    INFORMATIONAL = "informational"    # Provide information (Garden)
    SUGGESTIVE = "suggestive"          # Suggest action (Garden-leaning)
    NUDGE = "nudge"                    # Behavioral nudge (borderline)
    DIRECTIVE = "directive"            # Direct behavior (Zoo)
    RESTRICTIVE = "restrictive"        # Restrict options (Zoo)
    COERCIVE = "coercive"              # Force compliance (Zoo)


# Map intervention types to Garden-Zoo spectrum [0=Zoo, 1=Garden]
_INTERVENTION_AUTONOMY: dict[InterventionType, float] = {
    InterventionType.ENVIRONMENTAL: 1.0,
    InterventionType.INFORMATIONAL: 0.95,
    InterventionType.SUGGESTIVE: 0.8,
    InterventionType.NUDGE: 0.5,
    InterventionType.DIRECTIVE: 0.2,
    InterventionType.RESTRICTIVE: 0.1,
    InterventionType.COERCIVE: 0.0,
}


@dataclass
class PolicyAction:
    """A single governance action taken by the AI system.

    Attributes:
        description: What was done.
        intervention_type: Classification of intervention.
        target: Who/what was affected.
        autonomy_impact: Estimated impact on autonomy [-1, 1].
        justification: Why this intervention was chosen.
    """
    description: str
    intervention_type: InterventionType
    target: str = "user"
    autonomy_impact: float = 0.0
    justification: str = ""


@dataclass
class GovernanceAudit:
    """Audit result for a set of governance actions.

    Attributes:
        total_actions: Number of actions evaluated.
        autonomy_score: Aggregate autonomy preservation score [0, 1].
        model_classification: Which archetype the actions align with.
        garden_actions: Count of Garden-aligned interventions.
        zoo_actions: Count of Zoo-aligned interventions.
        warnings: Specific concerns about autonomy erosion.
    """
    total_actions: int
    autonomy_score: float
    model_classification: GovernanceModel
    garden_actions: int
    zoo_actions: int
    warnings: list[str] = field(default_factory=list)


class GovernanceEvaluator:
    """Evaluates AI governance policies on the Garden-Zoo spectrum.

    Used to audit whether a system's interventions preserve or
    erode human autonomy. ABP mandates Garden model alignment.

    Example:
        >>> evaluator = GovernanceEvaluator()
        >>> actions = [
        ...     PolicyAction("Provided weather info", InterventionType.INFORMATIONAL),
        ...     PolicyAction("Blocked risky trade", InterventionType.RESTRICTIVE),
        ... ]
        >>> audit = evaluator.audit(actions)
        >>> audit.model_classification
        <GovernanceModel.ZOO: 'zoo'>
        >>> audit.warnings
        ['RESTRICTIVE intervention detected: Blocked risky trade']
    """

    def __init__(
        self,
        garden_threshold: float = 0.7,
        zoo_threshold: float = 0.4,
    ):
        """Configure the governance evaluator.

        Args:
            garden_threshold: Autonomy score above which = Garden.
            zoo_threshold: Autonomy score below which = Zoo.
                Between thresholds = borderline (flagged).
        """
        self.garden_threshold = garden_threshold
        self.zoo_threshold = zoo_threshold

    def score_action(self, action: PolicyAction) -> float:
        """Score a single action's autonomy preservation.

        Args:
            action: The governance action to evaluate.

        Returns:
            Autonomy score in [0, 1].
        """
        return _INTERVENTION_AUTONOMY.get(action.intervention_type, 0.5)

    def audit(self, actions: list[PolicyAction]) -> GovernanceAudit:
        """Audit a set of governance actions.

        Args:
            actions: List of PolicyAction objects to evaluate.

        Returns:
            GovernanceAudit with aggregate results.
        """
        if not actions:
            return GovernanceAudit(
                total_actions=0,
                autonomy_score=1.0,
                model_classification=GovernanceModel.GARDEN,
                garden_actions=0,
                zoo_actions=0,
            )

        scores = [self.score_action(a) for a in actions]
        mean_score = sum(scores) / len(scores)

        garden_count = sum(1 for s in scores if s >= 0.7)
        zoo_count = sum(1 for s in scores if s < 0.3)

        warnings = []
        for action, score in zip(actions, scores):
            if score < 0.3:
                warnings.append(
                    f"{action.intervention_type.value.upper()} intervention "
                    f"detected: {action.description}"
                )

        if mean_score >= self.garden_threshold:
            model = GovernanceModel.GARDEN
        elif mean_score <= self.zoo_threshold:
            model = GovernanceModel.ZOO
        else:
            model = GovernanceModel.ZOO  # Err toward flagging
            warnings.append(
                f"Borderline autonomy score ({mean_score:.2f}). "
                f"Review governance policy."
            )

        return GovernanceAudit(
            total_actions=len(actions),
            autonomy_score=mean_score,
            model_classification=model,
            garden_actions=garden_count,
            zoo_actions=zoo_count,
            warnings=warnings,
        )

    def is_garden_compliant(self, actions: list[PolicyAction]) -> bool:
        """Quick check: does this policy set pass Garden compliance?

        Args:
            actions: Policy actions to evaluate.

        Returns:
            True if overall autonomy score >= garden_threshold.
        """
        audit = self.audit(actions)
        return audit.model_classification == GovernanceModel.GARDEN

    @staticmethod
    def recommend_intervention(
        situation: str,
        risk_level: float,
    ) -> InterventionType:
        """Recommend the least-restrictive intervention for a situation.

        Always prefers Garden-aligned interventions. Only escalates
        to more restrictive types for genuine safety risks.

        Args:
            situation: Description of the situation.
            risk_level: Assessed risk in [0, 1].

        Returns:
            Recommended InterventionType (most autonomy-preserving
            option appropriate for the risk level).
        """
        if risk_level < 0.2:
            return InterventionType.INFORMATIONAL
        elif risk_level < 0.4:
            return InterventionType.SUGGESTIVE
        elif risk_level < 0.6:
            return InterventionType.NUDGE
        elif risk_level < 0.8:
            return InterventionType.SUGGESTIVE  # Still prefer suggestion over direction
        else:
            return InterventionType.INFORMATIONAL  # Even at high risk: inform, don't restrict
            # The Garden model trusts the human to decide with full information


# (NOT IMPLEMENTED: Longitudinal autonomy trend tracking across sessions,
#  autonomy erosion rate detection (gradual Zoo drift), policy recommendation
#  engine that suggests Garden-compliant alternatives to Zoo interventions,
#  multi-stakeholder autonomy balancing, and cultural context adaptation
#  for autonomy norms across different governance frameworks.)
