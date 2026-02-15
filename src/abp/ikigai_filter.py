# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Ikigai Filter: Four-Quadrant Action Validation Gate.

Inspired by the Japanese concept of ikigai (生き甲斐), this filter
validates proposed system actions against four orthogonal criteria.
ALL four quadrants must pass for an action to be approved.

Quadrants
---------
1. Objective Alignment (What the world needs)
   Does this action serve the stated objective?

2. Capability Match (What you are good at)
   Is the system capable of executing this action reliably?

3. Value Generation (What you can be paid for)
   Does this action generate measurable value for stakeholders?

4. Economic Viability (What you love → sustainability)
   Is this action economically sustainable long-term?

Design Rationale:
    Single-criterion optimization is the root cause of many AI alignment
    failures. An action that is technically capable but economically
    unsustainable, or value-generating but misaligned with objectives,
    is a failure mode. The four-quadrant gate prevents corner-cutting.

Reference:
    Just (2026), Section 5.5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class IkigaiQuadrant(Enum):
    """The four Ikigai validation quadrants."""
    OBJECTIVE_ALIGNMENT = "objective_alignment"   # What the world needs
    CAPABILITY_MATCH = "capability_match"         # What you are good at
    VALUE_GENERATION = "value_generation"          # What you can be paid for
    ECONOMIC_VIABILITY = "economic_viability"      # Sustainability


@dataclass
class QuadrantScore:
    """Score for a single quadrant evaluation.

    Attributes:
        quadrant: Which quadrant was evaluated.
        score: Score in [0.0, 1.0].
        passed: Whether the score met the threshold.
        threshold: The threshold used.
        rationale: Human-readable explanation.
    """
    quadrant: IkigaiQuadrant
    score: float
    passed: bool
    threshold: float
    rationale: str = ""


@dataclass
class IkigaiResult:
    """Complete Ikigai filter evaluation result.

    Attributes:
        approved: True only if ALL four quadrants pass.
        scores: Per-quadrant evaluation results.
        overall_score: Geometric mean of all quadrant scores.
        weakest_quadrant: The quadrant with the lowest score.
        action_description: Description of the evaluated action.
    """
    approved: bool
    scores: dict[IkigaiQuadrant, QuadrantScore]
    overall_score: float
    weakest_quadrant: IkigaiQuadrant
    action_description: str

    def summary(self) -> str:
        """Human-readable summary of the evaluation."""
        status = "APPROVED" if self.approved else "REJECTED"
        lines = [f"Ikigai Filter: {status} — {self.action_description}"]
        for q, s in self.scores.items():
            mark = "✓" if s.passed else "✗"
            lines.append(f"  [{mark}] {q.value}: {s.score:.2f}/{s.threshold:.2f} — {s.rationale}")
        lines.append(f"  Overall: {self.overall_score:.3f} | Weakest: {self.weakest_quadrant.value}")
        return "
".join(lines)


class IkigaiFilter:
    """Four-quadrant action validation gate.

    Each quadrant requires a scorer function that evaluates a proposed
    action and returns a score in [0, 1]. Actions must pass ALL four
    quadrants to be approved.

    Example:
        >>> filt = IkigaiFilter(
        ...     objective_scorer=lambda a: 0.9,
        ...     capability_scorer=lambda a: 0.85,
        ...     value_scorer=lambda a: 0.7,
        ...     economic_scorer=lambda a: 0.6,
        ... )
        >>> result = filt.evaluate("Deploy new feature")
        >>> result.approved
        True
    """

    def __init__(
        self,
        objective_scorer: Optional[Callable[[Any], float]] = None,
        capability_scorer: Optional[Callable[[Any], float]] = None,
        value_scorer: Optional[Callable[[Any], float]] = None,
        economic_scorer: Optional[Callable[[Any], float]] = None,
        objective_threshold: float = 0.5,
        capability_threshold: float = 0.5,
        value_threshold: float = 0.5,
        economic_threshold: float = 0.5,
    ):
        """Configure the Ikigai filter.

        Args:
            objective_scorer: Scores objective alignment [0, 1].
            capability_scorer: Scores capability match [0, 1].
            value_scorer: Scores value generation [0, 1].
            economic_scorer: Scores economic viability [0, 1].
            *_threshold: Minimum score for each quadrant to pass.
        """
        # Default scorers return neutral 0.5 if not configured
        self._scorers = {
            IkigaiQuadrant.OBJECTIVE_ALIGNMENT: objective_scorer or (lambda a: 0.5),
            IkigaiQuadrant.CAPABILITY_MATCH: capability_scorer or (lambda a: 0.5),
            IkigaiQuadrant.VALUE_GENERATION: value_scorer or (lambda a: 0.5),
            IkigaiQuadrant.ECONOMIC_VIABILITY: economic_scorer or (lambda a: 0.5),
        }
        self._thresholds = {
            IkigaiQuadrant.OBJECTIVE_ALIGNMENT: objective_threshold,
            IkigaiQuadrant.CAPABILITY_MATCH: capability_threshold,
            IkigaiQuadrant.VALUE_GENERATION: value_threshold,
            IkigaiQuadrant.ECONOMIC_VIABILITY: economic_threshold,
        }

    def evaluate(self, action: Any, description: str = "") -> IkigaiResult:
        """Evaluate a proposed action against all four quadrants.

        Args:
            action: The action or proposal to evaluate.
            description: Human-readable description of the action.

        Returns:
            IkigaiResult with pass/fail and per-quadrant scores.
        """
        scores: dict[IkigaiQuadrant, QuadrantScore] = {}
        min_score = float('inf')
        weakest = IkigaiQuadrant.OBJECTIVE_ALIGNMENT

        for quadrant in IkigaiQuadrant:
            raw_score = self._scorers[quadrant](action)
            raw_score = max(0.0, min(1.0, raw_score))  # Clamp
            threshold = self._thresholds[quadrant]
            passed = raw_score >= threshold

            scores[quadrant] = QuadrantScore(
                quadrant=quadrant,
                score=raw_score,
                passed=passed,
                threshold=threshold,
            )

            if raw_score < min_score:
                min_score = raw_score
                weakest = quadrant

        all_passed = all(s.passed for s in scores.values())

        # Geometric mean as overall score (penalizes imbalance)
        import numpy as np
        score_vals = [s.score for s in scores.values()]
        # Clamp to avoid log(0)
        score_vals_safe = [max(s, 1e-10) for s in score_vals]
        geometric_mean = float(np.exp(np.mean(np.log(score_vals_safe))))

        return IkigaiResult(
            approved=all_passed,
            scores=scores,
            overall_score=geometric_mean,
            weakest_quadrant=weakest,
            action_description=description or str(action),
        )

    def batch_evaluate(
        self,
        actions: list[tuple[Any, str]],
    ) -> list[IkigaiResult]:
        """Evaluate multiple actions.

        Args:
            actions: List of (action, description) tuples.

        Returns:
            List of IkigaiResult objects.
        """
        return [self.evaluate(a, d) for a, d in actions]

    def update_threshold(self, quadrant: IkigaiQuadrant, new_threshold: float):
        """Update the threshold for a specific quadrant.

        Args:
            quadrant: Which quadrant to update.
            new_threshold: New threshold value in [0, 1].
        """
        self._thresholds[quadrant] = max(0.0, min(1.0, new_threshold))
