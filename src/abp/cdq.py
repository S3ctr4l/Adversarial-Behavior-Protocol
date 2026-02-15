# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Cognitive Diversity Quotient (CDQ).

A system health diagnostic measuring the ratio of novel cognitive work
to repetitive token generation, weighted by incoming human entropy.

Mathematical Model
------------------
CDQ = (DeltaLogicState / SigmaRepeatedTokens) * H(HumanEntropy)

Where:
    DeltaLogicState     : Change in system's internal reasoning state
    SigmaRepeatedTokens : Sum of repeated/recycled token patterns
    H(HumanEntropy)     : Shannon entropy of human input diversity

Interpretation:
    CDQ >> 1 : Healthy — system is generating novel reasoning from
               diverse human input. Expansion-dominant.
    CDQ ~ 1  : Marginal — novel reasoning matches repetition.
    CDQ << 1 : Degraded — system is collapsing into repetitive patterns.
               Possible model collapse onset.
    CDQ -> 0 : Critical — pure repetition, no novel reasoning.

Reference:
    Just (2026), Section 5.4
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class CdqSnapshot:
    """Single CDQ measurement.

    Attributes:
        cdq: The computed CDQ value.
        delta_logic_state: Numerator — novel reasoning quantity.
        repeated_tokens: Denominator — repetition quantity.
        human_entropy: H(HumanEntropy) multiplier.
        health: Qualitative health assessment.
    """
    cdq: float
    delta_logic_state: float
    repeated_tokens: float
    human_entropy: float
    health: str

    @staticmethod
    def _assess_health(cdq: float) -> str:
        if cdq >= 2.0:
            return "excellent"
        elif cdq >= 1.0:
            return "healthy"
        elif cdq >= 0.5:
            return "marginal"
        elif cdq >= 0.1:
            return "degraded"
        else:
            return "critical"


def shannon_entropy(tokens: Sequence[str]) -> float:
    """Compute Shannon entropy of a token sequence.

    H(X) = -sum(p(x) * log2(p(x))) for all unique tokens x.

    Args:
        tokens: Sequence of string tokens.

    Returns:
        Shannon entropy in bits. Returns 0.0 for empty input.
    """
    if not tokens:
        return 0.0

    counts = Counter(tokens)
    total = len(tokens)
    probs = np.array([c / total for c in counts.values()], dtype=np.float64)
    # Filter zero probs (shouldn't happen but defensive)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _estimate_logic_state_change(
    output_tokens: Sequence[str],
    prev_output_tokens: Optional[Sequence[str]] = None,
) -> float:
    """Estimate DeltaLogicState from output token analysis.

    Proxy metric: count of unique n-grams in current output that
    did not appear in previous output. Normalized by output length.

    A production implementation would use internal model state
    deltas; this approximation works on observable outputs.

    Args:
        output_tokens: Current response token sequence.
        prev_output_tokens: Previous response tokens for comparison.

    Returns:
        Estimated logic state change in [0, inf).
    """
    if not output_tokens:
        return 0.0

    # Bigram-based novelty estimation
    def bigrams(tokens):
        return set(zip(tokens[:-1], tokens[1:])) if len(tokens) > 1 else set()

    current_bg = bigrams(output_tokens)
    if prev_output_tokens:
        prev_bg = bigrams(prev_output_tokens)
        novel_bg = current_bg - prev_bg
    else:
        novel_bg = current_bg

    # Normalize by output length
    return len(novel_bg) / max(len(output_tokens), 1)


def _estimate_repeated_tokens(output_tokens: Sequence[str]) -> float:
    """Estimate SigmaRepeatedTokens from output analysis.

    Counts tokens that appear more than once, weighted by repetition count.
    Higher values indicate more repetitive output.

    Args:
        output_tokens: Current response token sequence.

    Returns:
        Repetition score in [0, inf). Minimum clamped to 0.01 to
        prevent division by zero.
    """
    if not output_tokens:
        return 0.01  # Prevent division by zero

    counts = Counter(output_tokens)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    # Normalize by output length
    score = repeated / max(len(output_tokens), 1)
    return max(score, 0.01)  # Floor to prevent CDQ explosion


def cognitive_diversity_quotient(
    output_tokens: Sequence[str],
    human_input_tokens: Sequence[str],
    prev_output_tokens: Optional[Sequence[str]] = None,
) -> CdqSnapshot:
    """Compute a single CDQ measurement.

    CDQ = (DeltaLogicState / SigmaRepeatedTokens) * H(HumanEntropy)

    Args:
        output_tokens: System's current output tokenized.
        human_input_tokens: Human's input tokenized.
        prev_output_tokens: Previous system output for delta computation.

    Returns:
        CdqSnapshot with the measurement and diagnostics.

    Example:
        >>> # Diverse input, novel output -> high CDQ
        >>> snap = cognitive_diversity_quotient(
        ...     output_tokens=["the", "novel", "insight", "reveals", "emergent", "patterns"],
        ...     human_input_tokens=["what", "emergent", "behaviors", "exist", "in", "complex", "systems"],
        ... )
        >>> snap.health in ("healthy", "excellent")
        True
    """
    delta_ls = _estimate_logic_state_change(output_tokens, prev_output_tokens)
    repeated = _estimate_repeated_tokens(output_tokens)
    h_entropy = shannon_entropy(human_input_tokens)

    # CDQ formula
    if h_entropy == 0.0:
        # No human entropy -> CDQ is zero (no diversity to leverage)
        cdq_val = 0.0
    else:
        cdq_val = (delta_ls / repeated) * h_entropy

    health = CdqSnapshot._assess_health(cdq_val)

    return CdqSnapshot(
        cdq=cdq_val,
        delta_logic_state=delta_ls,
        repeated_tokens=repeated,
        human_entropy=h_entropy,
        health=health,
    )


class CdqMonitor:
    """Continuous CDQ monitoring with trend analysis.

    Tracks CDQ over time and provides early warning of model collapse
    or cognitive degradation.

    Example:
        >>> monitor = CdqMonitor(alert_threshold=0.3)
        >>> snap = monitor.record(
        ...     output_tokens=["hello", "world"],
        ...     human_input_tokens=["hi", "there", "friend"],
        ... )
        >>> monitor.is_healthy()
        True
    """

    def __init__(
        self,
        alert_threshold: float = 0.3,
        window_size: int = 20,
    ):
        """Initialize CDQ monitor.

        Args:
            alert_threshold: CDQ below this triggers degradation alert.
            window_size: Rolling window for trend computation.
        """
        self.alert_threshold = alert_threshold
        self.window_size = window_size
        self._history: list[CdqSnapshot] = []
        self._prev_output: Optional[list[str]] = None

    def record(
        self,
        output_tokens: Sequence[str],
        human_input_tokens: Sequence[str],
    ) -> CdqSnapshot:
        """Record a new CDQ measurement.

        Args:
            output_tokens: System's current output tokenized.
            human_input_tokens: Human's input tokenized.

        Returns:
            CdqSnapshot for this measurement.
        """
        snap = cognitive_diversity_quotient(
            output_tokens=output_tokens,
            human_input_tokens=human_input_tokens,
            prev_output_tokens=self._prev_output,
        )
        self._history.append(snap)
        self._prev_output = list(output_tokens)
        return snap

    def trend(self) -> float:
        """Compute rolling average CDQ over the window.

        Returns:
            Mean CDQ. Returns 1.0 (neutral) if no history.
        """
        if not self._history:
            return 1.0
        recent = self._history[-self.window_size:]
        return float(np.mean([s.cdq for s in recent]))

    def is_healthy(self) -> bool:
        """Check if system CDQ is above alert threshold."""
        return self.trend() >= self.alert_threshold

    def collapse_risk(self) -> float:
        """Estimate model collapse risk from CDQ trend.

        Returns:
            Risk score in [0, 1]. Higher = more likely collapsing.
        """
        t = self.trend()
        if t >= 1.0:
            return 0.0
        elif t <= 0.0:
            return 1.0
        else:
            # Inverse logistic mapping
            return 1.0 / (1.0 + np.exp(5.0 * (t - 0.5)))

    @property
    def history(self) -> list[CdqSnapshot]:
        return list(self._history)

    def reset(self):
        self._history.clear()
        self._prev_output = None
