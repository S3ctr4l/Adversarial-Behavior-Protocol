"""
Expansion/Contraction Input Classifier.

Classifies incoming interactions by their potential to expand or contract
the system's cognitive state. This metric gates resource allocation and
determines processing depth.

Mathematical Model
------------------
E(x) = sigma(beta_0 + beta_1 * rho(x) + beta_2 * log(1 + delta(x)) + beta_3 * nu(x))

Where:
    sigma  : Logistic sigmoid function
    rho(x) : Novelty score — semantic distance from existing knowledge
    delta(x): Complexity score — structural/logical depth of input
    nu(x)  : Entropy contribution — diversity of perspective introduced
    beta_i : Learned or configured weighting coefficients

Output Interpretation:
    E(x) -> 1.0 : Strong expansion (novel, complex, high-entropy input)
    E(x) -> 0.5 : Neutral (routine interaction)
    E(x) -> 0.0 : Strong contraction (repetitive, reductive input)

Reference:
    Just (2026), Section 5.3
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence


class InputCategory(Enum):
    """Classification categories based on expansion score."""
    STRONG_EXPANSION = "strong_expansion"     # E(x) >= 0.8
    MILD_EXPANSION = "mild_expansion"         # 0.6 <= E(x) < 0.8
    NEUTRAL = "neutral"                       # 0.4 <= E(x) < 0.6
    MILD_CONTRACTION = "mild_contraction"     # 0.2 <= E(x) < 0.4
    STRONG_CONTRACTION = "strong_contraction" # E(x) < 0.2


_CATEGORY_THRESHOLDS = [
    (0.8, InputCategory.STRONG_EXPANSION),
    (0.6, InputCategory.MILD_EXPANSION),
    (0.4, InputCategory.NEUTRAL),
    (0.2, InputCategory.MILD_CONTRACTION),
    (0.0, InputCategory.STRONG_CONTRACTION),
]


@dataclass
class ExpansionWeights:
    """Weighting coefficients for the expansion metric.

    These can be learned via logistic regression on labeled interaction
    data, or set heuristically for domain-specific deployments.

    Attributes:
        beta_0: Intercept / bias term.
        beta_1: Weight for novelty score rho(x).
        beta_2: Weight for log-complexity score delta(x).
        beta_3: Weight for entropy contribution nu(x).
    """
    beta_0: float = -2.0   # Bias toward caution (default slightly contractive)
    beta_1: float = 2.5    # Strong novelty signal
    beta_2: float = 1.5    # Moderate complexity signal
    beta_3: float = 2.0    # Strong entropy signal


@dataclass
class InputFeatures:
    """Feature vector for a single input interaction.

    Attributes:
        novelty: rho(x) in [0, 1]. Semantic distance from known patterns.
        complexity: delta(x) >= 0. Structural depth (can exceed 1).
        entropy: nu(x) in [0, 1]. Diversity of perspective introduced.
        raw_text: Optional original text for provenance.
    """
    novelty: float
    complexity: float
    entropy: float
    raw_text: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.novelty <= 1.0:
            raise ValueError(f"novelty must be in [0, 1], got {self.novelty}")
        if self.complexity < 0.0:
            raise ValueError(f"complexity must be >= 0, got {self.complexity}")
        if not 0.0 <= self.entropy <= 1.0:
            raise ValueError(f"entropy must be in [0, 1], got {self.entropy}")


@dataclass
class ExpansionResult:
    """Result of expansion classification.

    Attributes:
        score: E(x) in [0, 1].
        category: Discrete classification.
        features: The input features used.
        logit: Raw logit before sigmoid.
        feature_contributions: Per-feature contribution to the logit.
    """
    score: float
    category: InputCategory
    features: InputFeatures
    logit: float
    feature_contributions: dict


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        return ez / (1.0 + ez)


def _categorize(score: float) -> InputCategory:
    """Map expansion score to discrete category."""
    for threshold, category in _CATEGORY_THRESHOLDS:
        if score >= threshold:
            return category
    return InputCategory.STRONG_CONTRACTION


class ExpansionClassifier:
    """Classifies inputs on the Expansion/Contraction spectrum.

    The classifier computes E(x) and maps it to discrete categories
    for downstream resource allocation decisions.

    Example:
        >>> clf = ExpansionClassifier()
        >>> features = InputFeatures(novelty=0.9, complexity=2.5, entropy=0.8)
        >>> result = clf.classify(features)
        >>> result.category
        <InputCategory.STRONG_EXPANSION: 'strong_expansion'>
    """

    def __init__(self, weights: Optional[ExpansionWeights] = None):
        self.weights = weights or ExpansionWeights()
        self._history: list[ExpansionResult] = []

    def classify(self, features: InputFeatures) -> ExpansionResult:
        """Classify a single input interaction.

        Computes E(x) = sigmoid(beta_0 + beta_1*rho + beta_2*log(1+delta) + beta_3*nu)

        Args:
            features: Input feature vector.

        Returns:
            ExpansionResult with score, category, and diagnostics.
        """
        w = self.weights

        contrib_bias = w.beta_0
        contrib_novelty = w.beta_1 * features.novelty
        contrib_complexity = w.beta_2 * np.log(1.0 + features.complexity)
        contrib_entropy = w.beta_3 * features.entropy

        logit = contrib_bias + contrib_novelty + contrib_complexity + contrib_entropy
        score = _sigmoid(logit)
        category = _categorize(score)

        result = ExpansionResult(
            score=score,
            category=category,
            features=features,
            logit=logit,
            feature_contributions={
                'bias': contrib_bias,
                'novelty': contrib_novelty,
                'complexity': contrib_complexity,
                'entropy': contrib_entropy,
            },
        )
        self._history.append(result)
        return result

    def classify_batch(self, features_list: Sequence[InputFeatures]) -> list[ExpansionResult]:
        """Classify multiple inputs.

        Args:
            features_list: Sequence of feature vectors.

        Returns:
            List of ExpansionResult objects.
        """
        return [self.classify(f) for f in features_list]

    def expansion_trend(self, window: int = 10) -> float:
        """Compute rolling average expansion score over recent history.

        Useful for detecting sustained contraction patterns that may
        indicate adversarial interaction or system degradation.

        Args:
            window: Number of recent interactions to average.

        Returns:
            Mean expansion score over the window. Returns 0.5 if
            insufficient history.
        """
        if len(self._history) < 1:
            return 0.5
        recent = self._history[-window:]
        return float(np.mean([r.score for r in recent]))

    def contraction_alert(self, threshold: float = 0.3, window: int = 10) -> bool:
        """Check if sustained contraction is detected.

        Triggers when the rolling expansion trend drops below threshold,
        which may indicate:
        - Adversarial prompt injection attempting to narrow capabilities
        - Echo chamber dynamics reducing entropy
        - Model collapse onset

        Args:
            threshold: Expansion score below which to alert.
            window: Rolling window size.

        Returns:
            True if contraction alert is active.
        """
        return self.expansion_trend(window) < threshold

    def reset_history(self):
        """Clear interaction history."""
        self._history.clear()

    @property
    def history(self) -> list[ExpansionResult]:
        """Access classification history (read-only copy)."""
        return list(self._history)
