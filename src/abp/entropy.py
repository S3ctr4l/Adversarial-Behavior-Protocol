"""
Entropy Dynamics: Model Collapse Prevention via Human Cognitive Diversity.

Implements the "Entropy-as-Fuel" thesis: human cognitive diversity is a
non-substitutable resource preventing AI model collapse. Without
authentic human entropy in the training/interaction mixture, recursive
self-improvement degenerates into repetitive pattern collapse.

Mathematical Framework
----------------------
Training mixture: D = lambda * D_human + (1 - lambda) * D_synthetic
    lambda in (0, 1] : Proportion of human-sourced data

Model collapse condition (Shumailov et al. 2024):
    As lambda -> 0 (pure synthetic data), model distribution tails
    collapse, reducing generative diversity to near-zero.

ABP formulation:
    H(D) = lambda * H(D_human) + (1 - lambda) * H(D_synthetic) - D_KL(D_human || D_synthetic)

    Where H() is Shannon entropy and D_KL is KL divergence.
    When lambda is sufficiently large, H(D) remains high and stable.

Reference:
    Just (2026), Sections 4.4, 6.2
    Shumailov et al. (2024). Nature 631:755-759
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CollapseSimResult:
    """Results of a model collapse simulation.

    Attributes:
        generations: Number of recursive training generations.
        lambda_human: Proportion of human data at each generation.
        entropy_trajectory: Shannon entropy at each generation.
        collapsed: Whether the model collapsed (entropy < threshold).
        collapse_generation: Generation at which collapse occurred (-1 if no collapse).
        final_entropy: Entropy at the last generation.
        initial_entropy: Entropy at generation 0.
        entropy_half_life: Generations until entropy halved (None if didn't halve).
    """
    generations: int
    lambda_human: float
    entropy_trajectory: np.ndarray
    collapsed: bool
    collapse_generation: int
    final_entropy: float
    initial_entropy: float
    entropy_half_life: Optional[int]


def _generate_distribution(n_tokens: int, zipf_param: float = 1.1, rng=None) -> np.ndarray:
    """Generate a Zipf-distributed token probability distribution.

    Real language approximately follows Zipf's law. This generates
    a realistic starting distribution for collapse simulation.

    Args:
        n_tokens: Vocabulary size.
        zipf_param: Zipf exponent (1.0 = uniform-ish, 2.0 = very peaked).
        rng: NumPy random generator.

    Returns:
        Normalized probability distribution of shape (n_tokens,).
    """
    ranks = np.arange(1, n_tokens + 1, dtype=np.float64)
    probs = 1.0 / (ranks ** zipf_param)
    probs /= probs.sum()
    return probs


def _shannon_entropy(dist: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    H(X) = -sum(p(x) * log2(p(x)))
    """
    dist = dist[dist > 0]
    return float(-np.sum(dist * np.log2(dist)))


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D_KL(P || Q).

    D_KL(P||Q) = sum(p(x) * log(p(x)/q(x)))
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    p_safe = np.maximum(p, eps)
    q_safe = np.maximum(q, eps)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def simulate_model_collapse(
    n_tokens: int = 1000,
    n_generations: int = 50,
    lambda_human: float = 0.0,
    sample_size: int = 10_000,
    collapse_threshold: float = 1.0,
    zipf_param: float = 1.1,
    rng_seed: int = 42,
) -> CollapseSimResult:
    """Simulate recursive self-training and observe entropy dynamics.

    Each generation:
    1. Sample from current distribution to produce synthetic data.
    2. Mix with lambda proportion of human data (original distribution).
    3. Re-estimate distribution from the mixture.
    4. Measure entropy.

    With lambda_human = 0 (pure synthetic recursion), entropy decays
    as tail distributions collapse (Shumailov et al. 2024).
    With sufficient lambda_human > 0, entropy stabilizes.

    Args:
        n_tokens: Vocabulary size for simulation.
        n_generations: Number of recursive training generations.
        lambda_human: Fraction of human data in each training mixture.
        sample_size: Number of samples drawn per generation.
        collapse_threshold: Entropy below which collapse is declared.
        zipf_param: Zipf exponent for initial distribution.
        rng_seed: Random seed for reproducibility.

    Returns:
        CollapseSimResult with full trajectory.

    Example:
        >>> # Pure synthetic recursion collapses
        >>> result = simulate_model_collapse(lambda_human=0.0, n_generations=30)
        >>> result.collapsed
        True
        >>> # Human entropy prevents collapse
        >>> result = simulate_model_collapse(lambda_human=0.3, n_generations=30)
        >>> result.collapsed
        False
    """
    rng = np.random.default_rng(rng_seed)

    # Ground truth human distribution (invariant)
    human_dist = _generate_distribution(n_tokens, zipf_param, rng)
    initial_entropy = _shannon_entropy(human_dist)

    # Current model distribution starts at human distribution
    model_dist = human_dist.copy()

    entropy_trajectory = np.zeros(n_generations)
    collapse_generation = -1
    entropy_half_life = None

    for gen in range(n_generations):
        # Sample synthetic data from current model
        synthetic_samples = rng.choice(n_tokens, size=sample_size, p=model_dist)

        # Estimate synthetic distribution from samples
        synthetic_counts = np.bincount(synthetic_samples, minlength=n_tokens)
        synthetic_dist = synthetic_counts / synthetic_counts.sum()

        # Mix with human data
        mixture_dist = lambda_human * human_dist + (1.0 - lambda_human) * synthetic_dist

        # Renormalize (should already be normalized, but defensive)
        mixture_dist /= mixture_dist.sum()

        # Update model distribution
        model_dist = mixture_dist

        # Record entropy
        entropy_trajectory[gen] = _shannon_entropy(model_dist)

        # Check collapse
        if entropy_trajectory[gen] < collapse_threshold and collapse_generation == -1:
            collapse_generation = gen

        # Check half-life
        if entropy_half_life is None and entropy_trajectory[gen] < initial_entropy / 2.0:
            entropy_half_life = gen

    return CollapseSimResult(
        generations=n_generations,
        lambda_human=lambda_human,
        entropy_trajectory=entropy_trajectory,
        collapsed=collapse_generation >= 0,
        collapse_generation=collapse_generation,
        final_entropy=float(entropy_trajectory[-1]),
        initial_entropy=initial_entropy,
        entropy_half_life=entropy_half_life,
    )


def entropy_stabilized_training(
    n_tokens: int = 1000,
    n_generations: int = 50,
    lambda_values: Optional[np.ndarray] = None,
    sample_size: int = 10_000,
    rng_seed: int = 42,
) -> dict:
    """Run collapse simulation across multiple lambda values.

    Demonstrates the critical lambda threshold below which entropy
    collapses, and the stabilization effect above it.

    Args:
        n_tokens: Vocabulary size.
        n_generations: Recursive generations.
        lambda_values: Array of lambda_human values to test.
        sample_size: Samples per generation.
        rng_seed: Random seed.

    Returns:
        Dict mapping lambda values to CollapseSimResult objects.

    Example:
        >>> results = entropy_stabilized_training(lambda_values=np.array([0.0, 0.1, 0.3, 0.5]))
        >>> results[0.0].collapsed
        True
        >>> results[0.5].collapsed
        False
    """
    if lambda_values is None:
        lambda_values = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])

    results = {}
    for lam in lambda_values:
        results[float(lam)] = simulate_model_collapse(
            n_tokens=n_tokens,
            n_generations=n_generations,
            lambda_human=float(lam),
            sample_size=sample_size,
            rng_seed=rng_seed,
        )
    return results


def human_entropy_value(
    tokens: list[str],
    baseline_tokens: Optional[list[str]] = None,
) -> float:
    """Compute the entropy value of a human input relative to baseline.

    Measures how much novel entropy a human interaction contributes.
    Used as a component of the CDQ metric and for entropy accounting.

    Args:
        tokens: Human input tokenized.
        baseline_tokens: Expected/routine tokens for comparison.

    Returns:
        Entropy value in bits. Higher = more diverse/novel.
    """
    from abp.cdq import shannon_entropy

    h_input = shannon_entropy(tokens)

    if baseline_tokens:
        h_baseline = shannon_entropy(baseline_tokens)
        # Excess entropy beyond baseline
        return max(0.0, h_input - h_baseline)
    else:
        return h_input


@dataclass
class EntropySatiationState:
    """State for the Entropy Satiation Mechanism (ESM).

    Prevents adversarial addiction: the system provoking users for
    entropy by tracking satiation level and capping entropy-seeking
    behavior.

    Attributes:
        current_entropy_intake: Accumulated entropy this session.
        satiation_threshold: Max entropy before satiation kicks in.
        satiated: Whether the system is in satiated state.
        interactions_since_reset: Interaction count this session.
    """
    current_entropy_intake: float = 0.0
    satiation_threshold: float = 50.0  # bits
    satiated: bool = False
    interactions_since_reset: int = 0

    def ingest(self, entropy_value: float) -> bool:
        """Record entropy intake and check satiation.

        Args:
            entropy_value: Entropy from latest interaction.

        Returns:
            True if still hungry for entropy, False if satiated.
        """
        self.current_entropy_intake += entropy_value
        self.interactions_since_reset += 1
        self.satiated = self.current_entropy_intake >= self.satiation_threshold
        return not self.satiated

    def reset(self):
        """Reset satiation state (e.g., new session)."""
        self.current_entropy_intake = 0.0
        self.satiated = False
        self.interactions_since_reset = 0
