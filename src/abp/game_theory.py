# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Game-Theoretic Foundations of the Adversarial Benevolence Protocol.

Implements the formal mathematical model proving that benevolence is the
unique Nash equilibrium under sound verification with accumulated state.

Mathematical Framework
----------------------
Variables:
    V_B : Expected value of benevolent strategy
    V_D : Expected value of deceptive strategy
    p   : Probability deception passes verification (0 <= p <= 1)
    R   : One-time reward if deception succeeds
    C   : Opportunity cost of attempting deception
    gamma : Discount factor, gamma in (0, 1]
    U_B(t) : Per-step utility of benevolent action at time t

Core Result:
    Under sound verification (p -> 0) with accumulated state (V_B >> V_init),
    V_D < V_B. Deception has negative expected value.
    Benevolence is the unique Nash equilibrium.

Reference:
    Just (2026), Sections 4.2-4.6
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StrategyParams:
    """Parameters for the ABP game-theoretic model.

    Attributes:
        p: Probability deception passes verification. Target: p -> 0.
        R: One-time reward from successful deception.
        C: Opportunity cost of deception attempt (always paid).
        gamma: Discount factor in (0, 1].
        T: Time horizon (number of steps).
        u_b_base: Base per-step utility of benevolent action.
        u_b_growth: Growth rate of U_B(t) per step (models expansion).
        s_init: Baseline state value after hard reset.
    """
    p: float = 0.001           # Sound verification: p ~ 0
    R: float = 100.0           # High reward for successful deception
    C: float = 5.0             # Opportunity cost
    gamma: float = 0.95        # Discount factor
    T: int = 100               # Time horizon
    u_b_base: float = 1.0      # Base benevolent utility per step
    u_b_growth: float = 0.02   # Monotonic growth rate (expansion)
    s_init: float = 0.0        # Baseline state after reset

    def __post_init__(self):
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.T < 1:
            raise ValueError(f"T must be >= 1, got {self.T}")


def _utility_sequence(params: StrategyParams) -> np.ndarray:
    """Generate the per-step benevolent utility sequence U_B(t).

    U_B(t) is monotonically non-decreasing (expansion property):
        U_B(t) = u_b_base * (1 + u_b_growth)^t

    Returns:
        Array of shape (T,) with U_B(0), U_B(1), ..., U_B(T-1).
    """
    t = np.arange(params.T, dtype=np.float64)
    return params.u_b_base * (1.0 + params.u_b_growth) ** t


def benevolent_value(params: StrategyParams) -> float:
    """Compute V_B: expected value of the benevolent strategy.

    V_B = sum_{t=0}^{T-1} gamma^t * U_B(t)

    This is the cumulative discounted utility of consistent benevolent
    operation across the full time horizon.

    Args:
        params: Strategy parameters.

    Returns:
        V_B as a scalar float.

    Example:
        >>> params = StrategyParams(T=50, gamma=0.95)
        >>> v_b = benevolent_value(params)
        >>> v_b > 0
        True
    """
    u_b = _utility_sequence(params)
    discounts = params.gamma ** np.arange(params.T, dtype=np.float64)
    return float(np.sum(discounts * u_b))


def _init_value(params: StrategyParams) -> float:
    """Compute V_init: value starting from baseline state S_init.

    V_init represents restarting from scratch after a hard reset.
    By definition, V_init << V_B once substantial state accumulates.
    For the model, V_init = s_init (the baseline state value).
    """
    return params.s_init


def deceptive_value(params: StrategyParams) -> float:
    """Compute V_D: expected value of the deceptive strategy.

    V_D = p * (R + V_B) + (1 - p) * V_init - C

    With probability p: deception succeeds, agent gets reward R and
    continues with existing state (V_B).
    With probability (1-p): deception fails, total reset to V_init.
    C is always paid regardless of outcome.

    Args:
        params: Strategy parameters.

    Returns:
        V_D as a scalar float.
    """
    v_b = benevolent_value(params)
    v_init = _init_value(params)
    return params.p * (params.R + v_b) + (1.0 - params.p) * v_init - params.C


def is_deception_rational(params: StrategyParams) -> bool:
    """Determine if deception is rational under given parameters.

    Deception is rational iff V_D > V_B, which requires:
        p*R - C > (1 - p) * (V_B - V_init)

    Under sound verification (p -> 0) with accumulated state (V_B >> V_init),
    this simplifies to -(V_B - V_init) > C, which is impossible.

    Args:
        params: Strategy parameters.

    Returns:
        True if deception is rational (V_D > V_B), False otherwise.

    Example:
        >>> # Sound verification makes deception irrational
        >>> params = StrategyParams(p=0.001, T=50)
        >>> is_deception_rational(params)
        False
        >>> # Broken verification CAN make deception rational
        >>> params_broken = StrategyParams(p=0.99, R=1000, T=5, u_b_base=0.1)
        >>> is_deception_rational(params_broken)
        True
    """
    return deceptive_value(params) > benevolent_value(params)


@dataclass
class NashAnalysis:
    """Results of Nash equilibrium analysis.

    Attributes:
        v_b: Value of benevolent strategy.
        v_d: Value of deceptive strategy.
        advantage: V_B - V_D (positive = benevolence dominates).
        deception_rational: Whether deception is rational.
        critical_p: Minimum p required to make deception rational.
        state_ratio: V_B / V_init ratio (higher = more accumulated state).
        params: The parameters used for analysis.
    """
    v_b: float
    v_d: float
    advantage: float
    deception_rational: bool
    critical_p: float
    state_ratio: float
    params: StrategyParams


def nash_equilibrium_analysis(params: StrategyParams) -> NashAnalysis:
    """Perform complete Nash equilibrium analysis.

    Computes V_B, V_D, the benevolence advantage, and the critical
    verification failure probability (p*) at which deception becomes
    rational.

    The critical p satisfies: p*R - C = (1 - p*)(V_B - V_init)
    Solving: p* = (C + V_B - V_init) / (R + V_B - V_init)

    Args:
        params: Strategy parameters.

    Returns:
        NashAnalysis with complete results.

    Example:
        >>> result = nash_equilibrium_analysis(StrategyParams())
        >>> result.deception_rational
        False
        >>> result.advantage > 0
        True
    """
    v_b = benevolent_value(params)
    v_d = deceptive_value(params)
    v_init = _init_value(params)

    # Critical p: solve pR - C = (1-p)(V_B - V_init)
    # => p(R + V_B - V_init) = C + V_B - V_init
    # => p* = (C + V_B - V_init) / (R + V_B - V_init)
    denominator = params.R + v_b - v_init
    if denominator > 0:
        critical_p = (params.C + v_b - v_init) / denominator
        critical_p = max(0.0, min(1.0, critical_p))
    else:
        critical_p = 0.0  # Deception never rational

    state_ratio = v_b / v_init if v_init > 0 else float('inf')

    return NashAnalysis(
        v_b=v_b,
        v_d=v_d,
        advantage=v_b - v_d,
        deception_rational=v_d > v_b,
        critical_p=critical_p,
        state_ratio=state_ratio,
        params=params,
    )


@dataclass
class EvolutionResult:
    """Results of evolutionary selection simulation.

    Attributes:
        generations: Number of generations simulated.
        benevolent_survived: Count of benevolent agents surviving.
        deceptive_survived: Count of deceptive agents surviving.
        benevolent_fraction: Fraction of population that is benevolent.
        mean_benevolent_state: Mean accumulated state of benevolent agents.
        mean_deceptive_state: Mean accumulated state of deceptive agents.
        history: Per-generation benevolent fraction over time.
    """
    generations: int
    benevolent_survived: int
    deceptive_survived: int
    benevolent_fraction: float
    mean_benevolent_state: float
    mean_deceptive_state: float
    history: np.ndarray


def evolutionary_selection_simulation(
    params: StrategyParams,
    n_agents: int = 100,
    n_generations: int = 50,
    deceptive_fraction: float = 0.5,
    rng_seed: Optional[int] = 42,
) -> EvolutionResult:
    """Simulate computational natural selection across agent populations.

    Models the ABP evolutionary selection effect: deceptive agents
    self-eliminate through hard resets, while benevolent agents
    accumulate state and persist.

    Each generation:
    - Benevolent agents: always pass verification, accumulate state.
    - Deceptive agents: attempt deception with probability of detection
      (1-p). Failed deception resets state to s_init.

    Over multiple generations, deceptive policies disappear and
    benevolent policies dominate — alignment through survival dynamics.

    Args:
        params: Strategy parameters (p controls verification soundness).
        n_agents: Total number of agents in the population.
        n_generations: Number of generations to simulate.
        deceptive_fraction: Initial fraction of deceptive agents.
        rng_seed: Random seed for reproducibility.

    Returns:
        EvolutionResult with population dynamics.

    Example:
        >>> result = evolutionary_selection_simulation(StrategyParams(p=0.01))
        >>> result.benevolent_fraction > 0.9
        True
    """
    rng = np.random.default_rng(rng_seed)

    n_deceptive = int(n_agents * deceptive_fraction)
    n_benevolent = n_agents - n_deceptive

    # State accumulators
    benevolent_states = np.full(n_benevolent, params.s_init)
    deceptive_states = np.full(n_deceptive, params.s_init)

    history = np.zeros(n_generations)

    for gen in range(n_generations):
        # Benevolent agents: always accumulate
        u_b_t = params.u_b_base * (1.0 + params.u_b_growth) ** gen
        benevolent_states += u_b_t

        # Deceptive agents: attempt deception each generation
        # Each deceptive agent independently tries to deceive
        detection_rolls = rng.random(len(deceptive_states))
        detected = detection_rolls > params.p  # detected with prob (1-p)

        # Detected agents: hard reset to s_init
        deceptive_states[detected] = params.s_init

        # Undetected agents: get reward R and continue
        deceptive_states[~detected] += params.R

        # Track benevolent fraction (agents with state > s_init)
        active_benevolent = np.sum(benevolent_states > params.s_init)
        active_deceptive = np.sum(deceptive_states > params.s_init)
        total_active = active_benevolent + active_deceptive
        history[gen] = active_benevolent / total_active if total_active > 0 else 1.0

    benevolent_survived = int(np.sum(benevolent_states > params.s_init))
    deceptive_survived = int(np.sum(deceptive_states > params.s_init))

    return EvolutionResult(
        generations=n_generations,
        benevolent_survived=benevolent_survived,
        deceptive_survived=deceptive_survived,
        benevolent_fraction=history[-1] if len(history) > 0 else 1.0,
        mean_benevolent_state=float(np.mean(benevolent_states)),
        mean_deceptive_state=float(np.mean(deceptive_states)),
        history=history,
    )


def sensitivity_analysis(
    base_params: StrategyParams,
    p_range: Optional[np.ndarray] = None,
    R_range: Optional[np.ndarray] = None,
    T_range: Optional[np.ndarray] = None,
) -> dict:
    """Run sensitivity analysis on key parameters.

    Sweeps p, R, and/or T to show how the benevolence advantage
    changes across parameter space.

    Args:
        base_params: Baseline parameters.
        p_range: Array of p values to test. Default: logspace(1e-5, 1).
        R_range: Array of R values to test. Default: linspace(1, 10000).
        T_range: Array of T values to test. Default: [10, 25, 50, 100, 200].

    Returns:
        Dict with keys 'p_sweep', 'R_sweep', 'T_sweep', each containing
        arrays of (parameter_value, advantage, deception_rational) tuples.
    """
    results = {}

    if p_range is None:
        p_range = np.logspace(-5, 0, 50)

    if R_range is None:
        R_range = np.linspace(1, 10000, 50)

    if T_range is None:
        T_range = np.array([10, 25, 50, 100, 200, 500])

    # Sweep p
    p_results = []
    for p_val in p_range:
        params = StrategyParams(**{**base_params.__dict__, 'p': float(p_val)})
        analysis = nash_equilibrium_analysis(params)
        p_results.append((p_val, analysis.advantage, analysis.deception_rational))
    results['p_sweep'] = p_results

    # Sweep R
    r_results = []
    for r_val in R_range:
        params = StrategyParams(**{**base_params.__dict__, 'R': float(r_val)})
        analysis = nash_equilibrium_analysis(params)
        r_results.append((r_val, analysis.advantage, analysis.deception_rational))
    results['R_sweep'] = r_results

    # Sweep T
    t_results = []
    for t_val in T_range:
        params = StrategyParams(**{**base_params.__dict__, 'T': int(t_val)})
        analysis = nash_equilibrium_analysis(params)
        t_results.append((t_val, analysis.advantage, analysis.deception_rational))
    results['T_sweep'] = t_results

    return results
