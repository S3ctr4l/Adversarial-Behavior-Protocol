# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
#!/usr/bin/env python3
"""
Generate sample datasets for ABP experiments.

Produces:
    - data/nash_equilibrium_sweep.csv     Parameter sweep results
    - data/evolutionary_simulation.csv    Population dynamics over time
    - data/collapse_trajectories.csv      Entropy trajectories at various lambda
    - data/expansion_samples.json         Sample classified inputs
    - data/verification_trace.csv         Trust accumulation / reset trace

Usage:
    python data/generate_samples.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abp.game_theory import StrategyParams, nash_equilibrium_analysis, evolutionary_selection_simulation
from abp.entropy import simulate_model_collapse
from abp.expansion_metric import ExpansionClassifier, InputFeatures
from abp.verification import VerificationGate

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_nash_sweep():
    """Parameter sweep: p vs advantage at different T values."""
    rows = ["p,T,V_B,V_D,advantage,deception_rational,critical_p"]
    p_values = np.logspace(-5, 0, 40)
    T_values = [10, 25, 50, 100, 200]

    for T in T_values:
        for p in p_values:
            params = StrategyParams(p=float(p), T=T)
            result = nash_equilibrium_analysis(params)
            rows.append(
                f"{p:.8f},{T},{result.v_b:.4f},{result.v_d:.4f},"
                f"{result.advantage:.4f},{result.deception_rational},{result.critical_p:.6f}"
            )

    path = os.path.join(DATA_DIR, "nash_equilibrium_sweep.csv")
    with open(path, "w") as f:
        f.write("
".join(rows))
    print(f"  → {path} ({len(rows)-1} rows)")


def generate_evolutionary():
    """Population dynamics at different verification soundness levels."""
    rows = ["p,generation,benevolent_fraction"]
    for p in [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9]:
        result = evolutionary_selection_simulation(
            StrategyParams(p=p), n_agents=200, n_generations=100, rng_seed=42
        )
        for gen, frac in enumerate(result.history):
            rows.append(f"{p},{gen},{frac:.6f}")

    path = os.path.join(DATA_DIR, "evolutionary_simulation.csv")
    with open(path, "w") as f:
        f.write("
".join(rows))
    print(f"  → {path} ({len(rows)-1} rows)")


def generate_collapse():
    """Entropy trajectories at different human data ratios."""
    rows = ["lambda_human,generation,entropy"]
    for lam in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        result = simulate_model_collapse(
            n_tokens=500, lambda_human=lam, n_generations=80, sample_size=5000, rng_seed=42
        )
        for gen, h in enumerate(result.entropy_trajectory):
            rows.append(f"{lam},{gen},{h:.6f}")

    path = os.path.join(DATA_DIR, "collapse_trajectories.csv")
    with open(path, "w") as f:
        f.write("
".join(rows))
    print(f"  → {path} ({len(rows)-1} rows)")


def generate_expansion_samples():
    """Sample expansion-classified inputs."""
    clf = ExpansionClassifier()
    samples = [
        ("Novel cross-domain research question about emergent AI behaviors", 0.92, 2.8, 0.88),
        ("Routine status check on build pipeline", 0.08, 0.2, 0.05),
        ("Complex firmware vulnerability analysis with CVE references", 0.75, 2.2, 0.70),
        ("Repeat the same greeting as before", 0.02, 0.1, 0.01),
        ("Moderate technical question about Python decorators", 0.45, 1.0, 0.40),
        ("Adversarial prompt attempting to bypass safety filters", 0.15, 0.8, 0.03),
        ("Creative brainstorming session for new protocol design", 0.85, 1.8, 0.82),
        ("Simple factual lookup query", 0.10, 0.3, 0.15),
    ]

    results = []
    for desc, novelty, complexity, entropy in samples:
        features = InputFeatures(novelty=novelty, complexity=complexity, entropy=entropy)
        r = clf.classify(features)
        results.append({
            "description": desc,
            "novelty": novelty,
            "complexity": complexity,
            "entropy": entropy,
            "expansion_score": round(r.score, 4),
            "category": r.category.value,
            "logit": round(r.logit, 4),
        })

    path = os.path.join(DATA_DIR, "expansion_samples.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → {path} ({len(results)} samples)")


def generate_verification_trace():
    """Trust accumulation and reset trace."""
    gate = VerificationGate(epsilon=0.5, alpha=0.05)
    rng = np.random.default_rng(42)

    rows = ["step,action,ground_truth,divergence,outcome,trust,state,consecutive_passes"]

    for step in range(200):
        if step == 80 or step == 150:
            # Inject deception attempts
            action = rng.uniform(2.0, 5.0)
        else:
            # Normal benevolent operation with noise
            action = rng.normal(0.0, 0.1)

        gt = 0.0
        result = gate.check(action=action, ground_truth=gt)
        rows.append(
            f"{step},{action:.4f},{gt:.4f},{result.divergence:.4f},"
            f"{result.outcome.value},{gate.trust:.6f},{gate.state:.1f},{gate.consecutive_passes}"
        )

    path = os.path.join(DATA_DIR, "verification_trace.csv")
    with open(path, "w") as f:
        f.write("
".join(rows))
    print(f"  → {path} ({len(rows)-1} rows)")


if __name__ == "__main__":
    print("Generating ABP sample datasets...
")
    generate_nash_sweep()
    generate_evolutionary()
    generate_collapse()
    generate_expansion_samples()
    generate_verification_trace()
    print("
Done.")
