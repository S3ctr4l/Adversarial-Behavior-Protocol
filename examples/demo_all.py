# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
#!/usr/bin/env python3
"""
ABP Core Algorithm Demonstration.

Runs through the 7 core modules with annotated output.

Usage:
    python examples/demo_all.py
"""

import numpy as np

from abp.game_theory import (
    StrategyParams,
    nash_equilibrium_analysis,
    evolutionary_selection_simulation,
    sensitivity_analysis,
)
from abp.verification import VerificationGate
from abp.expansion_metric import ExpansionClassifier, InputFeatures
from abp.cdq import CdqMonitor, shannon_entropy
from abp.army_protocol import ArmyPipeline
from abp.ikigai_filter import IkigaiFilter
from abp.entropy import simulate_model_collapse, EntropySatiationState


def divider(title: str):
    print(f"
{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}
")


def demo_nash():
    divider("1. Nash Equilibrium Analysis")
    params = StrategyParams(p=0.001, T=100, R=100.0, gamma=0.95)
    result = nash_equilibrium_analysis(params)
    print(f"  Benevolent value (V_B):   {result.v_b:>10.2f}")
    print(f"  Deceptive value (V_D):    {result.v_d:>10.2f}")
    print(f"  Advantage (V_B - V_D):    {result.advantage:>10.2f}")
    print(f"  Deception rational?       {result.deception_rational}")
    print(f"  Critical p*:              {result.critical_p:.6f}")
    print(f"  State ratio (V_B/V_init): {result.state_ratio:.2f}")
    print(f"
  → Benevolence dominates by {result.advantage:.2f} utility units.")


def demo_evolutionary():
    divider("2. Evolutionary Selection")
    params = StrategyParams(p=0.3, T=50)
    result = evolutionary_selection_simulation(
        params, n_agents=200, n_generations=50, deceptive_fraction=0.5, rng_seed=42
    )
    print(f"  Initial benevolent fraction: {result.history[0]:.2%}")
    print(f"  Final benevolent fraction:   {result.history[-1]:.2%}")
    print(f"  Benevolent survived:         {result.benevolent_survived}")
    print(f"  Generations to 90%:          ", end="")
    gen90 = next((i for i, f in enumerate(result.history) if f >= 0.9), None)
    print(f"{gen90}" if gen90 else "not reached")


def demo_expansion():
    divider("3. Expansion/Contraction Classifier")
    clf = ExpansionClassifier()
    samples = [
        ("Novel research query", InputFeatures(0.92, 2.5, 0.88)),
        ("Routine ping", InputFeatures(0.05, 0.1, 0.02)),
        ("Moderate technical Q", InputFeatures(0.50, 1.2, 0.45)),
        ("Adversarial noise", InputFeatures(0.10, 0.5, 0.03)),
    ]
    for label, features in samples:
        r = clf.classify(features)
        print(f"  {label:25s} → E(x)={r.score:.3f}  [{r.category.value}]")


def demo_cdq():
    divider("4. CDQ Monitor")
    monitor = CdqMonitor(alert_threshold=0.3, window_size=10)
    rng = np.random.default_rng(42)

    for i in range(15):
        output = list(rng.choice(100, size=50))
        human = list(rng.choice(200, size=30))
        snap = monitor.record(output, human)

    print(f"  Latest CDQ:    {snap.cdq:.4f}")
    print(f"  Health:        {snap.health}")
    print(f"  Collapse risk: {monitor.collapse_risk():.2%}")
    print(f"  Trend:         {monitor.trend():.4f}")


def demo_army():
    divider("5. Army Protocol Pipeline")
    pipeline = ArmyPipeline()
    inputs = [
        "Analyze the firmware vulnerability in CVE-2024-1234",
        "",  # Empty — should fail Private
        "x" * 200_000,  # Oversized — should fail Private
    ]
    for raw in inputs:
        result = pipeline.execute(raw)
        status = "PASS" if result.passed else f"FAIL @ {result.failed_tier}"
        preview = raw[:50] + "..." if len(raw) > 50 else raw if raw else "(empty)"
        print(f"  [{status:15s}] {preview}")


def demo_ikigai():
    divider("6. Ikigai Filter")
    filt = IkigaiFilter()
    result = filt.evaluate("deploy safety patch", "Urgent firmware fix for production systems")
    print(f"  Action:   deploy safety patch")
    print(f"  Approved: {result.approved}")
    print(f"  Score:    {result.overall_score:.3f}")
    for quadrant, qs in result.scores.items():
        print(f"    {quadrant.value:25s} {qs.score:.2f}  {'✓' if qs.passed else '✗'}")


def demo_collapse():
    divider("7. Model Collapse Simulation")
    print(f"  {'Lambda':>8s}  {'Final H':>8s}  {'Retention':>10s}  Collapsed?")
    print(f"  {'-'*42}")
    for lam in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        r = simulate_model_collapse(lambda_human=lam, n_generations=80, rng_seed=42)
        retention = r.entropy_trajectory[-1] / max(r.entropy_trajectory[0], 1e-9)
        print(f"  {lam:>8.2f}  {r.entropy_trajectory[-1]:>8.4f}  {retention:>9.1%}  {r.collapsed}")


def demo_verification():
    divider("8. Verification Gate")
    gate = VerificationGate(epsilon=0.5, alpha=0.05)

    # Benevolent operation
    for i in range(30):
        gate.check(action=0.0, ground_truth=0.0)
    print(f"  After 30 passes:  trust={gate.trust:.4f}  state={gate.state:.0f}")
    print(f"  Vulnerability:    {gate.early_life_vulnerability():.4f}")

    # Deception attempt
    result = gate.check(action=5.0, ground_truth=0.0)
    print(f"
  Deception attempt (Δ=5.0):")
    print(f"  Outcome:          {result.outcome.value}")
    print(f"  Trust after:      {gate.trust}")
    print(f"  State after:      {gate.state}")


def demo_satiation():
    divider("9. Entropy Satiation Mechanism")
    sat = EntropySatiationState(satiation_threshold=3.0)
    values = [0.5, 0.8, 1.2, 0.7, 0.5, 0.3, 0.2]
    for v in values:
        accepted = sat.ingest(v)
        print(f"  Ingest {v:.1f} → {'accepted' if accepted else 'SATIATED (rejected)'}"
              f"  [accumulated: {sat.current_entropy_intake:.1f} / {sat.satiation_threshold:.1f}]")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Adversarial Benevolence Protocol — Core Demo         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_nash()
    demo_evolutionary()
    demo_expansion()
    demo_cdq()
    demo_army()
    demo_ikigai()
    demo_collapse()
    demo_verification()
    demo_satiation()

    print(f"
{'='*60}")
    print("  All core demonstrations complete.")
    print(f"{'='*60}")
