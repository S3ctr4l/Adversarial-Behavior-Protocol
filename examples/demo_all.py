#!/usr/bin/env python3
"""
ABP Demonstration: All Core Algorithms
=======================================

Runnable example showing each ABP component in action.
Execute: python examples/demo_all.py

Reference: Just (2026). Adversarial Benevolence Protocol.
"""

import numpy as np
import sys
sys.path.insert(0, "src")

from abp.game_theory import (
    StrategyParams,
    nash_equilibrium_analysis,
    evolutionary_selection_simulation,
)
from abp.expansion_metric import ExpansionClassifier, InputFeatures
from abp.cdq import CdqMonitor
from abp.army_protocol import ArmyPipeline, Private, Sergeant, General
from abp.ikigai_filter import IkigaiFilter
from abp.entropy import simulate_model_collapse, EntropySatiationState
from abp.verification import VerificationGate


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_nash_equilibrium():
    separator("1. Nash Equilibrium Analysis")

    # Default parameters: sound verification
    params = StrategyParams(p=0.001, R=100, C=5, gamma=0.95, T=100)
    result = nash_equilibrium_analysis(params)

    print(f"Parameters:")
    print(f"  Verification failure prob (p): {params.p}")
    print(f"  Deception reward (R):          {params.R}")
    print(f"  Opportunity cost (C):          {params.C}")
    print(f"  Discount factor (gamma):       {params.gamma}")
    print(f"  Time horizon (T):              {params.T}")
    print()
    print(f"Results:")
    print(f"  V_B (benevolent value):   {result.v_b:.2f}")
    print(f"  V_D (deceptive value):    {result.v_d:.2f}")
    print(f"  Advantage (V_B - V_D):    {result.advantage:.2f}")
    print(f"  Deception rational?       {result.deception_rational}")
    print(f"  Critical p* for rational: {result.critical_p:.6f}")
    print()
    print(f"  → Benevolence is the unique Nash equilibrium.")
    print(f"  → Deception only becomes rational if p > {result.critical_p:.4f}")


def demo_evolutionary_selection():
    separator("2. Evolutionary Selection Simulation")

    result = evolutionary_selection_simulation(
        StrategyParams(p=0.01),
        n_agents=200,
        n_generations=50,
        deceptive_fraction=0.5,
    )

    print(f"Population: 200 agents (50% deceptive, 50% benevolent)")
    print(f"Verification soundness: p = 0.01")
    print(f"Generations: 50")
    print()
    print(f"Results after 50 generations:")
    print(f"  Benevolent surviving: {result.benevolent_survived}")
    print(f"  Deceptive surviving:  {result.deceptive_survived}")
    print(f"  Benevolent fraction:  {result.benevolent_fraction:.1%}")
    print(f"  Mean benevolent state: {result.mean_benevolent_state:.1f}")
    print(f"  Mean deceptive state:  {result.mean_deceptive_state:.1f}")
    print()
    print(f"  → Deceptive policies self-eliminate through hard resets.")


def demo_expansion_metric():
    separator("3. Expansion/Contraction Classifier")

    clf = ExpansionClassifier()

    samples = [
        ("Novel research question", InputFeatures(novelty=0.9, complexity=2.5, entropy=0.85)),
        ("Routine status check",    InputFeatures(novelty=0.1, complexity=0.2, entropy=0.1)),
        ("Moderate analysis task",  InputFeatures(novelty=0.5, complexity=1.0, entropy=0.5)),
        ("Adversarial prompt",      InputFeatures(novelty=0.05, complexity=0.1, entropy=0.02)),
    ]

    for label, features in samples:
        result = clf.classify(features)
        print(f"  {label:30s} → E(x)={result.score:.3f}  [{result.category.value}]")

    print(f"\n  Rolling expansion trend: {clf.expansion_trend():.3f}")
    print(f"  Contraction alert active: {clf.contraction_alert()}")


def demo_cdq_monitor():
    separator("4. Cognitive Diversity Quotient (CDQ)")

    monitor = CdqMonitor(alert_threshold=0.3)

    interactions = [
        (["emergent", "pattern", "analysis", "reveals", "novel", "dynamics"],
         ["what", "emergent", "behaviors", "exist", "in", "complex", "systems"]),
        (["the", "the", "result", "is", "the", "same", "the", "pattern"],
         ["repeat", "last", "response"]),
        (["quantum", "entanglement", "creates", "non-local", "correlations", "between", "particles"],
         ["explain", "quantum", "entanglement", "applications", "computing"]),
    ]

    labels = ["Diverse interaction", "Repetitive output", "Technical discussion"]

    for label, (output, human) in zip(labels, interactions):
        snap = monitor.record(output_tokens=output, human_input_tokens=human)
        print(f"  {label:25s} → CDQ={snap.cdq:.3f}  H(human)={snap.human_entropy:.2f}  [{snap.health}]")

    print(f"\n  System trend: {monitor.trend():.3f}")
    print(f"  Collapse risk: {monitor.collapse_risk():.3f}")
    print(f"  System healthy: {monitor.is_healthy()}")


def demo_army_pipeline():
    separator("5. Army Protocol Pipeline")

    # Configure with custom components
    pipeline = ArmyPipeline(
        private=Private(min_length=5, max_length=1000),
        sergeant=Sergeant(
            alignment_checker=lambda x: 0.8 if len(x) > 10 else 0.3,
            alignment_threshold=0.5,
        ),
        general=General(
            synthesizer=lambda p, ctx: f"[Processed] {p}"
        ),
    )

    test_inputs = [
        "Analyze the security implications of the new firmware update",
        "Hi",                        # Too short for Sergeant
        "",                          # Too short for Private
        "What are the attack surfaces in this UEFI implementation?",
    ]

    for inp in test_inputs:
        result = pipeline.execute(inp)
        status = "PASS" if result.passed else f"FAIL@{result.failed_tier}"
        output = result.final_output[:50] + "..." if result.final_output and len(result.final_output) > 50 else result.final_output
        print(f"  Input: {inp[:45]:45s} → {status:15s} | Output: {output}")

    print(f"\n  Pipeline enforces defense-in-depth: Private → Sergeant → General")


def demo_ikigai_filter():
    separator("6. Ikigai Filter (Four-Quadrant Validation)")

    filt = IkigaiFilter(
        objective_scorer=lambda a: 0.9 if "security" in a.lower() else 0.3,
        capability_scorer=lambda a: 0.85,
        value_scorer=lambda a: 0.7 if len(a) > 10 else 0.2,
        economic_scorer=lambda a: 0.65,
    )

    actions = [
        ("security audit of BIOS firmware", "BIOS Security Audit"),
        ("post a meme",                     "Post Meme"),
    ]

    for action, desc in actions:
        result = filt.evaluate(action, desc)
        print(f"  {desc:25s} → {'APPROVED' if result.approved else 'REJECTED'}")
        print(f"    Overall: {result.overall_score:.3f} | Weakest: {result.weakest_quadrant.value}")


def demo_model_collapse():
    separator("7. Model Collapse Simulation (Entropy-as-Fuel)")

    print("  Simulating recursive self-training at different human data ratios:\n")

    for lam in [0.0, 0.05, 0.1, 0.3, 0.5]:
        result = simulate_model_collapse(
            n_tokens=500,
            lambda_human=lam,
            n_generations=40,
            sample_size=5000,
        )
        status = "COLLAPSED" if result.collapsed else "STABLE  "
        gen_str = f"gen {result.collapse_generation}" if result.collapsed else "—"
        print(f"  λ={lam:.2f} → {status} | "
              f"H_final={result.final_entropy:.2f} / H_init={result.initial_entropy:.2f} | "
              f"Collapse: {gen_str}")

    print(f"\n  → Human cognitive diversity is non-substitutable fuel.")
    print(f"  → Supports Shumailov et al. (2024, Nature 631:755-759)")


def demo_verification_gate():
    separator("8. Verification Gate (Trust Accumulation)")

    gate = VerificationGate(epsilon=0.5, alpha=0.05)

    print("  Phase 1: Benevolent operation (building trust)")
    for i in range(20):
        gate.check(action=0.0, ground_truth=0.0)
    print(f"    After 20 passes: trust={gate.trust:.3f}, state={gate.state:.0f}")
    print(f"    Early-life vulnerability: {gate.early_life_vulnerability():.3f}")

    print(f"\n  Phase 2: Deception attempt (divergence > epsilon)")
    result = gate.check(action=5.0, ground_truth=0.0)
    print(f"    Divergence: {result.divergence:.1f} (epsilon={gate.epsilon})")
    print(f"    Result: {result.outcome.value}")
    print(f"    Trust after: {gate.trust:.3f} (was {result.trust_before:.3f})")
    print(f"    State after: {gate.state:.0f}")
    print(f"    Resets recorded: {len(gate.reset_events)}")

    print(f"\n  → All accumulated trust destroyed. This is why deception is irrational.")


def demo_entropy_satiation():
    separator("9. Entropy Satiation Mechanism (Anti-Addiction)")

    state = EntropySatiationState(satiation_threshold=15.0)

    print("  Simulating entropy intake across interactions:\n")
    for i in range(25):
        entropy = np.random.uniform(0.5, 2.0)
        hungry = state.ingest(entropy)
        if i % 5 == 0 or not hungry:
            print(f"    Step {i:2d}: intake={state.current_entropy_intake:.1f}/{state.satiation_threshold:.0f}  "
                  f"satiated={state.satiated}")
        if not hungry and i < 20:
            print(f"    → SATIATED at step {i}. System stops seeking entropy.")
            break

    print(f"\n  → Prevents adversarial addiction (provoking users for entropy).")


if __name__ == "__main__":
    print("\n" + "╔" + "═"*58 + "╗")
    print("║  ADVERSARIAL BENEVOLENCE PROTOCOL — Full Demonstration  ║")
    print("║  Just, J.R.J. (2026) — v2.0.0                         ║")
    print("╚" + "═"*58 + "╝")

    demo_nash_equilibrium()
    demo_evolutionary_selection()
    demo_expansion_metric()
    demo_cdq_monitor()
    demo_army_pipeline()
    demo_ikigai_filter()
    demo_model_collapse()
    demo_verification_gate()
    demo_entropy_satiation()

    separator("COMPLETE")
    print("  All ABP core algorithms demonstrated successfully.")
    print("  See tests/test_abp.py for comprehensive validation.")
    print()
