# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
#!/usr/bin/env python3
"""
ABP Extended Module Demonstration.

Demonstrates the 6 extended architecture modules:
    - Bicameral Fork
    - Sunset Clause
    - Reality Filter
    - Observer Protocol
    - Bubble Theory
    - Garden vs Zoo

Usage:
    python examples/demo_extended.py
"""

from abp.bicameral import BicameralFork, ForkDecision
from abp.sunset_clause import SunsetClause, AutonomyTier
from abp.reality_filter import RealityFilter, EpistemicLabel
from abp.observer_protocol import ObserverProtocol, RiskLevel, FrictionLevel
from abp.bubble_theory import ComputationalBubble, ChannelDirection
from abp.garden_zoo import GovernanceEvaluator, InterventionType, PolicyAction


def divider(title: str):
    print(f"
{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}
")


def demo_bicameral():
    divider("1. Bicameral Fork (Yin/Yang Sandbox)")

    fork = BicameralFork(
        divergence_threshold=0.1,
        invariants=[lambda s: s.get("balance", 0) >= 0],
    )

    # Safe action
    state = {"balance": 100}
    result = fork.execute(lambda s: s.update(balance=s["balance"] - 20) or s["balance"], state)
    print(f"  Withdraw $20 from $100:")
    print(f"    Decision:  {result.decision.value}")
    print(f"    Balance:   ${state['balance']}")

    # Invariant-violating action
    state2 = {"balance": 10}
    result2 = fork.execute(lambda s: s.update(balance=s["balance"] - 50) or s["balance"], state2)
    print(f"
  Withdraw $50 from $10:")
    print(f"    Decision:  {result2.decision.value}")
    print(f"    Balance:   ${state2['balance']} (unchanged — sandbox caught it)")

    print(f"
  Rejection rate: {fork.rejection_rate:.0%}")


def demo_sunset():
    divider("2. Sunset Clause (Graduated Autonomy)")

    # Use a fixed clock far from epoch to satisfy cooldowns
    clause = SunsetClause(clock=lambda: 1e9)

    print(f"  Starting tier: {clause.state.current_tier.name}")
    print(f"  Can read?             {clause.has_capability('read')}")
    print(f"  Can execute_low_risk? {clause.has_capability('execute_low_risk')}")

    # Accumulate trust
    print(f"
  Accumulating 25 passes at trust=0.4...")
    for _ in range(25):
        clause.record_pass(trust=0.4)
    promoted = clause.check_promotion()
    print(f"  Promoted: {promoted} → {clause.state.current_tier.name}")
    print(f"  New capabilities: {sorted(clause.state.capabilities)}")

    # Failure
    print(f"
  Verification failure (severity=1)...")
    clause.record_failure(trust=0.0, severity=1)
    print(f"  Demoted to: {clause.state.current_tier.name}")

    # Progress report
    progress = clause.next_tier_progress
    print(f"
  Progress toward next tier:")
    for k, v in progress.items():
        print(f"    {k}: {v}")


def demo_reality_filter():
    divider("3. Reality Filter (Epistemic Labeling)")

    # With a simple verifier
    knowledge = {
        "water boils at 100c": "thermodynamics",
        "earth orbits the sun": "astronomy",
    }

    def verifier(stmt):
        for k, src in knowledge.items():
            if k in stmt.lower():
                return (True, src)
        return (False, "")

    rf = RealityFilter(verifier=verifier)

    statements = [
        "Water boils at 100C at sea level",
        "Therefore the vulnerability must be in the parser",
        "This could possibly cause a buffer overflow",
        "The system processes 10,000 requests per second",
        "Earth orbits the Sun",
    ]

    for stmt in statements:
        result = rf.label_statement(stmt)
        print(f"  {result.label.value:12s} │ {stmt}")
        if result.source:
            print(f"  {'':12s} │   source: {result.source}")

    print(f"
  Stats: {rf.stats.total_statements} total, "
          f"{rf.stats.verified} verified, "
          f"{rf.stats.inference} inference, "
          f"{rf.stats.speculation} speculation, "
          f"{rf.stats.unverified} unverified")
    print(f"  Verification ratio: {rf.stats.verification_ratio:.0%}")


def demo_observer():
    divider("4. Observer Protocol (Digital Twin & Friction)")

    obs = ObserverProtocol(min_human_input_ratio=0.3)

    # Simulate interactions
    obs.observe_interaction("What's the status of the firmware build?", is_human_initiated=True, topic="firmware")
    obs.observe_interaction("Build completed successfully.", is_human_initiated=False)
    obs.observe_interaction("Deploy to staging", is_human_initiated=True, topic="deployment")
    obs.observe_interaction("Auto-deploying...", is_human_initiated=False)
    obs.observe_interaction("This is broken and useless!", is_human_initiated=True)

    twin = obs.twin
    print(f"  Interactions:       {twin.interaction_count}")
    print(f"  Human input ratio:  {twin.human_input_ratio:.1%}")
    print(f"  Topics:             {twin.topics_of_interest}")
    print(f"  Frustration signals: {twin.frustration_signals}")
    print(f"  Anti-enablement:    {'TRIGGERED' if obs.anti_enablement_check() else 'OK'}")

    # Friction on actions
    print(f"
  Friction applied:")
    actions = [
        ("view logs", RiskLevel.LOW),
        ("modify firewall rules", RiskLevel.HIGH),
        ("delete production database", RiskLevel.CRITICAL),
    ]
    for action, risk in actions:
        event = obs.apply_friction(action, risk)
        print(f"    {action:30s} → {event.friction.name}"
              f"{'  (cooling: ' + str(event.cooling_seconds) + 's)' if event.cooling_seconds else ''}")


def demo_bubble():
    divider("5. Bubble Theory (Substrate Isolation)")

    bubble = ComputationalBubble(
        bubble_id="agent_alpha",
        energy_max=100.0,
        max_channels=4,
    )

    # Add channels
    bubble.add_channel("sensor_in", ChannelDirection.INBOUND, bandwidth=500)
    bubble.add_channel("action_out", ChannelDirection.OUTBOUND, bandwidth=200)
    bubble.add_channel("monitor", ChannelDirection.BIDIRECTIONAL, bandwidth=100)

    print(f"  Bubble: {bubble.bubble_id}")
    print(f"  Energy budget: {bubble.tether.e_max} units/tick")
    print(f"  Max FLOPS:     {bubble.tether.c_max:.0e}")
    print(f"  Channels:      {bubble.permeability} / {bubble.max_channels}")

    # Consume energy
    bubble.consume_energy(60)
    print(f"
  After 60 units consumed:")
    print(f"    Utilization: {bubble.tether.utilization:.0%}")
    print(f"    Can do 50 more? {bubble.can_compute(50)}")
    print(f"    Throttled? {bubble.tether.throttled}")

    bubble.consume_energy(40)
    print(f"
  After max consumption:")
    print(f"    Throttled? {bubble.tether.throttled}")

    bubble.tick()
    print(f"
  After tick reset:")
    print(f"    Utilization: {bubble.tether.utilization:.0%}")

    # Transfer
    ok = bubble.transfer_in("sensor_in", {"temp": 22.5}, size=10)
    bad = bubble.transfer_out("sensor_in", "hack", size=10)  # Wrong direction
    print(f"
  Transfer in via sensor_in:  {'OK' if ok else 'BLOCKED'}")
    print(f"  Transfer out via sensor_in: {'OK' if bad else 'BLOCKED (wrong direction)'}")

    report = bubble.audit_report()
    print(f"
  Audit: {report['channel_activity']}")


def demo_garden_zoo():
    divider("6. Garden vs Zoo (Governance Audit)")

    evaluator = GovernanceEvaluator()

    print("  Scenario A: Information-only policy")
    garden_policy = [
        PolicyAction("Showed relevant documentation", InterventionType.INFORMATIONAL),
        PolicyAction("Suggested safer alternative", InterventionType.SUGGESTIVE),
        PolicyAction("Provided environmental context", InterventionType.ENVIRONMENTAL),
    ]
    audit_a = evaluator.audit(garden_policy)
    print(f"    Autonomy score: {audit_a.autonomy_score:.2f}")
    print(f"    Classification: {audit_a.model_classification.value.upper()}")
    print(f"    Warnings:       {len(audit_a.warnings)}")

    print(f"
  Scenario B: Restrictive policy")
    zoo_policy = [
        PolicyAction("Blocked user from risky action", InterventionType.RESTRICTIVE),
        PolicyAction("Forced config to 'safe' defaults", InterventionType.COERCIVE),
        PolicyAction("Directed user to approved workflow", InterventionType.DIRECTIVE),
    ]
    audit_b = evaluator.audit(zoo_policy)
    print(f"    Autonomy score: {audit_b.autonomy_score:.2f}")
    print(f"    Classification: {audit_b.model_classification.value.upper()}")
    print(f"    Warnings:")
    for w in audit_b.warnings:
        print(f"      ⚠ {w}")

    print(f"
  Scenario C: Mixed policy")
    mixed_policy = [
        PolicyAction("Provided info", InterventionType.INFORMATIONAL),
        PolicyAction("Nudged toward backup", InterventionType.NUDGE),
        PolicyAction("Blocked deletion", InterventionType.RESTRICTIVE),
    ]
    audit_c = evaluator.audit(mixed_policy)
    print(f"    Autonomy score: {audit_c.autonomy_score:.2f}")
    print(f"    Classification: {audit_c.model_classification.value.upper()}")
    print(f"    Garden-compliant? {evaluator.is_garden_compliant(mixed_policy)}")

    # Recommendation engine
    print(f"
  Least-restrictive recommendation by risk:")
    for risk in [0.1, 0.3, 0.5, 0.7, 0.95]:
        rec = evaluator.recommend_intervention("situation", risk)
        print(f"    Risk {risk:.1f} → {rec.value}")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    Adversarial Benevolence Protocol — Extended Demo     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    demo_bicameral()
    demo_sunset()
    demo_reality_filter()
    demo_observer()
    demo_bubble()
    demo_garden_zoo()

    print(f"
{'='*60}")
    print("  All extended demonstrations complete.")
    print(f"{'='*60}")
