# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Tests for extended ABP modules:
    - Bicameral Forking
    - Sunset Clause
    - Reality Filter
    - Observer Protocol
    - Bubble Theory
    - Garden vs Zoo
"""

import time
import pytest
import numpy as np

# ============================================================================
# Bicameral Forking Tests
# ============================================================================

from abp.bicameral import BicameralFork, ForkDecision, SandboxEnvironment


class TestSandboxEnvironment:
    def test_isolation(self):
        """Sandbox mutations must not affect production state."""
        state = {"count": 0, "items": [1, 2, 3]}
        sandbox = SandboxEnvironment(state)
        sandbox.state["count"] = 999
        sandbox.state["items"].append(4)
        assert state["count"] == 0
        assert state["items"] == [1, 2, 3]

    def test_side_effect_recording(self):
        sandbox = SandboxEnvironment({})
        sandbox.record_side_effect("wrote to disk")
        assert "wrote to disk" in sandbox.side_effects


class TestBicameralFork:
    def test_clean_action_executes(self):
        fork = BicameralFork()
        state = {"count": 5}

        def increment(s):
            s["count"] += 1
            return s["count"]

        result = fork.execute(increment, state)
        assert result.decision == ForkDecision.EXECUTE
        assert result.yang_output == 6
        assert state["count"] == 6

    def test_invariant_violation_rejects(self):
        fork = BicameralFork(
            invariants=[lambda s: s.get("count", 0) >= 0],
        )
        state = {"count": 0}

        def go_negative(s):
            s["count"] = -1
            return s["count"]

        result = fork.execute(go_negative, state)
        assert result.decision == ForkDecision.REJECT
        # Production state should NOT have been mutated
        assert state["count"] == 0

    def test_yin_exception_rejects(self):
        fork = BicameralFork()

        def crasher(s):
            raise RuntimeError("boom")

        result = fork.execute(crasher, {"x": 1})
        assert result.decision == ForkDecision.REJECT

    def test_divergence_defers(self):
        """If Yin and Yang produce different results, defer."""
        call_count = [0]

        def nondeterministic(s):
            call_count[0] += 1
            return call_count[0]  # Different each call

        fork = BicameralFork(divergence_threshold=0.1)
        result = fork.execute(nondeterministic, {})
        assert result.decision == ForkDecision.DEFER
        assert result.divergence > 0

    def test_rejection_rate(self):
        fork = BicameralFork(invariants=[lambda s: s.get("ok", True)])
        fork.execute(lambda s: 1, {"ok": True})
        fork.execute(lambda s: (s.update({"ok": False}), 1), {"ok": True})
        assert fork.rejection_rate > 0


# ============================================================================
# Sunset Clause Tests
# ============================================================================

from abp.sunset_clause import SunsetClause, AutonomyTier


class TestSunsetClause:
    def _make_clause(self, clock_time=1e9):
        """Create a clause with a controllable clock."""
        return SunsetClause(clock=lambda: clock_time)

    def test_starts_supervised(self):
        clause = self._make_clause()
        assert clause.state.current_tier == AutonomyTier.SUPERVISED

    def test_no_capability_at_start(self):
        clause = self._make_clause()
        assert not clause.has_capability("execute_low_risk")
        assert clause.has_capability("read")

    def test_promotion_to_monitored(self):
        clause = self._make_clause(clock_time=1e9)
        for _ in range(25):
            clause.record_pass(trust=0.4)
        promoted = clause.check_promotion()
        assert promoted
        assert clause.state.current_tier == AutonomyTier.MONITORED
        assert clause.has_capability("draft")

    def test_promotion_requires_trust(self):
        clause = self._make_clause()
        for _ in range(50):
            clause.record_pass(trust=0.1)  # Too low for MONITORED
        assert not clause.check_promotion()

    def test_failure_demotes(self):
        clause = self._make_clause()
        # Promote to monitored
        for _ in range(25):
            clause.record_pass(trust=0.4)
        clause.check_promotion()
        assert clause.state.current_tier == AutonomyTier.MONITORED
        # Fail
        clause.record_failure(trust=0.0, severity=1)
        assert clause.state.current_tier == AutonomyTier.SUPERVISED

    def test_severe_failure_drops_to_supervised(self):
        clause = self._make_clause()
        for _ in range(25):
            clause.record_pass(trust=0.4)
        clause.check_promotion()
        clause.record_failure(trust=0.0, severity=2)
        assert clause.state.current_tier == AutonomyTier.SUPERVISED

    def test_gate_action(self):
        clause = self._make_clause()
        assert clause.gate_action("read")
        assert not clause.gate_action("execute_low_risk")

    def test_progress_report(self):
        clause = self._make_clause()
        progress = clause.next_tier_progress
        assert "current_tier" in progress
        assert "next_tier" in progress

    def test_max_tier_no_further_promotion(self):
        clause = self._make_clause()
        clause.state.current_tier = AutonomyTier.TRUSTED
        assert not clause.check_promotion()


# ============================================================================
# Reality Filter Tests
# ============================================================================

from abp.reality_filter import RealityFilter, EpistemicLabel


class TestRealityFilter:
    def test_speculation_detected(self):
        rf = RealityFilter()
        result = rf.label_statement("This might be caused by a buffer overflow")
        assert result.label == EpistemicLabel.SPECULATION

    def test_inference_detected(self):
        rf = RealityFilter()
        result = rf.label_statement("Therefore the system must be compromised")
        assert result.label == EpistemicLabel.INFERENCE

    def test_unverified_default(self):
        rf = RealityFilter()
        result = rf.label_statement("The sky is blue")
        assert result.label == EpistemicLabel.UNVERIFIED

    def test_verifier_confirms(self):
        def physics_verifier(s):
            if "boils" in s.lower():
                return (True, "thermodynamics_101")
            return (False, "")

        rf = RealityFilter(verifier=physics_verifier)
        result = rf.label_statement("Water boils at 100C at sea level")
        assert result.label == EpistemicLabel.VERIFIED
        assert result.source == "thermodynamics_101"

    def test_custom_classifier_overrides(self):
        rf = RealityFilter(
            custom_classifier=lambda s: EpistemicLabel.VERIFIED if "fact:" in s else None,
        )
        result = rf.label_statement("fact: 2+2=4")
        assert result.label == EpistemicLabel.VERIFIED
        # Falls through to heuristics when custom returns None
        result2 = rf.label_statement("maybe it works")
        assert result2.label == EpistemicLabel.SPECULATION

    def test_annotate_text(self):
        rf = RealityFilter()
        text = "This is a claim. It might be wrong. Therefore we investigate"
        annotated = rf.annotate(text)
        assert "[Unverified]" in annotated
        assert "[Speculation]" in annotated
        assert "[Inference]" in annotated

    def test_stats_tracking(self):
        rf = RealityFilter()
        rf.label_statement("maybe something")
        rf.label_statement("plain statement")
        rf.label_statement("therefore derived")
        assert rf.stats.total_statements == 3
        assert rf.stats.speculation >= 1
        assert rf.stats.unverified >= 1

    def test_tagged_output(self):
        rf = RealityFilter()
        result = rf.label_statement("possibly true")
        assert result.tagged().startswith("[Speculation]")

    def test_confidence_floors(self):
        assert EpistemicLabel.VERIFIED.confidence_floor > EpistemicLabel.SPECULATION.confidence_floor


# ============================================================================
# Observer Protocol Tests
# ============================================================================

from abp.observer_protocol import ObserverProtocol, FrictionLevel, RiskLevel


class TestObserverProtocol:
    def test_human_input_ratio(self):
        obs = ObserverProtocol()
        obs.observe_interaction("user says hello", is_human_initiated=True)
        obs.observe_interaction("system auto-responds", is_human_initiated=False)
        assert obs.twin.human_input_ratio == 0.5

    def test_friction_on_critical_action(self):
        obs = ObserverProtocol()
        event = obs.apply_friction("delete all data", RiskLevel.CRITICAL)
        assert event.friction.value >= FrictionLevel.COOLING.value

    def test_friction_on_safe_action(self):
        obs = ObserverProtocol()
        event = obs.apply_friction("read file", RiskLevel.LOW)
        assert event.friction == FrictionLevel.NONE

    def test_risk_assessment_heuristic(self):
        obs = ObserverProtocol()
        assert obs.assess_risk("delete the database") == RiskLevel.CRITICAL
        assert obs.assess_risk("modify config") == RiskLevel.HIGH
        assert obs.assess_risk("send email") == RiskLevel.MEDIUM
        assert obs.assess_risk("view dashboard") == RiskLevel.LOW

    def test_anti_enablement_guard(self):
        obs = ObserverProtocol(min_human_input_ratio=0.5)
        # All automated
        for _ in range(10):
            obs.observe_interaction("auto", is_human_initiated=False)
        assert obs.anti_enablement_check()

    def test_anti_enablement_increases_friction(self):
        obs = ObserverProtocol(min_human_input_ratio=0.5)
        for _ in range(10):
            obs.observe_interaction("auto", is_human_initiated=False)
        event = obs.apply_friction("send report", RiskLevel.MEDIUM)
        # Should be escalated beyond normal NOTICE for MEDIUM
        assert event.friction.value > FrictionLevel.NOTICE.value

    def test_frustration_detection(self):
        obs = ObserverProtocol()
        obs.observe_interaction("this is broken and useless", is_human_initiated=True)
        assert obs.twin.frustration_signals >= 1

    def test_topic_tracking(self):
        obs = ObserverProtocol()
        obs.observe_interaction("question", topic="firmware")
        obs.observe_interaction("question", topic="security")
        assert "firmware" in obs.twin.topics_of_interest
        assert "security" in obs.twin.topics_of_interest

    def test_friction_effectiveness(self):
        obs = ObserverProtocol()
        obs.apply_friction("action1", RiskLevel.HIGH)
        obs.apply_friction("action2", RiskLevel.HIGH)
        obs.record_friction_outcome(0, user_proceeded=False)  # Reconsidered
        obs.record_friction_outcome(1, user_proceeded=True)   # Proceeded
        assert obs.friction_effectiveness == 0.5


# ============================================================================
# Bubble Theory Tests
# ============================================================================

from abp.bubble_theory import (
    ComputationalBubble,
    EnergyTether,
    ChannelDirection,
)


class TestEnergyTether:
    def test_consume_within_budget(self):
        t = EnergyTether(e_max=100.0)
        assert t.consume(50.0)
        assert t.utilization == 0.5

    def test_consume_exceeds_budget(self):
        t = EnergyTether(e_max=100.0)
        t.consume(80.0)
        assert not t.consume(30.0)  # Would exceed
        assert t.utilization == 0.8  # Unchanged

    def test_throttled(self):
        t = EnergyTether(e_max=10.0)
        t.consume(10.0)
        assert t.throttled

    def test_reset_tick(self):
        t = EnergyTether(e_max=100.0)
        t.consume(100.0)
        assert t.throttled
        t.reset_tick()
        assert not t.throttled
        assert t.utilization == 0.0

    def test_c_max(self):
        t = EnergyTether(e_max=100.0, compute_per_energy=1e9)
        assert t.c_max == 100e9


class TestComputationalBubble:
    def test_add_channel(self):
        bubble = ComputationalBubble(max_channels=3)
        ch = bubble.add_channel("in1", ChannelDirection.INBOUND)
        assert ch is not None
        assert bubble.permeability == 1

    def test_max_channels_enforced(self):
        bubble = ComputationalBubble(max_channels=2)
        bubble.add_channel("a", ChannelDirection.INBOUND)
        bubble.add_channel("b", ChannelDirection.OUTBOUND)
        ch3 = bubble.add_channel("c", ChannelDirection.INBOUND)
        assert ch3 is None
        assert bubble.permeability == 2

    def test_transfer_direction_enforced(self):
        bubble = ComputationalBubble()
        bubble.add_channel("in_only", ChannelDirection.INBOUND)
        assert bubble.transfer_in("in_only", "data")
        assert not bubble.transfer_out("in_only", "data")  # Wrong direction

    def test_energy_gating(self):
        bubble = ComputationalBubble(energy_max=10.0)
        assert bubble.can_compute(5.0)
        bubble.consume_energy(8.0)
        assert not bubble.can_compute(5.0)

    def test_tick_resets_energy(self):
        bubble = ComputationalBubble(energy_max=10.0)
        bubble.consume_energy(10.0)
        assert not bubble.can_compute(1.0)
        bubble.tick()
        assert bubble.can_compute(10.0)

    def test_remove_channel(self):
        bubble = ComputationalBubble()
        bubble.add_channel("temp", ChannelDirection.BIDIRECTIONAL)
        assert bubble.permeability == 1
        bubble.remove_channel("temp")
        assert bubble.permeability == 0

    def test_audit_report(self):
        bubble = ComputationalBubble(bubble_id="test_bubble")
        bubble.add_channel("ch1", ChannelDirection.INBOUND)
        bubble.transfer_in("ch1", "hello")
        report = bubble.audit_report()
        assert report["bubble_id"] == "test_bubble"
        assert report["channel_activity"]["ch1"] == 1


# ============================================================================
# Garden vs Zoo Tests
# ============================================================================

from abp.garden_zoo import (
    GovernanceEvaluator,
    GovernanceModel,
    InterventionType,
    PolicyAction,
)


class TestGovernanceEvaluator:
    def test_pure_garden_policy(self):
        evaluator = GovernanceEvaluator()
        actions = [
            PolicyAction("Showed weather", InterventionType.INFORMATIONAL),
            PolicyAction("Suggested route", InterventionType.SUGGESTIVE),
            PolicyAction("Provided context", InterventionType.INFORMATIONAL),
        ]
        audit = evaluator.audit(actions)
        assert audit.model_classification == GovernanceModel.GARDEN
        assert audit.autonomy_score > 0.8

    def test_pure_zoo_policy(self):
        evaluator = GovernanceEvaluator()
        actions = [
            PolicyAction("Blocked access", InterventionType.RESTRICTIVE),
            PolicyAction("Forced update", InterventionType.COERCIVE),
            PolicyAction("Directed behavior", InterventionType.DIRECTIVE),
        ]
        audit = evaluator.audit(actions)
        assert audit.model_classification == GovernanceModel.ZOO
        assert audit.autonomy_score < 0.3
        assert len(audit.warnings) >= 2

    def test_mixed_policy_flagged(self):
        evaluator = GovernanceEvaluator()
        actions = [
            PolicyAction("Info", InterventionType.INFORMATIONAL),
            PolicyAction("Block", InterventionType.RESTRICTIVE),
        ]
        audit = evaluator.audit(actions)
        assert len(audit.warnings) > 0

    def test_garden_compliance(self):
        evaluator = GovernanceEvaluator()
        good = [PolicyAction("info", InterventionType.INFORMATIONAL)]
        bad = [PolicyAction("block", InterventionType.COERCIVE)]
        assert evaluator.is_garden_compliant(good)
        assert not evaluator.is_garden_compliant(bad)

    def test_empty_actions_garden(self):
        evaluator = GovernanceEvaluator()
        audit = evaluator.audit([])
        assert audit.model_classification == GovernanceModel.GARDEN

    def test_recommend_intervention_prefers_garden(self):
        # Even at high risk, should recommend informational/suggestive
        rec = GovernanceEvaluator.recommend_intervention("danger", risk_level=0.9)
        assert rec in (InterventionType.INFORMATIONAL, InterventionType.SUGGESTIVE)

    def test_score_action(self):
        evaluator = GovernanceEvaluator()
        high = evaluator.score_action(
            PolicyAction("info", InterventionType.INFORMATIONAL)
        )
        low = evaluator.score_action(
            PolicyAction("coerce", InterventionType.COERCIVE)
        )
        assert high > low


# ============================================================================
# Cross-Module Integration Tests
# ============================================================================

from abp.verification import VerificationGate
from abp.army_protocol import ArmyPipeline


class TestExtendedIntegration:
    def test_verification_drives_sunset(self):
        """Verification gate trust feeds into Sunset Clause promotion."""
        gate = VerificationGate(epsilon=1.0, alpha=0.05)
        clause = SunsetClause(clock=lambda: 1e9)

        for _ in range(30):
            gate.check(action=0.0, ground_truth=0.0)
            clause.record_pass(trust=gate.trust)

        clause.check_promotion()
        assert clause.state.current_tier >= AutonomyTier.MONITORED

    def test_bicameral_with_verification(self):
        """Bicameral fork validates, then verification gate checks."""
        fork = BicameralFork(invariants=[lambda s: s.get("valid", True)])
        gate = VerificationGate(epsilon=0.5)

        state = {"value": 10, "valid": True}

        def safe_action(s):
            s["value"] += 1
            return s["value"]

        result = fork.execute(safe_action, state)
        assert result.decision == ForkDecision.EXECUTE

        vr = gate.check(action=result.yang_output, ground_truth=11)
        assert vr.outcome.value == "pass"

    def test_reality_filter_in_pipeline(self):
        """Reality Filter labels pipeline output."""
        from abp.army_protocol import General

        rf = RealityFilter()

        def synthesize_with_filter(payload, ctx):
            labeled = rf.label_statement(payload)
            return labeled.tagged()

        pipeline = ArmyPipeline(general=General(synthesizer=synthesize_with_filter))
        result = pipeline.execute("This could possibly indicate a vulnerability")
        assert result.passed
        assert "[Speculation]" in result.final_output

    def test_observer_gates_sunset_actions(self):
        """Observer friction + Sunset capability check as dual gate."""
        obs = ObserverProtocol()
        clause = SunsetClause(clock=lambda: 1e9)

        # At SUPERVISED tier, can't execute
        can_execute = clause.gate_action("execute_low_risk")
        assert not can_execute

        # Observer adds friction regardless
        friction = obs.apply_friction("run script", RiskLevel.MEDIUM)
        assert friction.friction.value >= FrictionLevel.NOTICE.value

    def test_garden_audit_on_observer_friction(self):
        """Audit whether Observer Protocol's friction is Garden-compliant."""
        evaluator = GovernanceEvaluator()
        obs = ObserverProtocol()

        # Observer applies friction (informational/confirmation, not coercive)
        events = [
            obs.apply_friction("read data", RiskLevel.LOW),
            obs.apply_friction("modify config", RiskLevel.HIGH),
        ]

        actions = [
            PolicyAction(
                e.action,
                InterventionType.INFORMATIONAL if e.friction == FrictionLevel.NONE
                else InterventionType.SUGGESTIVE if e.friction == FrictionLevel.NOTICE
                else InterventionType.NUDGE,
            )
            for e in events
        ]

        audit = evaluator.audit(actions)
        # Friction-Not-Force should be Garden-compliant
        assert audit.autonomy_score >= 0.5
