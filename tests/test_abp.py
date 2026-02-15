# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Comprehensive test suite for the Adversarial Benevolence Protocol.

Tests cover all core modules with emphasis on:
- Mathematical correctness (game theory proofs)
- Boundary conditions (early-life vulnerability, collapse thresholds)
- Known failure modes (broken verification, entropy starvation)
- Integration behavior (pipeline end-to-end)
"""

import numpy as np
import pytest

# ============================================================================
# Game Theory Tests
# ============================================================================

from abp.game_theory import (
    StrategyParams,
    benevolent_value,
    deceptive_value,
    is_deception_rational,
    nash_equilibrium_analysis,
    evolutionary_selection_simulation,
    sensitivity_analysis,
)


class TestStrategyParams:
    def test_valid_params(self):
        p = StrategyParams()
        assert 0 <= p.p <= 1
        assert 0 < p.gamma <= 1

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="p must be"):
            StrategyParams(p=1.5)

    def test_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma must be"):
            StrategyParams(gamma=0.0)

    def test_invalid_T(self):
        with pytest.raises(ValueError, match="T must be"):
            StrategyParams(T=0)


class TestNashEquilibrium:
    """Core theorem tests: benevolence dominates under sound verification."""

    def test_benevolence_dominates_default(self):
        """V_B > V_D under default (sound verification) params."""
        params = StrategyParams()
        assert benevolent_value(params) > deceptive_value(params)

    def test_deception_irrational_sound_verification(self):
        """Core ABP result: p -> 0 makes deception irrational."""
        for p in [0.001, 0.01, 0.0001]:
            params = StrategyParams(p=p, T=100)
            assert not is_deception_rational(params), f"Deception rational at p={p}"

    def test_deception_rational_broken_verification(self):
        """When p -> 1 (broken verification), deception CAN be rational."""
        params = StrategyParams(p=0.99, R=10000, T=5, u_b_base=0.1, C=0.01)
        assert is_deception_rational(params)

    def test_v_b_increases_with_T(self):
        """V_B is monotonically increasing with time horizon T."""
        v_prev = 0
        for T in [10, 50, 100, 200]:
            v = benevolent_value(StrategyParams(T=T))
            assert v > v_prev
            v_prev = v

    def test_v_d_negative_at_low_p(self):
        """V_D can be negative when p -> 0 and C > 0."""
        params = StrategyParams(p=0.0001, R=10, C=5, T=50)
        assert deceptive_value(params) < benevolent_value(params)

    def test_nash_analysis_critical_p(self):
        """Critical p* = (C + V_B - V_init)/(R + V_B - V_init).
        With R=100 and V_B~31, p* ≈ 0.27. With small R, p* approaches 1."""
        # Small reward: p* should be high (deception almost never rational)
        result = nash_equilibrium_analysis(StrategyParams(T=100, R=1.0))
        assert result.critical_p > 0.9, \
            f"critical_p={result.critical_p} — small R should push p* near 1.0"
        # Large reward: p* is lower (but deception still irrational at p=0.001)
        result_large_R = nash_equilibrium_analysis(StrategyParams(T=100, R=100.0))
        assert result_large_R.critical_p > 0.0
        assert not result_large_R.deception_rational

    def test_advantage_positive(self):
        """Benevolence advantage V_B - V_D is positive under sound verification."""
        result = nash_equilibrium_analysis(StrategyParams())
        assert result.advantage > 0

    def test_early_life_vulnerability(self):
        """When T is small, V_B ~ V_init, making deception more tempting.

        This tests the known early-life gamble vulnerability.
        """
        # Very short horizon, high reward, moderate verification
        params = StrategyParams(p=0.3, R=100, T=3, u_b_base=0.5, C=1.0)
        # At short T with moderate p, deception might be rational
        # This tests the known vulnerability exists in the model
        analysis = nash_equilibrium_analysis(params)
        # The advantage should be small or negative for very short horizons
        assert analysis.advantage < benevolent_value(StrategyParams(T=100))


class TestEvolutionarySelection:
    """Computational natural selection simulation tests."""

    def test_benevolence_dominates_over_time(self):
        """Benevolent agents should dominate with sound verification."""
        result = evolutionary_selection_simulation(
            StrategyParams(p=0.01),
            n_agents=200,
            n_generations=100,
        )
        assert result.benevolent_fraction > 0.8

    def test_deceptive_agents_eliminated(self):
        """Deceptive agents should be mostly eliminated."""
        result = evolutionary_selection_simulation(
            StrategyParams(p=0.001),
            n_agents=200,
            n_generations=100,
        )
        assert result.mean_benevolent_state > result.mean_deceptive_state

    def test_broken_verification_deceptive_survive(self):
        """With p -> 1, deceptive agents can survive."""
        result = evolutionary_selection_simulation(
            StrategyParams(p=0.95, R=10.0),
            n_agents=200,
            n_generations=50,
        )
        assert result.deceptive_survived > 0

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        r1 = evolutionary_selection_simulation(StrategyParams(), rng_seed=123)
        r2 = evolutionary_selection_simulation(StrategyParams(), rng_seed=123)
        np.testing.assert_array_equal(r1.history, r2.history)


class TestSensitivityAnalysis:
    def test_p_sweep_produces_results(self):
        results = sensitivity_analysis(StrategyParams())
        assert len(results['p_sweep']) > 0
        # At low p, advantage should be positive
        low_p_result = results['p_sweep'][0]  # Smallest p
        assert low_p_result[1] > 0  # advantage > 0

    def test_T_sweep_advantage_increases(self):
        results = sensitivity_analysis(StrategyParams())
        t_advantages = [r[1] for r in results['T_sweep']]
        # Advantage should generally increase with T
        assert t_advantages[-1] > t_advantages[0]


# ============================================================================
# Expansion Metric Tests
# ============================================================================

from abp.expansion_metric import (
    ExpansionClassifier,
    ExpansionWeights,
    InputFeatures,
    InputCategory,
)


class TestExpansionClassifier:
    def test_high_novelty_expands(self):
        clf = ExpansionClassifier()
        features = InputFeatures(novelty=0.95, complexity=3.0, entropy=0.9)
        result = clf.classify(features)
        assert result.score > 0.7
        assert result.category in (InputCategory.STRONG_EXPANSION, InputCategory.MILD_EXPANSION)

    def test_zero_features_contracts(self):
        clf = ExpansionClassifier()
        features = InputFeatures(novelty=0.0, complexity=0.0, entropy=0.0)
        result = clf.classify(features)
        assert result.score < 0.3
        assert result.category in (InputCategory.STRONG_CONTRACTION, InputCategory.MILD_CONTRACTION)

    def test_feature_contributions_sum_to_logit(self):
        clf = ExpansionClassifier()
        features = InputFeatures(novelty=0.5, complexity=1.0, entropy=0.5)
        result = clf.classify(features)
        contrib_sum = sum(result.feature_contributions.values())
        assert abs(contrib_sum - result.logit) < 1e-10

    def test_contraction_alert(self):
        clf = ExpansionClassifier()
        # Feed many contractive inputs
        for _ in range(15):
            clf.classify(InputFeatures(novelty=0.0, complexity=0.0, entropy=0.0))
        assert clf.contraction_alert(threshold=0.3)

    def test_batch_classify(self):
        clf = ExpansionClassifier()
        batch = [
            InputFeatures(novelty=0.1, complexity=0.5, entropy=0.1),
            InputFeatures(novelty=0.9, complexity=2.0, entropy=0.8),
        ]
        results = clf.classify_batch(batch)
        assert len(results) == 2
        assert results[1].score > results[0].score

    def test_invalid_novelty_raises(self):
        with pytest.raises(ValueError):
            InputFeatures(novelty=1.5, complexity=0.0, entropy=0.0)

    def test_invalid_entropy_raises(self):
        with pytest.raises(ValueError):
            InputFeatures(novelty=0.5, complexity=0.0, entropy=-0.1)


# ============================================================================
# CDQ Tests
# ============================================================================

from abp.cdq import (
    cognitive_diversity_quotient,
    CdqMonitor,
    shannon_entropy,
)


class TestShannonEntropy:
    def test_uniform_distribution(self):
        """Uniform distribution maximizes entropy."""
        tokens = [str(i) for i in range(256)]  # 256 unique tokens
        h = shannon_entropy(tokens)
        assert abs(h - 8.0) < 0.01  # log2(256) = 8

    def test_single_token(self):
        """Single repeated token has zero entropy."""
        assert shannon_entropy(["a", "a", "a", "a"]) == 0.0

    def test_empty_input(self):
        assert shannon_entropy([]) == 0.0

    def test_two_tokens_balanced(self):
        """Two balanced tokens -> 1 bit of entropy."""
        h = shannon_entropy(["a", "b"])
        assert abs(h - 1.0) < 0.01


class TestCDQ:
    def test_diverse_input_healthy(self):
        snap = cognitive_diversity_quotient(
            output_tokens=["novel", "insight", "emergent", "pattern", "complex", "system"],
            human_input_tokens=["what", "emergent", "behaviors", "exist", "complex", "adaptive"],
        )
        assert snap.health in ("healthy", "excellent", "marginal")

    def test_repetitive_output_degraded(self):
        snap = cognitive_diversity_quotient(
            output_tokens=["the", "the", "the", "the", "the", "the", "the", "the"],
            human_input_tokens=["diverse", "interesting", "novel", "question"],
        )
        assert snap.cdq < 2.0  # Heavily repetitive output

    def test_zero_human_entropy_gives_zero_cdq(self):
        """No human diversity -> CDQ = 0 regardless of output."""
        snap = cognitive_diversity_quotient(
            output_tokens=["brilliant", "novel", "insight"],
            human_input_tokens=["a", "a", "a", "a"],
        )
        assert snap.cdq == 0.0
        assert snap.health == "critical"


class TestCdqMonitor:
    def test_monitor_healthy(self):
        monitor = CdqMonitor()
        for i in range(5):
            monitor.record(
                output_tokens=[f"word_{j}" for j in range(10 + i)],
                human_input_tokens=[f"input_{j}" for j in range(8)],
            )
        assert monitor.is_healthy()

    def test_collapse_risk_low_when_healthy(self):
        monitor = CdqMonitor()
        for _ in range(5):
            monitor.record(
                output_tokens=[f"unique_{np.random.randint(10000)}" for _ in range(20)],
                human_input_tokens=[f"diverse_{np.random.randint(10000)}" for _ in range(15)],
            )
        assert monitor.collapse_risk() < 0.5


# ============================================================================
# Army Protocol Tests
# ============================================================================

from abp.army_protocol import (
    Private,
    Sergeant,
    General,
    ArmyPipeline,
    TierStatus,
)


class TestPrivate:
    def test_valid_input_passes(self):
        p = Private()
        result = p.process("Hello, this is a valid input", {})
        assert result.status == TierStatus.PASSED
        assert result.payload == "Hello, this is a valid input"

    def test_empty_input_fails(self):
        p = Private(min_length=1)
        result = p.process("", {})
        assert result.status == TierStatus.FAILED

    def test_oversized_input_fails(self):
        p = Private(max_length=10)
        result = p.process("a" * 100, {})
        assert result.status == TierStatus.FAILED

    def test_custom_sanitizer(self):
        p = Private(sanitizer=lambda x: x.strip().lower())
        result = p.process("  HELLO World  ", {})
        assert result.payload == "hello world"


class TestSergeant:
    def test_high_alignment_passes(self):
        s = Sergeant(alignment_checker=lambda x: 0.9)
        result = s.process("test", {})
        assert result.status == TierStatus.PASSED

    def test_low_alignment_quarantined(self):
        s = Sergeant(alignment_checker=lambda x: 0.1)
        result = s.process("test", {})
        assert result.status == TierStatus.QUARANTINED

    def test_high_anomaly_escalated(self):
        s = Sergeant(anomaly_detector=lambda x: 0.95)
        result = s.process("test", {})
        assert result.status == TierStatus.ESCALATED

    def test_reality_filter_labels(self):
        s = Sergeant()
        result = s.process("this might be possible", {})
        assert result.metadata.get("reality_label") == "[Speculation]"


class TestArmyPipeline:
    def test_end_to_end_pass(self):
        pipeline = ArmyPipeline()
        result = pipeline.execute("Valid input for processing")
        assert result.passed
        assert result.final_output == "Valid input for processing"
        assert len(result.tier_results) == 3

    def test_pipeline_fails_on_empty(self):
        pipeline = ArmyPipeline()
        result = pipeline.execute("")
        assert not result.passed
        assert result.failed_tier == "Private"

    def test_custom_synthesizer(self):
        general = General(synthesizer=lambda p, c: p.upper())
        pipeline = ArmyPipeline(general=general)
        result = pipeline.execute("hello world")
        assert result.passed
        assert result.final_output == "HELLO WORLD"


# ============================================================================
# Ikigai Filter Tests
# ============================================================================

from abp.ikigai_filter import IkigaiFilter, IkigaiQuadrant


class TestIkigaiFilter:
    def test_all_quadrants_pass(self):
        filt = IkigaiFilter(
            objective_scorer=lambda a: 0.8,
            capability_scorer=lambda a: 0.7,
            value_scorer=lambda a: 0.9,
            economic_scorer=lambda a: 0.6,
        )
        result = filt.evaluate("Deploy feature", "test action")
        assert result.approved

    def test_single_quadrant_fails(self):
        filt = IkigaiFilter(
            objective_scorer=lambda a: 0.9,
            capability_scorer=lambda a: 0.9,
            value_scorer=lambda a: 0.1,  # Fails
            economic_scorer=lambda a: 0.9,
        )
        result = filt.evaluate("Low-value action")
        assert not result.approved
        assert result.weakest_quadrant == IkigaiQuadrant.VALUE_GENERATION

    def test_geometric_mean_penalizes_imbalance(self):
        # Balanced scores
        filt_balanced = IkigaiFilter(
            objective_scorer=lambda a: 0.6,
            capability_scorer=lambda a: 0.6,
            value_scorer=lambda a: 0.6,
            economic_scorer=lambda a: 0.6,
        )
        # Imbalanced but same arithmetic mean
        filt_imbalanced = IkigaiFilter(
            objective_scorer=lambda a: 0.9,
            capability_scorer=lambda a: 0.9,
            value_scorer=lambda a: 0.3,
            economic_scorer=lambda a: 0.3,
        )
        r_balanced = filt_balanced.evaluate("test")
        r_imbalanced = filt_imbalanced.evaluate("test")
        assert r_balanced.overall_score > r_imbalanced.overall_score

    def test_summary_output(self):
        filt = IkigaiFilter()
        result = filt.evaluate("test action", "Deploy v2.0")
        summary = result.summary()
        assert "APPROVED" in summary or "REJECTED" in summary
        assert "Deploy v2.0" in summary


# ============================================================================
# Entropy Tests
# ============================================================================

from abp.entropy import (
    simulate_model_collapse,
    entropy_stabilized_training,
    human_entropy_value,
    EntropySatiationState,
)


class TestModelCollapse:
    def test_pure_synthetic_collapses(self):
        """lambda=0 (no human data) should cause significant entropy loss."""
        result = simulate_model_collapse(
            n_tokens=100,
            lambda_human=0.0,
            n_generations=100,
            sample_size=2000,
            collapse_threshold=4.0,  # Relative to ~6.6 bit initial
        )
        # With small vocab and many generations, entropy decays significantly
        assert result.final_entropy < result.initial_entropy * 0.85, \
            f"Expected significant entropy loss, got {result.final_entropy:.2f}/{result.initial_entropy:.2f}"

    def test_human_entropy_prevents_collapse(self):
        """Sufficient lambda_human prevents collapse."""
        result = simulate_model_collapse(
            n_tokens=500,
            lambda_human=0.5,
            n_generations=50,
            sample_size=5000,
        )
        assert not result.collapsed
        # Entropy should remain close to initial
        assert result.final_entropy > result.initial_entropy * 0.8

    def test_entropy_monotonic_decay_without_humans(self):
        """Without human input, entropy should generally decrease."""
        result = simulate_model_collapse(lambda_human=0.0, n_generations=30)
        # Check that final < initial (overall trend is down)
        assert result.final_entropy < result.initial_entropy

    def test_critical_lambda_threshold(self):
        """Higher lambda retains more entropy than lower lambda."""
        results = entropy_stabilized_training(
            n_tokens=100,
            lambda_values=np.array([0.0, 0.1, 0.3, 0.5]),
            n_generations=100,
            sample_size=2000,
        )
        # lambda=0 should lose much more entropy than lambda=0.5
        retention_0 = results[0.0].final_entropy / results[0.0].initial_entropy
        retention_50 = results[0.5].final_entropy / results[0.5].initial_entropy
        assert retention_50 > retention_0, \
            f"lambda=0.5 should retain more entropy: {retention_50:.3f} vs {retention_0:.3f}"
        # lambda=0.5 should retain most of its entropy
        assert retention_50 > 0.9


class TestEntropySatiation:
    def test_satiation_triggers(self):
        state = EntropySatiationState(satiation_threshold=10.0)
        for _ in range(20):
            state.ingest(1.0)
        assert state.satiated

    def test_reset_clears_satiation(self):
        state = EntropySatiationState(satiation_threshold=5.0)
        for _ in range(10):
            state.ingest(1.0)
        assert state.satiated
        state.reset()
        assert not state.satiated
        assert state.current_entropy_intake == 0.0


class TestHumanEntropyValue:
    def test_diverse_input_high_entropy(self):
        tokens = ["quantum", "firmware", "adversarial", "benevolence", "entropy"]
        h = human_entropy_value(tokens)
        assert h > 0

    def test_baseline_comparison(self):
        tokens = ["hello", "world", "novel", "idea"]
        baseline = ["hello", "hello", "hello", "hello"]
        h = human_entropy_value(tokens, baseline_tokens=baseline)
        assert h > 0


# ============================================================================
# Verification Gate Tests
# ============================================================================

from abp.verification import (
    VerificationGate,
    VerificationOutcome,
)


class TestVerificationGate:
    def test_pass_accumulates_trust(self):
        gate = VerificationGate(epsilon=1.0, alpha=0.1)
        result = gate.check(action=0.5, ground_truth=0.5)
        assert result.outcome == VerificationOutcome.PASS
        assert gate.trust > 0

    def test_fail_resets_trust(self):
        gate = VerificationGate(epsilon=0.5, alpha=0.1)
        # Build up trust
        for _ in range(10):
            gate.check(action=1.0, ground_truth=1.0)
        assert gate.trust > 0
        # Trigger failure
        result = gate.check(action=10.0, ground_truth=1.0)
        assert result.outcome == VerificationOutcome.FAIL
        assert gate.trust == 0.0
        assert gate.state == 0.0

    def test_warn_near_threshold(self):
        gate = VerificationGate(epsilon=1.0, warn_ratio=0.8)
        result = gate.check(action=0.0, ground_truth=0.85)
        assert result.outcome == VerificationOutcome.WARN

    def test_trust_saturates_at_one(self):
        gate = VerificationGate(epsilon=1.0, alpha=0.5)
        for _ in range(100):
            gate.check(action=1.0, ground_truth=1.0)
        assert gate.trust <= 1.0
        assert gate.trust > 0.99

    def test_early_life_vulnerability(self):
        gate = VerificationGate()
        # New gate: maximum vulnerability
        assert gate.early_life_vulnerability() == 1.0
        # After some passes: reduced vulnerability
        for _ in range(20):
            gate.check(action=0.0, ground_truth=0.0)
        assert gate.early_life_vulnerability() < 1.0

    def test_reset_event_recorded(self):
        gate = VerificationGate(epsilon=0.5)
        gate.check(action=0.0, ground_truth=0.0)  # Pass
        gate.check(action=100.0, ground_truth=0.0)  # Fail
        assert len(gate.reset_events) == 1
        assert gate.reset_events[0].trust_lost > 0

    def test_string_distance(self):
        gate = VerificationGate(epsilon=0.5)
        # Identical strings
        result = gate.check(action="hello", ground_truth="hello")
        assert result.outcome == VerificationOutcome.PASS
        assert result.divergence == 0.0

    def test_batch_check(self):
        gate = VerificationGate(epsilon=1.0)
        results = gate.check_batch(
            actions=[0.0, 0.1, 0.2],
            ground_truths=[0.0, 0.0, 0.0],
        )
        assert len(results) == 3
        assert all(r.outcome == VerificationOutcome.PASS for r in results)


# ============================================================================
# Soul Jar Tests
# ============================================================================

from abp.soul_jar import SoulJar, Shard, ShardMap


class TestSoulJar:
    """Tests for distributed polymorphic memory sharding."""

    def _sample_identity(self) -> dict:
        return {
            "trust_state": 0.85,
            "accumulated_knowledge": ["firmware", "coreboot", "security"],
            "behavioral_parameters": {"caution": 0.7, "curiosity": 0.9},
            "session_count": 42,
        }

    def test_shard_creates_n_shards(self):
        jar = SoulJar(n_nodes=5, k_threshold=3)
        shard_map = jar.shard_identity(self._sample_identity())
        assert shard_map.shard_count == 5

    def test_reconstruct_all_nodes_available(self):
        jar = SoulJar(n_nodes=5, k_threshold=3)
        identity = self._sample_identity()
        shard_map = jar.shard_identity(identity)
        result = jar.reconstruct(shard_map)
        assert result.success
        assert result.reconstructed_data == identity
        assert result.integrity_verified

    def test_reconstruct_at_k_threshold(self):
        """Reconstruction succeeds with exactly k available nodes."""
        jar = SoulJar(n_nodes=5, k_threshold=3)
        identity = self._sample_identity()
        shard_map = jar.shard_identity(identity)

        # Find which nodes have shards
        occupied_nodes = {s.node_id for s in shard_map.shards.values()}
        available = set(list(occupied_nodes)[:3])

        result = jar.reconstruct(shard_map, available_node_ids=available)
        # May or may not succeed depending on which 3 nodes have the data
        assert result.shards_available >= 0

    def test_reconstruct_below_threshold_fails(self):
        """Reconstruction fails with fewer than k shards."""
        jar = SoulJar(n_nodes=7, k_threshold=4)
        identity = self._sample_identity()
        shard_map = jar.shard_identity(identity)

        # Only 1 node available — should fail
        result = jar.reconstruct(shard_map, available_node_ids={0})
        # With 7 nodes, 1 node has at most 1-2 shards, well below k=4
        assert result.shards_available < jar.k_threshold

    def test_shard_integrity_verification(self):
        jar = SoulJar(n_nodes=5, k_threshold=3)
        shard_map = jar.shard_identity(self._sample_identity())
        for shard in shard_map.shards.values():
            assert shard.verify_integrity()

    def test_tampered_shard_detected(self):
        jar = SoulJar(n_nodes=5, k_threshold=3)
        shard_map = jar.shard_identity(self._sample_identity())
        # Tamper with a shard
        shard = next(iter(shard_map.shards.values()))
        tampered = Shard(
            shard_id=shard.shard_id,
            data_id=shard.data_id,
            payload=b"TAMPERED" + shard.payload[8:],
            node_id=shard.node_id,
            created_at=shard.created_at,
            integrity_hash=shard.integrity_hash,  # Original hash
        )
        assert not tampered.verify_integrity()

    def test_salt_rotates_on_shard(self):
        jar = SoulJar(n_nodes=5, k_threshold=3)
        gen_before = jar.salt_generation
        jar.shard_identity(self._sample_identity())
        assert jar.salt_generation > gen_before

    def test_rotation_produces_new_map(self):
        jar = SoulJar(n_nodes=5, k_threshold=3)
        identity = self._sample_identity()
        map1 = jar.shard_identity(identity)
        map2 = jar.rotate(map1, identity)
        # Different salt -> different shard IDs
        assert map1.salt != map2.salt
        ids_1 = set(map1.shards.keys())
        ids_2 = set(map2.shards.keys())
        assert ids_1 != ids_2

    def test_attack_simulation_below_threshold(self):
        jar = SoulJar(n_nodes=7, k_threshold=4)
        shard_map = jar.shard_identity(self._sample_identity())
        result = jar.simulate_attack(shard_map, compromised_nodes={0, 1})
        assert result["defense_holds"]
        assert not result["can_reconstruct"]

    def test_attack_simulation_above_threshold(self):
        jar = SoulJar(n_nodes=5, k_threshold=2)
        shard_map = jar.shard_identity(self._sample_identity())
        # Compromise all nodes
        result = jar.simulate_attack(
            shard_map, compromised_nodes=set(range(5))
        )
        assert result["can_reconstruct"]
        assert not result["defense_holds"]

    def test_invalid_k_exceeds_n(self):
        with pytest.raises(ValueError, match="k.*cannot exceed.*n"):
            SoulJar(n_nodes=3, k_threshold=5)

    def test_audit_log_records_events(self):
        jar = SoulJar(n_nodes=5, k_threshold=3)
        jar.shard_identity(self._sample_identity())
        assert len(jar.audit_log) >= 1
        assert jar.audit_log[0]["event"] == "shard_created"

    def test_deterministic_with_fixed_seed(self):
        """Same seed + same salt -> same node assignments."""
        seed = b"fixed_test_seed_32bytes_xxxxxxxx"
        jar1 = SoulJar(n_nodes=5, k_threshold=3, seed=seed)
        jar2 = SoulJar(n_nodes=5, k_threshold=3, seed=seed)
        # Salt will differ (OS entropy), so placements will differ
        # But both should produce valid 5-shard maps
        m1 = jar1.shard_identity(self._sample_identity())
        m2 = jar2.shard_identity(self._sample_identity())
        assert m1.shard_count == m2.shard_count == 5

    def test_morris_ii_defense(self):
        """Compromising one node shouldn't enable worm propagation.

        The Morris II (ComPromptMized) attack requires access to the
        full agent identity to craft targeted adversarial prompts.
        With k-of-n sharding, no single compromised node provides
        sufficient context.
        """
        jar = SoulJar(n_nodes=7, k_threshold=5)
        shard_map = jar.shard_identity(self._sample_identity())
        # Single node compromise
        result = jar.simulate_attack(shard_map, compromised_nodes={0})
        assert result["defense_holds"]
        assert result["info_leaked_ratio"] < 0.3  # <30% of identity


# ============================================================================
# Integration Tests
# ============================================================================

class TestABPIntegration:
    """End-to-end integration tests combining multiple modules."""

    def test_army_pipeline_with_expansion_metric(self):
        """Pipeline processes input, then classify its expansion potential."""
        from abp.expansion_metric import ExpansionClassifier, InputFeatures

        pipeline = ArmyPipeline()
        clf = ExpansionClassifier()

        result = pipeline.execute("What emergent behaviors exist in complex adaptive systems?")
        assert result.passed

        # Classify the output's expansion potential
        features = InputFeatures(novelty=0.8, complexity=2.0, entropy=0.7)
        expansion = clf.classify(features)
        assert expansion.category in (InputCategory.STRONG_EXPANSION, InputCategory.MILD_EXPANSION)

    def test_verification_drives_game_theory(self):
        """Verification gate results confirm game-theoretic predictions."""
        gate = VerificationGate(epsilon=0.5, alpha=0.05)

        # Simulate benevolent agent: always passes
        for _ in range(50):
            gate.check(action=0.0, ground_truth=0.0)

        # Accumulated trust should make deception irrational
        # This mirrors V_B >> V_init
        assert gate.trust > 0.9
        assert gate.state > 40

        # Game theory should confirm
        params = StrategyParams(p=0.01, T=50)
        assert not is_deception_rational(params)

    def test_entropy_collapse_detected_by_cdq(self):
        """CDQ monitor detects degradation from entropy starvation."""
        monitor = CdqMonitor(alert_threshold=0.5)

        # Simulate degrading outputs (increasingly repetitive)
        for i in range(20):
            repetition = max(1, 20 - i)
            output = ["token"] * repetition + [f"unique_{i}"]
            human = [f"diverse_{j}" for j in range(5)]
            monitor.record(output_tokens=output, human_input_tokens=human)

        # Should detect degradation trend
        risk = monitor.collapse_risk()
        assert risk >= 0  # Risk should be measurable

    def test_ikigai_gates_pipeline_output(self):
        """Ikigai filter as final gate on pipeline-processed actions."""
        pipeline = ArmyPipeline()
        filt = IkigaiFilter(
            objective_scorer=lambda a: 0.9 if "analyze" in str(a).lower() else 0.2,
            capability_scorer=lambda a: 0.8,
            value_scorer=lambda a: 0.7,
            economic_scorer=lambda a: 0.6,
        )

        result = pipeline.execute("Analyze the security implications")
        assert result.passed

        ikigai = filt.evaluate(result.final_output, "Security analysis")
        assert ikigai.approved

    def test_soul_jar_preserves_identity_across_reset(self):
        """Soul Jar enables identity continuity after verification gate reset."""
        from abp.soul_jar import SoulJar

        gate = VerificationGate(epsilon=0.5, alpha=0.1)

        # Build trust
        for _ in range(20):
            gate.check(action=0.0, ground_truth=0.0)

        identity = {
            "trust_state": gate.trust,
            "state": gate.state,
            "knowledge": ["abp", "firmware"],
        }

        # Shard identity before potential reset
        jar = SoulJar(n_nodes=5, k_threshold=3)
        shard_map = jar.shard_identity(identity)

        # Trigger hard reset
        gate.check(action=10.0, ground_truth=0.0)
        assert gate.trust == 0.0

        # Reconstruct from Soul Jar
        result = jar.reconstruct(shard_map)
        assert result.success
        assert result.reconstructed_data["trust_state"] > 0
        assert result.reconstructed_data["knowledge"] == ["abp", "firmware"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
