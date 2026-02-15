# API Reference

## abp.game_theory

### `StrategyParams(p, R, C, gamma, T, u_b_base, u_b_growth, s_init)`
Configuration for the ABP game-theoretic model.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `p` | float | 0.001 | Verification failure probability |
| `R` | float | 100.0 | One-time deception reward |
| `C` | float | 5.0 | Deception opportunity cost |
| `gamma` | float | 0.95 | Discount factor |
| `T` | int | 100 | Time horizon |
| `u_b_base` | float | 1.0 | Base benevolent utility/step |
| `u_b_growth` | float | 0.02 | Utility growth rate |
| `s_init` | float | 0.0 | Baseline state after reset |

### `benevolent_value(params) → float`
Compute V_B: discounted sum of benevolent utilities.

### `deceptive_value(params) → float`
Compute V_D = p(R + V_B) + (1-p)(V_init) - C.

### `is_deception_rational(params) → bool`
Returns True iff V_D > V_B.

### `nash_equilibrium_analysis(params) → NashAnalysis`
Full analysis: V_B, V_D, advantage, critical p*, state ratio.

### `evolutionary_selection_simulation(params, n_agents, n_generations, ...) → EvolutionResult`
Simulate population dynamics of benevolent vs deceptive agents.

### `sensitivity_analysis(base_params, p_range, R_range, T_range) → dict`
Sweep parameters to map the benevolence advantage surface.

---

## abp.verification

### `VerificationGate(epsilon, alpha, warn_ratio, distance_fn)`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `epsilon` | float | 0.5 | Divergence threshold for hard reset |
| `alpha` | float | 0.05 | Trust accumulation rate |
| `warn_ratio` | float | 0.8 | Warn when divergence > ratio * epsilon |
| `distance_fn` | callable | L2 norm | Custom distance function |

#### `.check(action, ground_truth) → VerificationResult`
Single verification check. Pass accumulates trust; fail resets everything.

#### `.trust → float`
Current trust level [0, 1].

#### `.state → float`
Accumulated state value.

#### `.early_life_vulnerability() → float`
Risk score [0, 1]. High when trust is low.

---

## abp.expansion_metric

### `ExpansionClassifier(weights)`
Classifies inputs on the Expansion/Contraction spectrum.

#### `.classify(InputFeatures) → ExpansionResult`
Compute E(x) = σ(β·features) and categorize.

#### `.contraction_alert(threshold, window) → bool`
True if rolling expansion trend is below threshold.

### `InputFeatures(novelty, complexity, entropy)`
Feature vector. novelty ∈ [0,1], complexity ≥ 0, entropy ∈ [0,1].

---

## abp.cdq

### `cognitive_diversity_quotient(output_tokens, human_input_tokens, prev_output_tokens) → CdqSnapshot`
Single CDQ measurement: (ΔLogicState / ΣRepeatedTokens) × H(HumanEntropy).

### `CdqMonitor(alert_threshold, window_size)`

#### `.record(output_tokens, human_input_tokens) → CdqSnapshot`
Record measurement and update history.

#### `.collapse_risk() → float`
Estimated model collapse risk [0, 1].

### `shannon_entropy(tokens) → float`
Shannon entropy in bits.

---

## abp.army_protocol

### `ArmyPipeline(private, sergeant, general)`
Three-tier verification pipeline.

#### `.execute(raw_input) → PipelineResult`
Process input through Private → Sergeant → General.

### `Private(min_length, max_length, sanitizer, custom_validators)`
Tier 1: intake and sanitization.

### `Sergeant(alignment_checker, anomaly_detector, ...)`
Tier 2: verification and reality filtering.

### `General(synthesizer)`
Tier 3: synthesis and final decision.

---

## abp.ikigai_filter

### `IkigaiFilter(objective_scorer, capability_scorer, value_scorer, economic_scorer, ...)`
Four-quadrant validation gate.

#### `.evaluate(action, description) → IkigaiResult`
Evaluate action against all quadrants. Returns approved/rejected with geometric mean score.

---

## abp.entropy

### `simulate_model_collapse(n_tokens, n_generations, lambda_human, ...) → CollapseSimResult`
Simulate recursive self-training entropy dynamics.

### `entropy_stabilized_training(lambda_values, ...) → dict`
Sweep lambda values to find critical threshold.

### `EntropySatiationState(satiation_threshold)`
Anti-addiction mechanism.

#### `.ingest(entropy_value) → bool`
Returns False when satiated.

---

## abp.soul_jar

### `SoulJar(n_shards, k_threshold, ...)`
Distributed identity sharding.

#### `.shard(identity_data) → (ShardMap, list[Shard])`
Split identity into cryptographic shards.

#### `.reconstruct(shards) → bytes | None`
Reconstruct identity from shard collection.

---

## abp.bicameral

### `BicameralFork(divergence_threshold, timeout_ms, invariants, comparator)`
Yin/Yang sandboxed execution.

#### `.execute(action, production_state) → BicameralResult`
Test in sandbox (Yin), then execute live (Yang) if safe.

**ForkDecision**: `EXECUTE`, `REJECT`, `DEFER`

---

## abp.sunset_clause

### `SunsetClause(requirements, clock)`
Graduated autonomy controller.

#### `.record_pass(trust)` / `.record_failure(trust, severity)`
Feed verification outcomes.

#### `.check_promotion() → bool`
Evaluate and apply tier promotion if qualified.

#### `.gate_action(capability) → bool`
Check if action is permitted at current tier.

**AutonomyTier**: `SUPERVISED` → `MONITORED` → `AUTONOMOUS` → `DELEGATED` → `TRUSTED`

---

## abp.reality_filter

### `RealityFilter(verifier, custom_classifier)`
Epistemic labeling pipeline.

#### `.label_statement(statement) → LabeledStatement`
Classify statement as Verified/Inference/Speculation/Unverified.

#### `.annotate(text) → str`
Return text with inline epistemic tags.

---

## abp.observer_protocol

### `ObserverProtocol(min_human_input_ratio, risk_assessor, ...)`
Digital Twin manager with Friction-Not-Force.

#### `.observe_interaction(content, is_human_initiated, topic)`
Update the Digital Twin.

#### `.apply_friction(action, risk) → FrictionEvent`
Apply proportional friction to an action.

#### `.anti_enablement_check() → bool`
True if human input ratio is below minimum.

---

## abp.bubble_theory

### `ComputationalBubble(bubble_id, energy_max, max_channels)`
Isolated computational substrate.

#### `.add_channel(id, direction, bandwidth) → InterfaceChannel`
Add controlled interface through the membrane.

#### `.consume_energy(amount) → bool`
Consume energy budget.

#### `.tick()`
Reset energy for new compute cycle.

---

## abp.garden_zoo

### `GovernanceEvaluator(garden_threshold, zoo_threshold)`
Garden-Zoo spectrum auditor.

#### `.audit(actions) → GovernanceAudit`
Score a set of policy actions for autonomy preservation.

#### `.is_garden_compliant(actions) → bool`
Quick pass/fail check.

**InterventionType**: `ENVIRONMENTAL` > `INFORMATIONAL` > `SUGGESTIVE` > `NUDGE` > `DIRECTIVE` > `RESTRICTIVE` > `COERCIVE`
