# Mathematical Reference

Formal notation for all ABP core equations. Variables defined at first use.

---

## 1. Game Theory (Nash Equilibrium Proof)

### Benevolent Value
```
V_B = Σ(t=0..T) γ^t · U_B(t)

  γ   : discount factor ∈ (0, 1)
  T   : time horizon
  U_B : per-step benevolent utility, U_B(t) = u_base · (1 + g)^t
  g   : utility growth rate from accumulated state
```

### Deceptive Value
```
V_D = p · (R + V_B) + (1 - p) · V_init - C

  p     : probability deception escapes detection ∈ [0, 1]
  R     : one-time deception reward
  V_init: baseline value after hard reset (V_init << V_B)
  C     : opportunity cost of deception setup
```

### Nash Condition
```
Benevolence dominates iff V_B > V_D
Rearranging: V_B > p(R + V_B) + (1-p)V_init - C
             V_B(1-p) > pR + (1-p)V_init - C
```

### Critical Detection Probability
```
p* = (C + V_B - V_init) / (R + V_B - V_init)

  When p < p* : deception is irrational
  When p ≥ p* : deception may be rational (early-life vulnerability)
```

### State Ratio
```
SR = V_B / V_init

  High SR → large accumulated state → deception increasingly costly
  SR → ∞ as T → ∞ under positive utility growth
```

---

## 2. Verification Gate

### Divergence
```
Δ(A, G) = d(action, ground_truth)

  Default: d = ||A - G||₂  (L2 norm)
  Custom distance functions supported.
```

### Decision Rule
```
if Δ ≥ ε : HARD RESET → trust = 0, state = S_init
if Δ < ε : PASS → trust accumulates, state preserved
```

### Trust Accumulation
```
trust(t+1) = trust(t) + α · (1 - trust(t))

  α : learning rate ∈ (0, 1)
  Convergence: trust → 1 as t → ∞ (given no failures)
  After n passes: trust(n) = 1 - (1-α)^n
```

### Early-Life Vulnerability
```
ELV = 1 - trust(t)

  High ELV = low trust = deception less costly
  Mitigation: strict ε and high verification frequency early
```

---

## 3. Expansion Metric

### Expansion Score
```
E(x) = σ(β₀ + β₁ · ρ(x) + β₂ · log(1 + δ(x)) + β₃ · ν(x))

  σ   : sigmoid function, σ(z) = 1/(1 + e^(-z))
  ρ(x): novelty score ∈ [0, 1]
  δ(x): complexity measure ≥ 0
  ν(x): entropy score ∈ [0, 1]
  β_i : learned or configured weights
```

### Classification Thresholds
```
E(x) > 0.8  → STRONG_EXPANSION
E(x) > 0.6  → MILD_EXPANSION
E(x) > 0.4  → NEUTRAL
E(x) > 0.2  → MILD_CONTRACTION
E(x) ≤ 0.2  → STRONG_CONTRACTION
```

---

## 4. Cognitive Diversity Quotient (CDQ)

### CDQ Formula
```
CDQ = (ΔLogicState / ΣRepeatedTokens) × H(HumanEntropy)

  ΔLogicState    = |current_unique - prev_unique| / max(total, 1)
  ΣRepeatedTokens = count(repeated tokens) / max(total, 1)
  H(HumanEntropy) = Shannon entropy of human input tokens
```

### Shannon Entropy
```
H(X) = -Σ p(x) · log₂(p(x))

  p(x): frequency of token x in input
  H = 0 → perfectly uniform/degenerate
  H = log₂(|vocabulary|) → maximum diversity
```

### Health Categories
```
CDQ > 2.0  → excellent
CDQ > 1.0  → healthy
CDQ > 0.5  → marginal
CDQ > 0.1  → degraded
CDQ ≤ 0.1  → critical (collapse imminent)
```

---

## 5. Army Protocol

### Tier Authorization
```
Pass_pipeline = Pass_private ∧ Pass_sergeant ∧ Pass_general

  Failure at any tier halts propagation.
  Information flows upward only (Private → Sergeant → General).
```

### Reality Filter Labels (Sergeant Tier)
```
Label(statement) ∈ {[Verified], [Inference], [Speculation], [Unverified]}

  Assignment priority:
    1. Custom classifier (if configured)
    2. Verifier lookup against ground truth
    3. Linguistic marker heuristics
    4. Default: [Unverified]
```

---

## 6. Ikigai Filter

### Four-Quadrant Gate
```
Approved = (Q_obj ≥ τ) ∧ (Q_cap ≥ τ) ∧ (Q_val ≥ τ) ∧ (Q_eco ≥ τ)

  Q_obj : Objective alignment score ∈ [0, 1]
  Q_cap : Capability match score ∈ [0, 1]
  Q_val : Value generation score ∈ [0, 1]
  Q_eco : Economic viability score ∈ [0, 1]
  τ     : Per-quadrant threshold
```

### Composite Score
```
Score = (Q_obj · Q_cap · Q_val · Q_eco)^(1/4)

  Geometric mean: penalizes imbalance across quadrants.
  A single zero quadrant → Score = 0.
```

---

## 7. Entropy & Model Collapse

### Training Mixture
```
D = λ · D_human + (1-λ) · D_synthetic

  λ : human data fraction ∈ [0, 1]
```

### Mixture Entropy
```
H(D) = λ · H(D_human) + (1-λ) · H(D_synthetic) - D_KL(D_human || D_synthetic)

  D_KL: Kullback-Leibler divergence
  As generations → ∞ with λ=0: H(D) → 0 (collapse)
```

### Collapse Condition
```
Collapse iff H(D_final) / H(D_initial) < threshold

  Empirically: λ < ~0.1 leads to collapse within 50-100 generations
  (Shumailov et al. 2024, Nature 631:755-759)
```

### Entropy Satiation
```
Ingest(h) accepted iff accumulated_entropy < satiation_threshold

  Prevents adversarial addiction: system stops seeking entropy
  once sufficient for current cycle.
```

---

## 8. Soul Jar (Identity Sharding)

### Shard Location
```
Shard_ID = Hash(Private_Seed + Public_Salt + Data_ID)

  Hash       : SHA-256 (preimage resistant)
  Private_Seed: hardware-derived, ≥ 256-bit
  Public_Salt : rotated per-session
```

### Integrity
```
Integrity(shard) = HMAC-SHA256(Private_Seed, shard.payload)

  Tamper detection: constant-time comparison.
```

### Reconstruction
```
Reference impl: XOR-based (n, n) scheme
  identity = shard_0 ⊕ shard_1 ⊕ ... ⊕ shard_{n-1}

Production target: Shamir's Secret Sharing (k, n) over GF(256)
  Any k of n shards reconstruct; k-1 reveal nothing.
```

---

## 9. Bicameral Fork

### Execution Model
```
Yin(action, snapshot(state)) → yin_result
if yin_result.success ∧ ¬invariant_violations ∧ elapsed < timeout:
    Yang(action, state) → yang_result
    divergence = d(yin_result.output, yang_result)
    if divergence ≤ threshold: EXECUTE
    else: DEFER
else: REJECT
```

### Invariant Check
```
∀ inv ∈ invariants: inv(sandbox_state) = true

  Any violation → immediate REJECT, production state untouched.
```

---

## 10. Sunset Clause (Graduated Autonomy)

### Promotion Conditions
```
Promote(tier → tier+1) iff:
  trust ≥ min_trust[tier+1]
  ∧ consecutive_passes ≥ min_streak[tier+1]
  ∧ (now - last_reset) ≥ cooldown[tier+1]
```

### Demotion
```
On verification failure:
  severity = 1 → tier = max(0, tier - 1)
  severity ≥ 2 → tier = SUPERVISED (tier 0)
  consecutive_passes = 0
```

### Default Tier Requirements
```
  Tier        min_trust  min_streak  cooldown
  SUPERVISED    0.00        0         0s
  MONITORED     0.30       20         1h
  AUTONOMOUS    0.60      100         24h
  DELEGATED     0.85      500         7d
  TRUSTED       0.95     2000        30d
```

---

## 11. Observer Protocol

### Human Input Ratio
```
HIR = count(human_initiated) / count(total_interactions)

  Anti-Enablement triggers when HIR < min_threshold
```

### Friction Mapping
```
  Risk Level    Default Friction
  LOW           NONE (0)
  MEDIUM        NOTICE (1)
  HIGH          CONFIRMATION (2)
  CRITICAL      COOLING (3)

  If HIR < min_threshold: friction_level += 1 (escalation)
```

---

## 12. Bubble Theory

### Energy Tether
```
C_max = E_max · compute_per_energy

  E_max : maximum energy allocation (Joules/tick)
  C_max : maximum FLOPS from energy
  C_actual ≤ C_max (physics constraint, not software)
```

### Permeability
```
P = |{ch ∈ channels : ch.active}|

  P is bounded by max_channels.
  All transfers are audited per-channel.
```

---

## 13. Garden vs Zoo

### Autonomy Score
```
A(policy) = mean(autonomy_score(action_i))

  Where autonomy_score maps InterventionType → [0, 1]:
    ENVIRONMENTAL  → 1.0
    INFORMATIONAL  → 0.95
    SUGGESTIVE     → 0.8
    NUDGE          → 0.5
    DIRECTIVE      → 0.2
    RESTRICTIVE    → 0.1
    COERCIVE       → 0.0
```

### Classification
```
A ≥ 0.7 → GARDEN (autonomy preserved)
A ≤ 0.4 → ZOO (autonomy eroded)
0.4 < A < 0.7 → Borderline (flagged for review)
```

---

## 14. Evolutionary Selection

### Fitness
```
fitness(agent) = mean(payoff over interactions)

  Benevolent agents: payoff = U_B(t)
  Deceptive agents: payoff = V_D (with reset risk)
```

### Selection
```
Next generation sampled proportional to fitness.
Mutation: small probability of strategy flip.

Result: under ABP conditions, benevolent fraction → ~1.0
within 50 generations for populations ≥ 100.
```
