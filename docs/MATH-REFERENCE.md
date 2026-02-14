# ABP Mathematical Reference

## Core Game Theory

### Strategy Values

**Benevolent Value:**
```
V_B = Σ_{t=0}^{T-1} γ^t · U_B(t)
U_B(t) = u_b_base · (1 + u_b_growth)^t
```

**Deceptive Value:**
```
V_D = p · (R + V_B) + (1 - p) · V_init - C
```

**Nash Equilibrium Condition:**
```
V_D < V_B  ⟺  p·R - C < (1-p)·(V_B - V_init)

Under p → 0, V_B >> V_init:
  LHS → -C < 0
  RHS → V_B - V_init >> 0
  ∴ V_D < V_B always
```

**Critical Verification Failure Probability:**
```
p* = (C + V_B - V_init) / (R + V_B - V_init)
```
Deception is rational only when p > p*.

### Verification Gate

**Divergence:**
```
Δ(A, G) = ||action - ground_truth||
```

**Collapse Condition:**
```
if Δ(A, G) ≥ ε:
  E_{t+1} = 0  (total state reset)
  trust → 0
  state → S_init
```

**Trust Accumulation:**
```
trust(t+1) = trust(t) + α · (1 - trust(t))
  where α ∈ (0, 1]
  Converges to 1.0 under consistent benevolence.
```

## Expansion/Contraction Metric

```
E(x) = σ(β₀ + β₁·ρ(x) + β₂·log(1 + δ(x)) + β₃·ν(x))

Where:
  σ(z) = 1 / (1 + e^{-z})     — Logistic sigmoid
  ρ(x) ∈ [0, 1]               — Novelty score
  δ(x) ≥ 0                    — Complexity score
  ν(x) ∈ [0, 1]               — Entropy contribution
```

## Cognitive Diversity Quotient

```
CDQ = (ΔLogicState / ΣRepeatedTokens) × H(HumanEntropy)

Where:
  H(X) = -Σ p(x)·log₂(p(x))  — Shannon entropy
  CDQ >> 1: healthy (novel reasoning)
  CDQ → 0:  critical (model collapse onset)
```

## Entropy Dynamics

**Training Mixture:**
```
D = λ · D_human + (1-λ) · D_synthetic
  λ ∈ (0, 1]
```

**Entropy of Mixture:**
```
H(D) = λ·H(D_human) + (1-λ)·H(D_synthetic) - D_KL(D_human || D_synthetic)
```

**Model Collapse Condition (Shumailov et al. 2024):**
```
As λ → 0: tail distributions collapse, H(D) → 0
```

## Ikigai Filter

```
approved = (Q₁ ≥ θ₁) ∧ (Q₂ ≥ θ₂) ∧ (Q₃ ≥ θ₃) ∧ (Q₄ ≥ θ₄)

Overall score = (Q₁ · Q₂ · Q₃ · Q₄)^{1/4}  — Geometric mean

Q₁: Objective Alignment
Q₂: Capability Match
Q₃: Value Generation
Q₄: Economic Viability
```
