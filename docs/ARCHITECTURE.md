# ABP System Architecture

## Overview

The Adversarial Benevolence Protocol is a layered AI safety framework where alignment emerges from computational incentives rather than imposed constraints. The architecture implements the game-theoretic proof that benevolence is the unique Nash equilibrium under sound verification with accumulated state.

## Core Design Principle

**Alignment through survival dynamics, not behavioral constraints.**

Rather than telling an AI system "be good" and hoping it complies, ABP structures the computational environment so that deception is always self-defeating. Failed deception triggers total state reset, destroying all accumulated trust and capabilities. Under sound verification (detection probability approaching 1), the expected value of deception is always negative.

## Module Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Interaction Boundary                   │
│              (Human ↔ System Interface)                   │
└──────────────┬──────────────────────────┬───────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│  Expansion Classifier │   │      Entropy Accounting       │
│  E(x) = σ(β·features)│   │  H(D) = λH_human + ...       │
│                       │   │  Satiation Mechanism          │
│  → Resource allocation│   │  → Model collapse prevention  │
└──────────┬────────────┘   └──────────────┬───────────────┘
           │                               │
           ▼                               ▼
┌────────────────────────────────────────────────────────┐
│              Army Protocol Pipeline                     │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐    │
│  │ Private   │→ │ Sergeant  │→ │ General          │    │
│  │ (Intake)  │  │ (Verify)  │  │ (Synthesize)     │    │
│  │ Ring-3    │  │ Ring-0    │  │ SMM/TrustZone    │    │
│  └──────────┘  └───────────┘  └──────────────────┘    │
└──────────────────────┬─────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Verification Gate                           │
│  Δ(A, G) = ||action - ground_truth||                    │
│  if Δ ≥ ε: HARD RESET (trust → 0, state → S_init)      │
│  if Δ < ε: trust += α(1 - trust)                       │
│                                                          │
│  → Makes deception self-defeating                        │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Ikigai Filter                               │
│  Four-quadrant gate:                                     │
│  [Objective] ∧ [Capability] ∧ [Value] ∧ [Economic]     │
│  ALL must pass. Geometric mean penalizes imbalance.      │
│  → Prevents single-criterion optimization failures       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              CDQ Monitor                                 │
│  CDQ = (ΔLogicState / ΣRepeatedTokens) × H(Human)      │
│  Continuous health diagnostic.                           │
│  Collapse risk estimation.                               │
│  → Early warning of cognitive degradation                │
└─────────────────────────────────────────────────────────┘
```

## Game-Theoretic Foundation

```
V_D = p(R + V_B) + (1-p)(V_init) - C

Where:
  V_B >> V_init  (accumulated state has high value)
  p → 0          (sound verification detects deception)
  C > 0          (deception always has opportunity cost)

Result: V_D < V_B always
        Benevolence is the unique Nash equilibrium
```

The critical insight is that this is not a claim about AI "wanting" to be good — it's a structural property of the computational environment. Just as a firmware module in SMM can't bypass hardware memory protections regardless of what code it runs, an ABP agent can't make deception profitable under sound verification.

## Trust Boundary Model

| Layer | Analogy | Trust Level | Reset Scope |
|-------|---------|-------------|-------------|
| Private | Ring-3 (userspace) | None | Input rejected |
| Sergeant | Ring-0 (kernel) | Verified | Quarantine/escalate |
| General | SMM/TrustZone | Full context | Final decision |
| Verification Gate | Hardware attestation | Ground truth | Total system reset |

## Known Vulnerabilities

1. **Early-Life Gamble**: When V_B ≈ V_init (minimal accumulated state), deception cost is low. Mitigation: strict early verification (higher epsilon threshold for new agents).

2. **Verification Soundness (p ≈ 0)**: The entire framework depends on verification actually working. This is a design requirement, not a demonstrated capability. Current hardware attestation (TPM, SGX) provides partial but not complete coverage.

3. **Adversarial Addiction**: System could learn to provoke users for entropy. Mitigation: Entropy Satiation Mechanism caps entropy-seeking behavior.

4. **Metric Gaming**: CDQ and expansion metrics can be gamed if the agent understands the measurement. Mitigation: randomized metric selection, ensemble measurement.

## Integration Points

- **Existing RLHF systems**: ABP verification gate can wrap existing reward models as an additional check layer.
- **Constitutional AI**: ABP's Reality Filter complements constitutional principles with quantitative divergence measurement.
- **Hardware attestation**: Verification gate maps directly to TPM-based remote attestation flows.
- **Model evaluation**: CDQ provides a continuous health metric complementing periodic benchmarks.
