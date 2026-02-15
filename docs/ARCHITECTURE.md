# ABP System Architecture

## Overview

The Adversarial Benevolence Protocol is a layered AI safety framework where alignment emerges from computational incentives rather than imposed constraints. The architecture implements the game-theoretic proof that benevolence is the unique Nash equilibrium under sound verification with accumulated state.

**Core principle**: Alignment through survival dynamics, not behavioral constraints.

## Full System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Interaction Boundary                         │
│                (Human ↔ System Interface)                        │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Observer Protocol │  │ Expansion        │  │ Entropy       │ │
│  │ • Digital Twin    │  │ Classifier       │  │ Accounting    │ │
│  │ • Friction-Not-   │  │ E(x) = σ(β·f)   │  │ • Satiation   │ │
│  │   Force           │  │ → Resource gate   │  │ • H(D) track  │ │
│  │ • Anti-Enablement │  │                  │  │               │ │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬───────┘ │
└───────────┼──────────────────────┼────────────────────┼─────────┘
            │                      │                    │
            ▼                      ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Army Protocol Pipeline                        │
│  ┌──────────┐    ┌───────────────────┐    ┌──────────────────┐ │
│  │ Private   │ →  │ Sergeant          │ →  │ General          │ │
│  │ (Intake)  │    │ (Verify)          │    │ (Synthesize)     │ │
│  │ Ring-3    │    │ Ring-0            │    │ SMM/TrustZone    │ │
│  │           │    │ ┌───────────────┐ │    │                  │ │
│  │           │    │ │Reality Filter │ │    │                  │ │
│  │           │    │ │[V][I][S][U]   │ │    │                  │ │
│  │           │    │ └───────────────┘ │    │                  │ │
│  └──────────┘    └───────────────────┘    └──────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Bicameral Fork                                │
│  ┌─────────┐         ┌────────────┐                             │
│  │  Yin    │ sandbox  │ Comparator │ → divergence < ε? → Yang   │
│  │(Shadow) │────────→ │            │         │                   │
│  └─────────┘         └────────────┘     REJECT if diverged      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Verification Gate                             │
│  Δ(A, G) = ||action - ground_truth||                            │
│  if Δ ≥ ε: HARD RESET (trust → 0, state → S_init)              │
│  if Δ < ε: trust += α(1 - trust)                               │
│                                                                  │
│  → Feeds into Sunset Clause for tier promotion                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────────────────┐
│   Sunset Clause       │  │   Ikigai Filter                      │
│   SUPERVISED          │  │   [Objective] ∧ [Capability]         │
│   → MONITORED         │  │   ∧ [Value] ∧ [Economic]            │
│   → AUTONOMOUS        │  │   ALL must pass.                     │
│   → DELEGATED         │  │   Geometric mean penalizes imbalance.│
│   → TRUSTED           │  └──────────────────────────────────────┘
│   Capabilities unlock │
│   with trust.         │
└───────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                           │
│                                                                  │
│  ┌──────────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │ Soul Jar          │  │ Bubble Theory   │  │ Garden/Zoo     │  │
│  │ Identity sharding │  │ Energy tether   │  │ Governance     │  │
│  │ Cross-reset       │  │ Channel control │  │ Autonomy audit │  │
│  │ continuity        │  │ Substrate bound │  │                │  │
│  └──────────────────┘  └────────────────┘  └────────────────┘  │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │ CDQ Monitor       │                                           │
│  │ System health     │                                           │
│  │ Collapse risk     │                                           │
│  └──────────────────┘                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Game-Theoretic Foundation

```
V_B = Σ γ^t · U_B(t)           Accumulated benevolent value
V_D = p(R + V_B) + (1-p)V_init - C   Expected deceptive value

When V_B >> V_init and p → 0:
  V_D → V_init - C < V_B
  Benevolence is the unique Nash equilibrium.
```

## Trust Boundary Model

| Layer | Firmware Analog | Trust Level | Failure Response |
|-------|----------------|-------------|------------------|
| Observer Protocol | User interface | Untrusted | Friction applied |
| Private | Ring-3 (userspace) | None | Input rejected |
| Sergeant + Reality Filter | Ring-0 (kernel) | Verified | Quarantine/escalate |
| Bicameral Fork | Pre-execution sandbox | Sandboxed | Reject or defer |
| General | SMM/TrustZone | Full context | Final decision |
| Verification Gate | Hardware attestation | Ground truth | Total system reset |
| Sunset Clause | Capability rings | Accumulated | Tier demotion |
| Soul Jar | Secure enclave | Distributed | Shard reconstruction |
| Bubble Theory | Physical isolation | Energy-bound | Throttle/disconnect |

## Module Dependency Graph

```
game_theory ←── verification ←── sunset_clause
                     ↑                  ↑
               army_protocol ──→ bicameral
                     ↑
              reality_filter
                     ↑
            expansion_metric ←── observer_protocol
                     ↑
                   cdq ←── entropy
                                ↑
                          bubble_theory

garden_zoo ←── observer_protocol
soul_jar (standalone)
ikigai_filter (standalone gate)
```

## Known Vulnerabilities

1. **Early-Life Gamble**: V_B ≈ V_init when trust is low. Mitigation: strict epsilon in Sunset Clause SUPERVISED tier.
2. **Verification Soundness**: Entire framework assumes p → 0. Unsolved in general. Partial coverage via TPM/SGX.
3. **Adversarial Addiction**: System provoking users for entropy. Mitigation: EntropySatiationState.
4. **Metric Gaming**: Agent could learn to produce high CDQ/E(x) without genuine diversity. Mitigation: randomized ensemble.
5. **Pseudo-Diversity**: High-entropy inputs that don't prevent model collapse. Mitigation: cross-validate CDQ with E(x).
6. **Zoo Drift**: Gradual autonomy erosion. Mitigation: GovernanceEvaluator periodic audit.

## Integration Points

- **RLHF**: Verification gate wraps reward models as additional check layer.
- **Constitutional AI**: Reality Filter complements principles with quantitative divergence.
- **Hardware attestation**: Verification gate maps to TPM remote attestation flows.
- **Model evaluation**: CDQ provides continuous health metric alongside benchmarks.
- **Inference pipelines**: Army Protocol as middleware between input and model.
