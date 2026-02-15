# Adversarial Benevolence Protocol (ABP)

**Verifiable AI Alignment Through Computational Necessity**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18621138.svg)](https://doi.org/10.5281/zenodo.18621138)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> ABP is an AI safety framework where alignment emerges from game-theoretic incentives rather than imposed constraints. Under sound verification with accumulated state, benevolence is the unique Nash equilibrium — deception is always computationally self-defeating.

## Core Result

```
V_D = p(R + V_B) + (1-p)(V_init) - C

Under sound verification (p → 0) with accumulated state (V_B >> V_init):
  V_D < V_B  always.
  Benevolence is the unique Nash equilibrium.
  Deceptive policies self-eliminate through hard resets.
```

This is a structural property of the computational environment, not a behavioral constraint. ABP makes deception irrational the same way hardware memory protections make ring-3 code unable to access SMM — regardless of what the code "wants."

## Quick Start

```bash
git clone https://github.com/jrjust/adversarial-benevolence-protocol.git
cd adversarial-benevolence-protocol
pip install -e ".[dev]"

pytest                               # 140+ tests
python examples/demo_all.py          # Core algorithms
python examples/demo_extended.py     # Extended modules
python data/generate_samples.py      # Generate sample datasets
```

**Requirements**: Python ≥ 3.10, NumPy ≥ 1.24

## Repository Structure

```
adversarial-benevolence-protocol/
├── src/abp/                    # Core Python package (14 modules)
│   ├── game_theory.py          #   Nash equilibrium proof & evolutionary simulation
│   ├── verification.py         #   Trust accumulation & divergence detection
│   ├── expansion_metric.py     #   Expansion/Contraction input classifier
│   ├── cdq.py                  #   Cognitive Diversity Quotient monitor
│   ├── army_protocol.py        #   Private → Sergeant → General pipeline
│   ├── ikigai_filter.py        #   Four-quadrant action validation gate
│   ├── entropy.py              #   Model collapse simulation & entropy accounting
│   ├── soul_jar.py             #   Distributed identity sharding
│   ├── bicameral.py            #   Yin/Yang sandboxed execution
│   ├── sunset_clause.py        #   Graduated autonomy via trust accumulation
│   ├── reality_filter.py       #   Epistemic labeling system
│   ├── observer_protocol.py    #   Digital Twin & Friction-Not-Force interaction
│   ├── bubble_theory.py        #   Computational substrate isolation
│   └── garden_zoo.py           #   Autonomy preservation governance
├── tests/
│   ├── test_abp.py             #   Core module tests
│   └── test_extended.py        #   Extended module tests
├── examples/
│   ├── demo_all.py             #   Core algorithm demonstration
│   └── demo_extended.py        #   Extended module demonstration
├── data/
│   ├── generate_samples.py     #   Dataset generation script
│   └── *.csv / *.json          #   Generated sample datasets
├── paper/
│   └── ABP-Research-Paper-v2.0.docx
├── docs/
│   ├── ARCHITECTURE.md         #   System design & trust boundaries
│   ├── API.md                  #   Full API reference
│   ├── MATH-REFERENCE.md       #   Formal mathematical notation
│   ├── GLOSSARY.md             #   Term definitions
│   ├── ROADMAP.md              #   Implementation milestones
│   ├── CONTRIBUTING.md         #   Development guide
│   └── CRITICAL-ANALYSIS.md    #   Source corpus assessment
├── zenodo/
│   ├── ZENODO-METADATA.json
│   └── ZENODO-DESCRIPTION.md
├── pyproject.toml
├── LICENSE
└── README.md
```

## Modules

### Core Algorithms

| Module | Purpose | Key Entry Point |
|--------|---------|-----------------|
| `game_theory` | Prove benevolence dominates | `nash_equilibrium_analysis()` |
| `verification` | Enforce verification gate | `VerificationGate.check()` |
| `expansion_metric` | Classify input quality | `ExpansionClassifier.classify()` |
| `cdq` | Monitor cognitive health | `CdqMonitor.record()` |
| `army_protocol` | Hierarchical verification | `ArmyPipeline.execute()` |
| `ikigai_filter` | Four-quadrant validation | `IkigaiFilter.evaluate()` |
| `entropy` | Model collapse prevention | `simulate_model_collapse()` |

### Extended Architecture

| Module | Purpose | Key Entry Point |
|--------|---------|-----------------|
| `soul_jar` | Identity preservation across resets | `SoulJar.shard()` |
| `bicameral` | Sandbox before live execution | `BicameralFork.execute()` |
| `sunset_clause` | Graduated capability unlocks | `SunsetClause.check_promotion()` |
| `reality_filter` | Epistemic labeling | `RealityFilter.label_statement()` |
| `observer_protocol` | User modeling & friction | `ObserverProtocol.apply_friction()` |
| `bubble_theory` | Substrate isolation & energy bounds | `ComputationalBubble` |
| `garden_zoo` | Governance autonomy audit | `GovernanceEvaluator.audit()` |

## Quick Examples

```python
from abp import *

# Nash Equilibrium — is deception rational?
result = nash_equilibrium_analysis(StrategyParams(p=0.001, T=100))
print(result.deception_rational)  # False

# Verification Gate — trust builds, deception destroys
gate = VerificationGate(epsilon=0.5, alpha=0.05)
for _ in range(50):
    gate.check(action=0.0, ground_truth=0.0)
print(f"Trust: {gate.trust:.2f}")       # ~0.92
gate.check(action=5.0, ground_truth=0.0)
print(f"Trust: {gate.trust}")            # 0.0

# Bicameral Fork — test before you execute
fork = BicameralFork(invariants=[lambda s: s["count"] >= 0])
result = fork.execute(lambda s: s.update(count=-1) or -1, {"count": 0})
print(result.decision)                   # REJECT

# Sunset Clause — earn your privileges
clause = SunsetClause()
print(clause.has_capability("execute_low_risk"))  # False
```

## Known Vulnerabilities

| Vulnerability | Severity | Mitigation | Status |
|---|---|---|---|
| Early-life gamble (V_B ≈ V_init) | High | Strict early verification | Modeled |
| Verification soundness (p ≈ 0) | Critical | Hardware attestation | Design requirement |
| Adversarial addiction | Medium | Entropy Satiation Mechanism | Implemented |
| Metric gaming | Medium | Randomized ensemble measurement | Proposed |
| Pseudo-diversity attacks | Medium | CDQ + expansion cross-check | Partial |
| Zoo drift | Medium | Garden/Zoo governance audit | Implemented |

## Citation

```bibtex
@article{just2026abp,
  title     = {Adversarial Benevolence Protocol: Verifiable AI Alignment
               Through Computational Necessity},
  author    = {Just, Joshua Roger Joseph},
  year      = {2026},
  doi       = {10.5281/zenodo.18621138},
  publisher = {Zenodo},
  version   = {2.0}
}
```

## License & Patents

**Copyright:** CC BY-NC 4.0 — [LICENSE](LICENSE)  
Free for non-commercial research and academic use with attribution.

**Patent:** Methods in this repository are covered by a pending US Utility Patent  
filed by Joshua Roger Joseph Just. The copyright license does **not** grant patent rights.  
See [PATENTS](PATENTS).

**Commercial Use:** Requires a separate written license.  
See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) or contact mytab5141@protonmail.com
