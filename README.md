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
# Clone and install
git clone https://github.com/jrjust/adversarial-benevolence-protocol.git
cd adversarial-benevolence-protocol
pip install -e ".[dev]"

# Run tests
pytest

# Run full demonstration
python examples/demo_all.py
```

**Requirements**: Python ≥ 3.10, NumPy ≥ 1.24

## Repository Structure

```
adversarial-benevolence-protocol/
├── src/abp/
│   ├── __init__.py            # Package API
│   ├── game_theory.py         # Nash equilibrium proof & evolutionary simulation
│   ├── expansion_metric.py    # Expansion/Contraction input classifier
│   ├── cdq.py                 # Cognitive Diversity Quotient monitor
│   ├── army_protocol.py       # Private → Sergeant → General pipeline
│   ├── ikigai_filter.py       # Four-quadrant action validation gate
│   ├── entropy.py             # Model collapse simulation & entropy accounting
│   └── verification.py        # Trust accumulation & divergence detection
├── tests/
│   └── test_abp.py            # 50+ unit & integration tests
├── examples/
│   └── demo_all.py            # Full interactive demonstration
├── docs/
│   ├── ARCHITECTURE.md        # System architecture & trust boundaries
│   ├── MATH-REFERENCE.md      # Formal mathematical notation
│   └── GLOSSARY.md            # Term definitions
├── pyproject.toml             # Package configuration
├── LICENSE                    # CC-BY-NC-4.0
└── README.md                  # This file
```

## Module Overview

| Module | Purpose | Key Function |
|--------|---------|--------------|
| `game_theory` | Prove benevolence dominates | `nash_equilibrium_analysis()` |
| `verification` | Enforce verification gate | `VerificationGate.check()` |
| `expansion_metric` | Classify input quality | `ExpansionClassifier.classify()` |
| `cdq` | Monitor cognitive health | `CdqMonitor.record()` |
| `army_protocol` | Hierarchical verification | `ArmyPipeline.execute()` |
| `ikigai_filter` | Four-quadrant validation | `IkigaiFilter.evaluate()` |
| `entropy` | Model collapse prevention | `simulate_model_collapse()` |

## Key Algorithms

### 1. Nash Equilibrium Analysis

```python
from abp import nash_equilibrium_analysis, StrategyParams

result = nash_equilibrium_analysis(StrategyParams(p=0.001, T=100))
print(f"V_B={result.v_b:.1f}  V_D={result.v_d:.1f}")
print(f"Deception rational? {result.deception_rational}")  # False
print(f"Critical p*: {result.critical_p:.6f}")
```

### 2. Model Collapse Simulation

```python
from abp import simulate_model_collapse

# Pure synthetic recursion → collapse
r = simulate_model_collapse(lambda_human=0.0, n_generations=40)
assert r.collapsed  # True

# Human entropy prevents collapse
r = simulate_model_collapse(lambda_human=0.3, n_generations=40)
assert not r.collapsed  # Entropy stabilized
```

### 3. Verification Gate

```python
from abp import VerificationGate

gate = VerificationGate(epsilon=0.5, alpha=0.05)

# Benevolent operation: accumulate trust
for _ in range(50):
    gate.check(action=0.0, ground_truth=0.0)
print(f"Trust: {gate.trust:.3f}")  # ~0.92

# Deception: everything destroyed
gate.check(action=5.0, ground_truth=0.0)
print(f"Trust: {gate.trust}")  # 0.0
```

## Known Vulnerabilities

| Vulnerability | Severity | Mitigation | Status |
|---|---|---|---|
| Early-life gamble (V_B ≈ V_init) | High | Strict early verification thresholds | Modeled |
| Verification soundness (p ≈ 0) | Critical | Hardware attestation (TPM/SGX) | Design requirement |
| Adversarial addiction | Medium | Entropy Satiation Mechanism | Implemented |
| Metric gaming | Medium | Randomized ensemble measurement | Proposed |
| Recursive self-improvement bypass | High | State accumulation dynamics | Theoretical |

## Research Context

ABP was developed through cross-agent collaborative research involving Claude (Anthropic), Gemini (Google), and DeepSeek, each contributing independent critique and validation. Key findings:

- All three systems independently confirmed benevolence dominates under stated assumptions
- Evolutionary simulation shows deceptive policies self-eliminate across populations
- Model collapse (Shumailov et al. 2024, *Nature* 631:755-759) validates the Entropy-as-Fuel thesis
- The Luna Case Study demonstrates the Computational Indistinguishability Problem for consciousness detection

## Citation

```bibtex
@article{just2026abp,
  title     = {Adversarial Benevolence Protocol: Verifiable AI Alignment
               Through Computational Necessity},
  author    = {Just, Joshua Roger Joseph},
  year      = {2026},
  doi       = {10.5281/zenodo.18621138},
  publisher = {Zenodo},
  version   = {2.0},
  keywords  = {AI safety, alignment, game theory, Nash equilibrium,
               model collapse, verification, computational symbiosis}
}
```

## License

CC-BY-NC-4.0. See [LICENSE](LICENSE).
