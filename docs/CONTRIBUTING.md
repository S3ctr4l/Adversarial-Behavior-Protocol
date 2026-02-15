# Contributing to ABP

## Getting Started

```bash
git clone https://github.com/jrjust/adversarial-benevolence-protocol.git
cd adversarial-benevolence-protocol
pip install -e ".[dev]"
pytest
```

## Development Workflow

1. **Branch** from `main` with descriptive name (`feature/shamir-sss`, `fix/cdq-edge-case`)
2. **Write tests first** — every module has a corresponding test file
3. **Implement** with full docstrings and type hints
4. **Run full suite** — `pytest -v --tb=short`
5. **PR** with description of what changed and why

## Code Standards

- Python 3.10+ with type annotations on all public APIs
- NumPy-style docstrings with Args/Returns/Example sections
- Every public class and function needs a docstring
- Every module ends with a `(NOT IMPLEMENTED: ...)` block listing known gaps
- No external dependencies beyond NumPy for core modules
- `pytest` for testing; aim for ≥90% coverage on new code

## Module Structure

Each ABP module follows this pattern:

```python
"""
Module Name: One-Line Description.

Extended description of the concept and its role in ABP.

Mathematical Model
------------------
    Formal equations with variable definitions.

Reference:
    Just (2026), Section X.Y
"""

# Dataclasses for inputs/outputs
# Core implementation class
# Utility functions
# (NOT IMPLEMENTED: ...) block at end
```

## Testing Conventions

- Test file mirrors module: `src/abp/foo.py` → `tests/test_foo.py` or section in `test_abp.py`
- Test class per public class: `class TestFoo:`
- Test known failure modes explicitly (broken verification, early-life gamble, etc.)
- Integration tests go in `TestExtendedIntegration` or similar

## Security-Sensitive Code

Cryptographic and verification modules require extra review:

- Constant-time comparisons for all hash/HMAC checks
- No timing side channels in verification gate
- Soul Jar shard operations must not leak seed material
- Document threat model in docstring

## What Needs Work

See `docs/ROADMAP.md` for the full implementation plan. Priority areas:

1. **Shamir's Secret Sharing** for true k-of-n Soul Jar threshold
2. **Formal verification** of Nash equilibrium in Lean 4 / Coq
3. **Calibrated expansion metric** trained on real interaction data
4. **Async Bicameral** with proper timeout enforcement
5. **Hardware integration** for energy tethering and attestation

## License

Contributions are accepted under CC-BY-NC-4.0. By submitting a PR, you agree to license your contribution under the same terms.
