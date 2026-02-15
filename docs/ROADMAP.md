# Implementation Roadmap

## Phase 0 — Reference Implementation (Current)
**Status: Complete**

Minimal Python implementation demonstrating all core ABP algorithms with tests.

| Milestone | Deliverable | Status |
|-----------|-------------|--------|
| Core game theory proof | `game_theory.py` — Nash equilibrium, evolutionary simulation | ✅ |
| Verification gate | `verification.py` — trust accumulation, hard reset | ✅ |
| Input classification | `expansion_metric.py` — E(x) sigmoid classifier | ✅ |
| System health monitor | `cdq.py` — Cognitive Diversity Quotient | ✅ |
| Hierarchical pipeline | `army_protocol.py` — Private/Sergeant/General | ✅ |
| Action validation | `ikigai_filter.py` — four-quadrant gate | ✅ |
| Entropy modeling | `entropy.py` — collapse simulation, satiation | ✅ |
| Identity preservation | `soul_jar.py` — distributed sharding | ✅ |
| Sandboxed execution | `bicameral.py` — Yin/Yang fork | ✅ |
| Graduated autonomy | `sunset_clause.py` — tier promotion/demotion | ✅ |
| Epistemic labeling | `reality_filter.py` — statement classification | ✅ |
| User modeling | `observer_protocol.py` — Digital Twin, friction | ✅ |
| Substrate isolation | `bubble_theory.py` — energy tethering | ✅ |
| Governance evaluation | `garden_zoo.py` — autonomy audit | ✅ |
| Test suite | 140+ unit, boundary, and integration tests | ✅ |
| Research paper | 50+ page academic paper (v2.0) | ✅ |

## Phase 1 — Hardened Primitives
**Target: Q3 2026**

Replace reference implementations with production-grade cryptographic and mathematical primitives.

| Milestone | Description | Dependencies |
|-----------|-------------|--------------|
| Shamir's Secret Sharing | True k-of-n threshold for Soul Jar | `galois` or custom GF(256) |
| Hardware entropy source | TPM/RDRAND integration for shard seeds | `tpm2-tools`, Linux kernel |
| Calibrated metrics | Train E(x) weights on labeled interaction data | Labeled dataset |
| Formal verification | Prove Nash equilibrium in Lean 4 or Coq | Lean 4 toolchain |
| Async Bicameral | Parallel Yin execution with timeout enforcement | `asyncio` |

## Phase 2 — System Integration
**Target: Q1 2027**

Integrate ABP modules into a running AI system for live validation.

| Milestone | Description | Dependencies |
|-----------|-------------|--------------|
| RLHF wrapper | Verification gate wrapping existing reward models | Phase 1 |
| LLM pipeline integration | Army Protocol as middleware in inference pipeline | API access |
| CDQ dashboard | Real-time monitoring with alerting | Grafana/Prometheus |
| A/B testing framework | Compare ABP-gated vs ungated outputs | Evaluation harness |
| Energy metering | RAPL/IPMI integration for Bubble Theory | Hardware access |

## Phase 3 — Hardware Attestation
**Target: Q3 2027**

Move verification gate to hardware trust boundary.

| Milestone | Description | Dependencies |
|-----------|-------------|--------------|
| TPM-backed verification | Remote attestation for verification gate state | TPM 2.0 |
| SGX enclave for Soul Jar | Shard reconstruction in trusted execution environment | Intel SGX |
| Hardware kill switch | Physical reset mechanism for verification failures | Custom PCB |
| Side-channel hardening | Constant-time operations for all crypto paths | Phase 1 crypto |

## Phase 4 — Multi-Agent
**Target: 2028**

Extend ABP across multi-agent systems.

| Milestone | Description | Dependencies |
|-----------|-------------|--------------|
| Cross-agent verification | Agents verify each other's actions | Phase 2 |
| Trust delegation chains | Sunset Clause across agent hierarchies | Phase 2 |
| Federated Soul Jar | Identity shards across organizational boundaries | Phase 1 |
| Swarm CDQ | Aggregate cognitive health across agent populations | Phase 2 |
| Governance meta-audit | Garden/Zoo evaluation of multi-agent policies | Phase 2 |

## Success Criteria

Each phase requires:
1. All existing tests pass (no regression)
2. New components have ≥90% test coverage
3. Performance benchmarks within 2x of baseline
4. Security audit of cryptographic components
5. Documentation updated for all new interfaces
