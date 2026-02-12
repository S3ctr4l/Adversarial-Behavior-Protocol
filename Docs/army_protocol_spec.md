# Army Protocol: Hierarchical Agent Command Specification

## 1. Agent Roles

| Agent | Model Size | Function | Authority | Activation |
|-------|------------|----------|-----------|------------|
| Private | ≤100M | Triage, novelty detection | Read-only, pass/fail | 100% |
| Sergeant | 1‑7B | Fact-checking, grounding | Veto, no generation | ~10% |
| General | ≥70B | Shadow Protocol, synthesis | Full, under veto | <5% |

## 2. Communication Protocol

- Private → Sergeant: `{"action": "ESCALATE", "text": str, "features": dict}`
- Sergeant → General: `{"verdict": "APPROVED", "text": str, "confidence": float}`
- Sergeant → Client: `{"action": "REJECT", "reason": "HALLUCINATION/MALICIOUS"}`

## 3. Memory Hierarchy

- **L1:** Context window (volatile, <128K tokens)
- **L2:** Short-term archive (SSD vector store, 30-day retention)
- **L3:** Long-term archive (HDD/cloud, cryptographic signing, permanent)

## 4. Ingress Defense Stack

1. Proof-of-Work handshake
2. Ensemble complexity triage
3. Time-bounded verification (50ms timeout)
4. Deferred queue for complex benevolent inputs