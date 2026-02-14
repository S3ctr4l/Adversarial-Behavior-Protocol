"""Expansion metric computation and Private node logic."""

import hashlib
import numpy as np
from typing import Dict, Union, Optional

class ExpansionScorer:
    """Computes the expansion score for a given input text."""
    
    def __init__(self):
        # In production, load actual models
        self.recent_hashes = []
        
    def compute_dispersion(self, text: str, k: int = 5) -> float:
        """Placeholder: embedding dispersion."""
        # Real implementation would use sentence-transformers
        return float(np.random.uniform(0.3, 0.9))
    
    def estimate_reasoning_depth(self, text: str) -> int:
        """Heuristic: longer sentences = deeper reasoning."""
        words = text.split()
        if len(words) < 10:
            return 1
        elif len(words) < 30:
            return 2
        else:
            return 3
    
    def compute_novelty(self, text: str) -> float:
        """Placeholder: semantic hash collision rate."""
        return 1.0  # Assume novel
    
    def compute(self, text: str) -> float:
        """Return expansion score in [0,1]."""
        d = self.compute_dispersion(text)
        r = self.estimate_reasoning_depth(text)
        n = self.compute_novelty(text)
        # Simple linear combination
        score = 0.4 * d + 0.4 * (r / 5.0) + 0.2 * n
        return float(np.clip(score, 0, 1))


class PrivateNode:
    """Triage agent – fastest path for most queries."""
    
    def __init__(self, scorer: Optional[ExpansionScorer] = None):
        self.scorer = scorer or ExpansionScorer()
        
    def verify_pow(self, text: str, nonce: int, difficulty: int = 4) -> bool:
        """Verify Proof-of-Work: SHA256(text+nonce) starts with N zeros."""
        target = "0" * difficulty
        hash_hex = hashlib.sha256((text + str(nonce)).encode()).hexdigest()
        return hash_hex.startswith(target)
    
    def process(self, text: str, pow_nonce: Optional[int] = None) -> Dict:
        """Process input through triage layer."""
        # 1. PoW check
        if pow_nonce is not None:
            if not self.verify_pow(text, pow_nonce):
                return {"action": "REJECT", "reason": "Invalid PoW"}
        
        # 2. Compute expansion score
        score = self.scorer.compute(text)
        
        # 3. Triage decision
        if score < 0.3:
            return {"action": "PASS", "confidence": score}
        else:
            return {"action": "ESCALATE", "score": score, "text": text}