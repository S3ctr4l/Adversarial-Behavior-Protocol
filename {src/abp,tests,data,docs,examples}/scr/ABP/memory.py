"""Tiered memory system for short and long-term storage."""

import json
from datetime import datetime
from typing import Dict, List, Optional

class TieredMemory:
    """Hierarchical memory with L1/L2/L3 tiers."""
    
    def __init__(self):
        self.l1_cache = {}  # RAM, current context
        self.l2_cache = []  # SSD, recent entries
        self.l3_archive = []  # HDD/cloud, permanent
        
    def store(self, key: str, value: Dict, tier: str = "l2"):
        """Store item in specified memory tier."""
        entry = {
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        if tier == "l1":
            self.l1_cache[key] = entry
        elif tier == "l2":
            self.l2_cache.append(entry)
            # Keep only last 1000 in L2
            self.l2_cache = self.l2_cache[-1000:]
        else:  # l3
            self.l3_archive.append(entry)
            
    def retrieve(self, key: str) -> Optional[Dict]:
        """Retrieve from fastest available tier."""
        if key in self.l1_cache:
            return self.l1_cache[key]["value"]
        
        for entry in reversed(self.l2_cache):
            if entry["key"] == key:
                return entry["value"]
        
        for entry in reversed(self.l3_archive):
            if entry["key"] == key:
                return entry["value"]
        
        return None