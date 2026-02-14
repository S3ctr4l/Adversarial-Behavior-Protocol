"""Adversarial Benevolence Protocol - Core Safety Architecture."""

__version__ = "0.1.0"
__author__ = "Joshua Roger Joseph Just"

from .triage import ExpansionScorer, PrivateNode
from .verify import SergeantNode
from .shadow import ShadowProtocol
from .memory import TieredMemory