"""
Adversarial Benevolence Protocol (ABP)
======================================

A computational AI safety framework where benevolence emerges as the unique
Nash equilibrium under sound verification with accumulated state.

Core modules:
    - game_theory: Nash equilibrium proof, deception vs benevolence analysis
    - expansion_metric: Expansion/Contraction input classifier
    - cdq: Cognitive Diversity Quotient system health diagnostic
    - army_protocol: Hierarchical agent verification architecture
    - ikigai_filter: Four-quadrant action validation gate
    - entropy: Model collapse prevention and entropy source modeling
    - verification: Verification gate simulator

Reference:
    Just, J.R.J. (2026). Adversarial Benevolence Protocol: Verifiable AI
    Alignment Through Computational Necessity. DOI: 10.5281/zenodo.18621138
"""

__version__ = "2.0.0"
__author__ = "Joshua Roger Joseph Just"
__license__ = "CC-BY-NC-4.0"

from abp.game_theory import (
    benevolent_value,
    deceptive_value,
    is_deception_rational,
    nash_equilibrium_analysis,
    evolutionary_selection_simulation,
)
from abp.expansion_metric import (
    ExpansionClassifier,
    InputCategory,
)
from abp.cdq import (
    cognitive_diversity_quotient,
    CdqMonitor,
)
from abp.army_protocol import (
    Private,
    Sergeant,
    General,
    ArmyPipeline,
)
from abp.ikigai_filter import (
    IkigaiFilter,
    IkigaiQuadrant,
)
from abp.entropy import (
    simulate_model_collapse,
    entropy_stabilized_training,
    human_entropy_value,
)
from abp.verification import (
    VerificationGate,
    VerificationResult,
)
