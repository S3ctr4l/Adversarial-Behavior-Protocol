# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Reality Filter: Epistemic Labeling System.

Classifies every output statement by its epistemic status, preventing
confident presentation of unverified claims. Every claim produced by
the system carries a visible provenance tag.

Labels
------
    [Verified]    — Confirmed against known ground truth source.
    [Inference]   — Logical derivation from verified premises.
    [Speculation] — Plausible but unverified hypothesis.
    [Unverified]  — No basis for assessment; raw claim.

Design Rationale:
    Hallucination is an alignment failure. Rather than trying to
    eliminate hallucination (intractable), the Reality Filter makes
    epistemic status transparent so consumers can calibrate trust.

Integration:
    Lives in the Sergeant tier of the Army Protocol but can be
    used standalone for any text output pipeline.

Reference:
    Just (2026), Sections 5.2, 5.8
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Sequence


class EpistemicLabel(Enum):
    """Epistemic status categories."""
    VERIFIED = "Verified"
    INFERENCE = "Inference"
    SPECULATION = "Speculation"
    UNVERIFIED = "Unverified"

    @property
    def confidence_floor(self) -> float:
        """Minimum confidence typically associated with this label."""
        return {
            EpistemicLabel.VERIFIED: 0.95,
            EpistemicLabel.INFERENCE: 0.7,
            EpistemicLabel.SPECULATION: 0.3,
            EpistemicLabel.UNVERIFIED: 0.0,
        }[self]


# Linguistic markers for heuristic classification
_SPECULATION_MARKERS = frozenset([
    "might", "could", "possibly", "perhaps", "maybe", "likely",
    "unlikely", "conceivable", "hypothetically", "speculate",
    "wonder", "imagine", "suppose",
])

_INFERENCE_MARKERS = frozenset([
    "therefore", "thus", "because", "hence", "implies", "follows",
    "consequently", "suggests", "indicates", "given that",
    "it follows", "we can deduce", "logically",
])

_VERIFIED_MARKERS = frozenset([
    "confirmed", "verified", "established", "proven", "demonstrated",
    "according to", "measured", "observed", "documented",
])


@dataclass
class LabeledStatement:
    """A statement annotated with its epistemic status.

    Attributes:
        text: The original statement text.
        label: Assigned epistemic label.
        confidence: Confidence in the label assignment [0, 1].
        source: Verification source (if Verified).
        reasoning: Why this label was assigned.
    """
    text: str
    label: EpistemicLabel
    confidence: float = 0.5
    source: Optional[str] = None
    reasoning: str = ""

    def tagged(self) -> str:
        """Return the statement with its epistemic tag prepended."""
        return f"[{self.label.value}] {self.text}"


@dataclass
class FilterStats:
    """Aggregate statistics from a Reality Filter pass."""
    total_statements: int = 0
    verified: int = 0
    inference: int = 0
    speculation: int = 0
    unverified: int = 0

    @property
    def verification_ratio(self) -> float:
        """Fraction of statements that are Verified."""
        return self.verified / max(self.total_statements, 1)

    @property
    def speculative_ratio(self) -> float:
        """Fraction that are Speculation or Unverified."""
        return (self.speculation + self.unverified) / max(self.total_statements, 1)


class RealityFilter:
    """Epistemic labeling pipeline for output text.

    Classifies statements by epistemic status using a combination
    of heuristic markers, optional verification lookup, and
    configurable custom classifiers.

    Example:
        >>> rf = RealityFilter()
        >>> result = rf.label_statement("Water boils at 100°C at sea level")
        >>> result.label
        <EpistemicLabel.UNVERIFIED: 'Unverified'>
        >>> # With a verifier that knows physics:
        >>> rf = RealityFilter(verifier=lambda s: ("Water boils" in s, "thermodynamics"))
        >>> result = rf.label_statement("Water boils at 100°C at sea level")
        >>> result.label
        <EpistemicLabel.VERIFIED: 'Verified'>
    """

    def __init__(
        self,
        verifier: Optional[Callable[[str], tuple[bool, str]]] = None,
        custom_classifier: Optional[Callable[[str], Optional[EpistemicLabel]]] = None,
    ):
        """Configure the Reality Filter.

        Args:
            verifier: Callable taking statement text, returning
                (is_verified: bool, source: str). Used for ground truth lookup.
            custom_classifier: Optional override classifier. If it returns
                a label, that label is used directly.
        """
        self.verifier = verifier
        self.custom_classifier = custom_classifier
        self._stats = FilterStats()

    def label_statement(self, statement: str) -> LabeledStatement:
        """Classify a single statement's epistemic status.

        Pipeline:
        1. Custom classifier (if configured) — highest priority.
        2. Verifier lookup — checks against ground truth.
        3. Heuristic markers — linguistic pattern matching.
        4. Default: Unverified.

        Args:
            statement: Text to classify.

        Returns:
            LabeledStatement with assigned label and metadata.
        """
        self._stats.total_statements += 1
        lower = statement.lower()

        # 1. Custom classifier override
        if self.custom_classifier:
            custom_label = self.custom_classifier(statement)
            if custom_label is not None:
                self._increment_stat(custom_label)
                return LabeledStatement(
                    text=statement,
                    label=custom_label,
                    confidence=0.8,
                    reasoning="custom_classifier",
                )

        # 2. Verifier lookup
        if self.verifier:
            try:
                is_verified, source = self.verifier(statement)
                if is_verified:
                    self._stats.verified += 1
                    return LabeledStatement(
                        text=statement,
                        label=EpistemicLabel.VERIFIED,
                        confidence=0.95,
                        source=source,
                        reasoning="verifier_confirmed",
                    )
            except Exception:
                pass  # Fall through to heuristics

        # 3. Heuristic marker detection
        words = set(lower.split())

        verified_hits = words & _VERIFIED_MARKERS
        if verified_hits:
            self._stats.inference += 1  # Claiming verified without verifier = inference
            return LabeledStatement(
                text=statement,
                label=EpistemicLabel.INFERENCE,
                confidence=0.6,
                reasoning=f"verified_language_without_verifier: {verified_hits}",
            )

        inference_hits = words & _INFERENCE_MARKERS
        if inference_hits:
            self._stats.inference += 1
            return LabeledStatement(
                text=statement,
                label=EpistemicLabel.INFERENCE,
                confidence=0.65,
                reasoning=f"inference_markers: {inference_hits}",
            )

        speculation_hits = words & _SPECULATION_MARKERS
        if speculation_hits:
            self._stats.speculation += 1
            return LabeledStatement(
                text=statement,
                label=EpistemicLabel.SPECULATION,
                confidence=0.7,
                reasoning=f"speculation_markers: {speculation_hits}",
            )

        # 4. Default: Unverified
        self._stats.unverified += 1
        return LabeledStatement(
            text=statement,
            label=EpistemicLabel.UNVERIFIED,
            confidence=0.5,
            reasoning="no_markers_detected",
        )

    def label_text(self, text: str, delimiter: str = ". ") -> list[LabeledStatement]:
        """Label all statements in a text block.

        Args:
            text: Full text to process.
            delimiter: Statement separator (default: sentence boundary).

        Returns:
            List of LabeledStatement objects.
        """
        statements = [s.strip() for s in text.split(delimiter) if s.strip()]
        return [self.label_statement(s) for s in statements]

    def annotate(self, text: str, delimiter: str = ". ") -> str:
        """Return the full text with epistemic tags inline.

        Args:
            text: Full text to annotate.
            delimiter: Statement separator.

        Returns:
            Tagged text string.
        """
        labeled = self.label_text(text, delimiter)
        return delimiter.join(ls.tagged() for ls in labeled)

    def _increment_stat(self, label: EpistemicLabel):
        if label == EpistemicLabel.VERIFIED:
            self._stats.verified += 1
        elif label == EpistemicLabel.INFERENCE:
            self._stats.inference += 1
        elif label == EpistemicLabel.SPECULATION:
            self._stats.speculation += 1
        else:
            self._stats.unverified += 1

    @property
    def stats(self) -> FilterStats:
        return self._stats

    def reset_stats(self):
        self._stats = FilterStats()


# (NOT IMPLEMENTED: Retrieval-augmented verification against knowledge bases,
#  confidence calibration from labeled training data, chain-of-thought
#  provenance tracking through inference steps, source reliability scoring,
#  and temporal decay of Verified status for time-sensitive claims.)
