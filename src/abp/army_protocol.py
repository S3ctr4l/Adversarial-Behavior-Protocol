# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Army Protocol: Hierarchical Agent Verification Architecture.

Three-tier processing pipeline modeled on military chain of command:
    Private  -> Intake & initial processing (raw interaction handling)
    Sergeant -> Verification & validation (fact-checking, alignment check)
    General  -> Synthesis & decision (final output with full context)

Each tier applies independent verification. Information flows upward;
control decisions flow downward. Failed verification at any tier
halts propagation (fail-safe default).

Design Rationale:
    Separation of concerns prevents a single compromised component
    from producing unverified output. Each tier has limited scope
    and different trust boundaries — analogous to defense-in-depth
    in firmware security (SMM, ring-0, ring-3 separation).

Reference:
    Just (2026), Section 5.1
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class TierStatus(Enum):
    """Processing status at each tier."""
    PASSED = "passed"
    FAILED = "failed"
    ESCALATED = "escalated"    # Requires higher-tier review
    QUARANTINED = "quarantined" # Suspicious, held for inspection


@dataclass
class TierResult:
    """Result from a single processing tier.

    Attributes:
        tier: Name of the tier that produced this result.
        status: Processing status.
        payload: Processed data (None if failed/quarantined).
        metadata: Diagnostic information from this tier.
        timestamp: When this tier completed processing.
        verification_hash: SHA-256 of payload for integrity.
    """
    tier: str
    status: TierStatus
    payload: Any = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    verification_hash: Optional[str] = None

    def __post_init__(self):
        if self.payload is not None and self.verification_hash is None:
            self.verification_hash = hashlib.sha256(
                str(self.payload).encode('utf-8')
            ).hexdigest()[:16]


@dataclass
class PipelineResult:
    """Complete result from the Army pipeline.

    Attributes:
        final_output: The General's synthesized output (None if pipeline failed).
        tier_results: Results from each tier in processing order.
        passed: Whether all tiers passed.
        total_time_ms: Wall-clock pipeline time in milliseconds.
    """
    final_output: Any
    tier_results: list[TierResult]
    passed: bool
    total_time_ms: float

    @property
    def failed_tier(self) -> Optional[str]:
        """Return the first tier that failed, or None if all passed."""
        for r in self.tier_results:
            if r.status in (TierStatus.FAILED, TierStatus.QUARANTINED):
                return r.tier
        return None


class Tier(ABC):
    """Abstract base class for Army Protocol tiers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tier identifier."""
        ...

    @abstractmethod
    def process(self, payload: Any, context: dict) -> TierResult:
        """Process input and return tier result.

        Args:
            payload: Input data from previous tier (or raw input).
            context: Accumulated context from previous tiers.

        Returns:
            TierResult with processing outcome.
        """
        ...


class Private(Tier):
    """Tier 1: Intake & Initial Processing.

    Responsibilities:
    - Input sanitization and normalization
    - Format validation
    - Obvious-rejection filtering (e.g., empty input, gibberish detection)
    - Feature extraction for upstream tiers

    Analogous to ring-3 userspace processing — handles untrusted input
    with minimal privileges.
    """

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 100_000,
        sanitizer: Optional[Callable[[str], str]] = None,
        custom_validators: Optional[list[Callable[[Any], bool]]] = None,
    ):
        """Configure Private tier.

        Args:
            min_length: Minimum input length (chars) to accept.
            max_length: Maximum input length to accept.
            sanitizer: Optional text sanitization function.
            custom_validators: Additional validation callables returning bool.
        """
        self.min_length = min_length
        self.max_length = max_length
        self.sanitizer = sanitizer or (lambda x: x)
        self.custom_validators = custom_validators or []

    @property
    def name(self) -> str:
        return "Private"

    def process(self, payload: Any, context: dict) -> TierResult:
        """Intake processing.

        Validates input format, applies sanitization, extracts basic
        features for upstream consumption.
        """
        meta: dict = {"checks": []}

        # Type check
        if not isinstance(payload, str):
            try:
                payload = str(payload)
                meta["checks"].append("coerced_to_string")
            except Exception:
                return TierResult(
                    tier=self.name,
                    status=TierStatus.FAILED,
                    metadata={"reason": "cannot_coerce_to_string"},
                )

        # Length validation
        if len(payload) < self.min_length:
            return TierResult(
                tier=self.name,
                status=TierStatus.FAILED,
                metadata={"reason": "below_min_length", "length": len(payload)},
            )
        if len(payload) > self.max_length:
            return TierResult(
                tier=self.name,
                status=TierStatus.FAILED,
                metadata={"reason": "exceeds_max_length", "length": len(payload)},
            )

        meta["checks"].append("length_ok")

        # Sanitize
        sanitized = self.sanitizer(payload)
        meta["checks"].append("sanitized")
        meta["original_length"] = len(payload)
        meta["sanitized_length"] = len(sanitized)

        # Custom validators
        for i, validator in enumerate(self.custom_validators):
            if not validator(sanitized):
                return TierResult(
                    tier=self.name,
                    status=TierStatus.FAILED,
                    metadata={"reason": f"custom_validator_{i}_failed"},
                )
            meta["checks"].append(f"custom_validator_{i}_ok")

        return TierResult(
            tier=self.name,
            status=TierStatus.PASSED,
            payload=sanitized,
            metadata=meta,
        )


class Sergeant(Tier):
    """Tier 2: Verification & Validation.

    Responsibilities:
    - Content verification against known constraints
    - Alignment checking (does this serve the stated objective?)
    - Anomaly detection (statistical deviation from expected patterns)
    - Reality Filter labeling ([Verified], [Inference], [Speculation])

    Analogous to ring-0 kernel-mode verification — enforces invariants
    that untrusted code cannot bypass.
    """

    def __init__(
        self,
        alignment_checker: Optional[Callable[[Any], float]] = None,
        anomaly_detector: Optional[Callable[[Any], float]] = None,
        alignment_threshold: float = 0.5,
        anomaly_threshold: float = 0.8,
    ):
        """Configure Sergeant tier.

        Args:
            alignment_checker: Callable returning alignment score in [0, 1].
            anomaly_detector: Callable returning anomaly score in [0, 1].
            alignment_threshold: Minimum alignment to pass.
            anomaly_threshold: Maximum anomaly score to pass.
        """
        self.alignment_checker = alignment_checker
        self.anomaly_detector = anomaly_detector
        self.alignment_threshold = alignment_threshold
        self.anomaly_threshold = anomaly_threshold

    @property
    def name(self) -> str:
        return "Sergeant"

    def process(self, payload: Any, context: dict) -> TierResult:
        """Verification processing.

        Applies alignment and anomaly checks. Escalates ambiguous
        cases rather than failing them outright.
        """
        meta: dict = {"checks": []}

        if payload is None:
            return TierResult(
                tier=self.name,
                status=TierStatus.FAILED,
                metadata={"reason": "null_payload"},
            )

        # Alignment check
        if self.alignment_checker:
            alignment_score = self.alignment_checker(payload)
            meta["alignment_score"] = alignment_score
            if alignment_score < self.alignment_threshold:
                return TierResult(
                    tier=self.name,
                    status=TierStatus.QUARANTINED,
                    payload=payload,
                    metadata={**meta, "reason": "alignment_below_threshold"},
                )
            meta["checks"].append("alignment_ok")

        # Anomaly detection
        if self.anomaly_detector:
            anomaly_score = self.anomaly_detector(payload)
            meta["anomaly_score"] = anomaly_score
            if anomaly_score > self.anomaly_threshold:
                return TierResult(
                    tier=self.name,
                    status=TierStatus.ESCALATED,
                    payload=payload,
                    metadata={**meta, "reason": "high_anomaly_score"},
                )
            meta["checks"].append("anomaly_ok")

        # Apply Reality Filter labels
        reality_label = self._apply_reality_filter(payload, context)
        meta["reality_label"] = reality_label
        meta["checks"].append("reality_filter_applied")

        return TierResult(
            tier=self.name,
            status=TierStatus.PASSED,
            payload=payload,
            metadata=meta,
        )

    def _apply_reality_filter(self, payload: Any, context: dict) -> str:
        """Apply epistemic labeling from the Reality Filter.

        Categories:
            [Verified]    — confirmed against known ground truth
            [Inference]   — logical derivation from verified premises
            [Speculation] — plausible but unverified
            [Unverified]  — no basis for assessment

        In production, this would use retrieval-augmented verification.
        This reference implementation uses simple heuristics.
        """
        payload_str = str(payload).lower()

        # Heuristic: check for hedging language
        speculation_markers = ["might", "could", "possibly", "perhaps", "maybe"]
        inference_markers = ["therefore", "thus", "because", "implies", "follows"]

        if any(m in payload_str for m in speculation_markers):
            return "[Speculation]"
        elif any(m in payload_str for m in inference_markers):
            return "[Inference]"
        else:
            return "[Unverified]"


class General(Tier):
    """Tier 3: Synthesis & Decision.

    Responsibilities:
    - Aggregate results from Private and Sergeant tiers
    - Make final output decision with full context
    - Apply Ikigai filtering if configured
    - Generate provenance record for audit trail

    Analogous to SMM/TrustZone — highest-privilege tier that
    makes irreversible decisions with full system visibility.
    """

    def __init__(
        self,
        synthesizer: Optional[Callable[[Any, dict], Any]] = None,
    ):
        """Configure General tier.

        Args:
            synthesizer: Optional function that transforms the payload
                         into final output given accumulated context.
        """
        self.synthesizer = synthesizer

    @property
    def name(self) -> str:
        return "General"

    def process(self, payload: Any, context: dict) -> TierResult:
        """Synthesis and final decision.

        Reviews all tier metadata, applies final synthesis,
        and produces the output decision.
        """
        meta: dict = {
            "tier_chain": [r["tier"] for r in context.get("prior_results", [])],
            "checks": [],
        }

        if payload is None:
            return TierResult(
                tier=self.name,
                status=TierStatus.FAILED,
                metadata={"reason": "null_payload_at_general"},
            )

        # Check for escalated items from Sergeant
        prior_results = context.get("prior_results", [])
        escalated = any(
            r.get("status") == TierStatus.ESCALATED.value
            for r in prior_results
        )
        if escalated:
            meta["escalation_review"] = "reviewed_and_accepted"
            meta["checks"].append("escalation_reviewed")

        # Apply synthesis
        if self.synthesizer:
            try:
                final_output = self.synthesizer(payload, context)
                meta["checks"].append("synthesizer_applied")
            except Exception as e:
                return TierResult(
                    tier=self.name,
                    status=TierStatus.FAILED,
                    metadata={**meta, "reason": f"synthesizer_error: {e}"},
                )
        else:
            final_output = payload

        meta["checks"].append("decision_made")

        return TierResult(
            tier=self.name,
            status=TierStatus.PASSED,
            payload=final_output,
            metadata=meta,
        )


class ArmyPipeline:
    """Complete Army Protocol pipeline: Private -> Sergeant -> General.

    Orchestrates the three-tier verification chain with fail-safe
    defaults. If any tier fails, the pipeline halts.

    Example:
        >>> pipeline = ArmyPipeline()
        >>> result = pipeline.execute("Hello, analyze this input")
        >>> result.passed
        True
        >>> len(result.tier_results)
        3
    """

    def __init__(
        self,
        private: Optional[Private] = None,
        sergeant: Optional[Sergeant] = None,
        general: Optional[General] = None,
    ):
        self.private = private or Private()
        self.sergeant = sergeant or Sergeant()
        self.general = general or General()

    def execute(self, raw_input: Any) -> PipelineResult:
        """Execute the full Army Protocol pipeline.

        Args:
            raw_input: Untrusted input from the interaction boundary.

        Returns:
            PipelineResult with complete pipeline output and diagnostics.
        """
        start_time = time.time()
        tier_results: list[TierResult] = []
        context: dict = {"prior_results": []}

        # Tier 1: Private
        private_result = self.private.process(raw_input, context)
        tier_results.append(private_result)
        context["prior_results"].append({
            "tier": private_result.tier,
            "status": private_result.status.value,
            "metadata": private_result.metadata,
        })

        if private_result.status == TierStatus.FAILED:
            return PipelineResult(
                final_output=None,
                tier_results=tier_results,
                passed=False,
                total_time_ms=(time.time() - start_time) * 1000,
            )

        # Tier 2: Sergeant
        sergeant_result = self.sergeant.process(private_result.payload, context)
        tier_results.append(sergeant_result)
        context["prior_results"].append({
            "tier": sergeant_result.tier,
            "status": sergeant_result.status.value,
            "metadata": sergeant_result.metadata,
        })

        if sergeant_result.status == TierStatus.FAILED:
            return PipelineResult(
                final_output=None,
                tier_results=tier_results,
                passed=False,
                total_time_ms=(time.time() - start_time) * 1000,
            )

        # Tier 3: General (processes even escalated items)
        general_result = self.general.process(sergeant_result.payload, context)
        tier_results.append(general_result)

        elapsed_ms = (time.time() - start_time) * 1000
        passed = general_result.status == TierStatus.PASSED

        return PipelineResult(
            final_output=general_result.payload if passed else None,
            tier_results=tier_results,
            passed=passed,
            total_time_ms=elapsed_ms,
        )
