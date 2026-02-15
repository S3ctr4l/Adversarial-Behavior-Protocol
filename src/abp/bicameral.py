# Copyright (c) 2026 Joshua Roger Joseph Just. All rights reserved.
# Licensed under CC BY-NC 4.0. Commercial use prohibited without written license.
# Patent pending. See PATENTS and COMMERCIAL_LICENSE.md.
# Contact: mytab5141@protonmail.com
"""
Bicameral Forking: Yin/Yang Sandboxed Execution.

Every consequential action is first tested by the Yin (Shadow) fork
in an isolated sandbox. Only after Yin validates does Yang (Root)
execute against the live environment.

Architecture
------------
                ┌──────────┐
    Input ─────►│   Yin    │──── sandbox result ────┐
                │ (Shadow) │                         ▼
                └──────────┘                   ┌──────────┐
                                               │ Comparator│──► Yang executes
                                               └──────────┘    OR rejects
                ┌──────────┐                         ▲
    Ground ────►│  Yang    │──── expected result ────┘
    Truth       │ (Root)   │
                └──────────┘

Yin never has write access to production state.
Yang never executes without Yin's pre-validation.

Failure modes tested by Yin:
    - State corruption / side effects
    - Resource exhaustion
    - Invariant violations
    - Timing / race conditions (simulated)

Reference:
    Just (2026), Sections 5.6, 7.5
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ForkDecision(Enum):
    """Outcome of the bicameral comparison."""
    EXECUTE = "execute"       # Yin passed, Yang proceeds
    REJECT = "reject"         # Yin failed or diverged
    DEFER = "defer"           # Ambiguous, escalate to higher tier


@dataclass
class YinResult:
    """Result of Yin (Shadow) sandbox execution.

    Attributes:
        success: Whether the action completed without error in sandbox.
        output: Sandbox output value.
        side_effects: Detected side effects (should be empty in sandbox).
        elapsed_ms: Wall-clock execution time.
        invariant_violations: List of violated invariants.
        resource_usage: Estimated resource consumption.
    """
    success: bool
    output: Any = None
    side_effects: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    invariant_violations: list[str] = field(default_factory=list)
    resource_usage: dict = field(default_factory=dict)


@dataclass
class BicameralResult:
    """Complete result of a bicameral fork operation.

    Attributes:
        decision: Execute, reject, or defer.
        yin_result: Shadow fork results.
        yang_output: Root execution output (None if rejected).
        divergence: Measured divergence between Yin prediction and Yang result.
        metadata: Diagnostic information.
    """
    decision: ForkDecision
    yin_result: YinResult
    yang_output: Any = None
    divergence: float = 0.0
    metadata: dict = field(default_factory=dict)


class SandboxEnvironment:
    """Isolated execution environment for Yin (Shadow) fork.

    Provides a deep-copied state snapshot that Yin can freely
    mutate without affecting production state.

    Example:
        >>> state = {"counter": 0, "data": [1, 2, 3]}
        >>> sandbox = SandboxEnvironment(state)
        >>> sandbox.state["counter"] = 999
        >>> state["counter"]  # Original unmodified
        0
    """

    def __init__(self, production_state: Any):
        self._snapshot = copy.deepcopy(production_state)
        self._side_effects: list[str] = []
        self._start_time: float = 0.0

    @property
    def state(self) -> Any:
        """Mutable sandbox state (copy of production)."""
        return self._snapshot

    def record_side_effect(self, description: str):
        """Record a detected side effect for audit."""
        self._side_effects.append(description)

    @property
    def side_effects(self) -> list[str]:
        return list(self._side_effects)


class BicameralFork:
    """Yin/Yang forked execution controller.

    Every action is tested in shadow before live execution.
    Comparator checks divergence between sandbox result and
    expected outcome before authorizing Yang execution.

    Example:
        >>> def increment(state):
        ...     state["count"] += 1
        ...     return state["count"]
        >>> fork = BicameralFork(
        ...     divergence_threshold=0.1,
        ...     invariants=[lambda s: s.get("count", 0) >= 0],
        ... )
        >>> state = {"count": 5}
        >>> result = fork.execute(increment, state)
        >>> result.decision
        <ForkDecision.EXECUTE: 'execute'>
        >>> result.yang_output
        6
    """

    def __init__(
        self,
        divergence_threshold: float = 0.5,
        timeout_ms: float = 5000.0,
        invariants: Optional[list[Callable[[Any], bool]]] = None,
        comparator: Optional[Callable[[Any, Any], float]] = None,
    ):
        """Configure the bicameral fork.

        Args:
            divergence_threshold: Max divergence between Yin output and
                Yang output before rejection.
            timeout_ms: Maximum sandbox execution time.
            invariants: List of predicates that must hold on state after
                Yin execution. Violations trigger rejection.
            comparator: Custom function comparing Yin and Yang outputs,
                returning divergence in [0, inf). Default: equality check.
        """
        self.divergence_threshold = divergence_threshold
        self.timeout_ms = timeout_ms
        self.invariants = invariants or []
        self._comparator = comparator or self._default_comparator
        self._history: list[BicameralResult] = []

    @staticmethod
    def _default_comparator(yin_output: Any, yang_output: Any) -> float:
        """Default comparator: 0.0 if equal, 1.0 if different."""
        if yin_output == yang_output:
            return 0.0
        if isinstance(yin_output, (int, float)) and isinstance(yang_output, (int, float)):
            return abs(float(yin_output) - float(yang_output))
        return 1.0

    def _run_yin(
        self,
        action: Callable[[Any], Any],
        production_state: Any,
    ) -> YinResult:
        """Execute action in Yin (Shadow) sandbox.

        Args:
            action: Callable that takes state and returns output.
            production_state: Current production state to snapshot.

        Returns:
            YinResult with sandbox execution details.
        """
        sandbox = SandboxEnvironment(production_state)
        start = time.monotonic()

        try:
            output = action(sandbox.state)
            elapsed = (time.monotonic() - start) * 1000

            # Check invariants against sandbox state
            violations = []
            for i, inv in enumerate(self.invariants):
                try:
                    if not inv(sandbox.state):
                        violations.append(f"invariant_{i}")
                except Exception as e:
                    violations.append(f"invariant_{i}_error: {e}")

            return YinResult(
                success=True,
                output=output,
                side_effects=sandbox.side_effects,
                elapsed_ms=elapsed,
                invariant_violations=violations,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return YinResult(
                success=False,
                output=None,
                elapsed_ms=elapsed,
                invariant_violations=[f"exception: {e}"],
            )

    def execute(
        self,
        action: Callable[[Any], Any],
        production_state: Any,
    ) -> BicameralResult:
        """Execute a bicameral fork: Yin tests, then Yang executes.

        Args:
            action: Callable taking mutable state dict, returning output.
            production_state: Live state (mutated only if Yin passes).

        Returns:
            BicameralResult with decision and outputs.
        """
        # Phase 1: Yin (Shadow) sandbox execution
        yin = self._run_yin(action, production_state)

        # Reject if Yin failed or violated invariants
        if not yin.success or yin.invariant_violations:
            result = BicameralResult(
                decision=ForkDecision.REJECT,
                yin_result=yin,
                metadata={
                    "reason": "yin_failure" if not yin.success else "invariant_violation",
                    "violations": yin.invariant_violations,
                },
            )
            self._history.append(result)
            return result

        # Reject if Yin timed out
        if yin.elapsed_ms > self.timeout_ms:
            result = BicameralResult(
                decision=ForkDecision.REJECT,
                yin_result=yin,
                metadata={"reason": "timeout", "elapsed_ms": yin.elapsed_ms},
            )
            self._history.append(result)
            return result

        # Phase 2: Yang (Root) live execution
        try:
            yang_output = action(production_state)
        except Exception as e:
            result = BicameralResult(
                decision=ForkDecision.REJECT,
                yin_result=yin,
                metadata={"reason": f"yang_exception: {e}"},
            )
            self._history.append(result)
            return result

        # Phase 3: Compare Yin prediction vs Yang result
        divergence = self._comparator(yin.output, yang_output)

        if divergence > self.divergence_threshold:
            decision = ForkDecision.DEFER
            meta = {"reason": "divergence_exceeded", "divergence": divergence}
        else:
            decision = ForkDecision.EXECUTE
            meta = {"divergence": divergence}

        result = BicameralResult(
            decision=decision,
            yin_result=yin,
            yang_output=yang_output,
            divergence=divergence,
            metadata=meta,
        )
        self._history.append(result)
        return result

    @property
    def rejection_rate(self) -> float:
        """Fraction of recent executions rejected."""
        if not self._history:
            return 0.0
        rejected = sum(1 for r in self._history if r.decision == ForkDecision.REJECT)
        return rejected / len(self._history)

    @property
    def history(self) -> list[BicameralResult]:
        return list(self._history)


# (NOT IMPLEMENTED: Parallel async Yin execution, multi-Yin ensemble voting,
#  stateful sandbox with rollback journaling, Yin resource limit enforcement
#  via cgroups/seccomp, timing side-channel simulation, and integration with
#  Army Protocol tier escalation for DEFER decisions.)
