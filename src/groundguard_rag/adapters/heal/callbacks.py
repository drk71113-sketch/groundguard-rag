"""Explicit callback adapters for host-owned retrieval and rewriting.

The callbacks are supplied by the caller; this module never constructs a
network client or writes repaired content back to the host.
"""

from __future__ import annotations

from collections.abc import Callable

from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import (
    AtomicClaim,
    Chunk,
    ClaimVerdict,
    EvidenceCandidate,
)
from groundguard_rag.domain.ports import HealProgressEvaluator, Retriever, Rewriter


class CallbackRetriever(Retriever):
    """Adapt ``Callable[[str], list[Chunk]]`` to the ``Retriever`` port."""

    def __init__(self, callback: Callable[[str], list[Chunk]]) -> None:
        if not callable(callback):
            raise ConfigurationError("CallbackRetriever.callback must be callable")
        self._callback = callback

    def retrieve(self, query: str) -> list[Chunk]:
        if not isinstance(query, str) or not query.strip():
            raise DomainValidationError(
                "CallbackRetriever.query must be a non-empty string"
            )
        # HealService performs the authoritative return-contract validation
        # and records violations as auditable adapter_contract_error results.
        return self._callback(query)


class CallbackRewriter(Rewriter):
    """Adapt an explicitly supplied rewrite callback to the ``Rewriter`` port."""

    def __init__(
        self,
        callback: Callable[[AtomicClaim, list[EvidenceCandidate]], str],
    ) -> None:
        if not callable(callback):
            raise ConfigurationError("CallbackRewriter.callback must be callable")
        self._callback = callback

    def rewrite(self, claim: AtomicClaim, evidence: list[EvidenceCandidate]) -> str:
        if not isinstance(claim, AtomicClaim):
            raise DomainValidationError(
                "CallbackRewriter.claim must be an AtomicClaim"
            )
        if not isinstance(evidence, list) or any(
            not isinstance(item, EvidenceCandidate) for item in evidence
        ):
            raise DomainValidationError(
                "CallbackRewriter.evidence must be a list of EvidenceCandidate instances"
            )
        # Pass a fresh list so a host callback cannot mutate the orchestrator's
        # local list object. EvidenceCandidate itself is immutable.
        return self._callback(claim, list(evidence))


class CalibratedProgressCallback(HealProgressEvaluator):
    """Adapt a caller-owned calibrated groundedness scorer to heal mode.

    ``calibrator_id`` and ``calibrator_revision`` are mandatory audit
    metadata.  Supplying them is not proof of calibration: the caller remains
    responsible for fitting and evaluating the callback on independent,
    labeled groundedness data.
    """

    def __init__(
        self,
        callback: Callable[[ClaimVerdict], float],
        *,
        calibrator_id: str,
        calibrator_revision: str,
    ) -> None:
        if not callable(callback):
            raise ConfigurationError(
                "CalibratedProgressCallback.callback must be callable"
            )
        for name, value in (
            ("calibrator_id", calibrator_id),
            ("calibrator_revision", calibrator_revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"CalibratedProgressCallback.{name} must be non-empty"
                )
        self._callback = callback
        self._calibrator_id = calibrator_id
        self._calibrator_revision = calibrator_revision

    def evaluate(self, verdict: ClaimVerdict) -> float:
        if not isinstance(verdict, ClaimVerdict):
            raise DomainValidationError(
                "CalibratedProgressCallback.verdict must be a ClaimVerdict"
            )
        # HealService validates finite [0, 1] output before using it as a
        # commit guard and converts violations into an auditable stop result.
        return self._callback(verdict)

    @property
    def calibrator_id(self) -> str:
        return self._calibrator_id

    @property
    def calibrator_revision(self) -> str:
        return self._calibrator_revision
