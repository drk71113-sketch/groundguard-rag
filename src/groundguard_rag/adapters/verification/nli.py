"""Provider-neutral three-way NLI verifier core.

The injected backend owns provider-specific label mapping. This module only
normalizes logits, applies explicit decision thresholds, records one audit
edge per supplied evidence candidate, and aggregates edge states. Model
probabilities remain uncalibrated and are never exposed as claim confidence.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import (
    ConfigurationError,
    DomainValidationError,
    GroundGuardError,
)
from groundguard_rag.domain.models import (
    AtomicClaim,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceCandidate,
    LabelScores,
)
from groundguard_rag.domain.ports import Verifier


class NliBackendContractError(GroundGuardError):
    """An NLI backend returned a value outside its explicit contract."""


class NliBackend(ABC):
    """Provider-specific premise/hypothesis inference boundary.

    Implementations must map their own model labels to domain semantics before
    returning. ``supported`` means entailment, ``contradicted`` means
    contradiction, and ``insufficient`` means neutral. ``NliVerifier`` never
    guesses the meaning of provider labels such as ``LABEL_0``.
    """

    @abstractmethod
    def predict_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[LabelScores]:
        """Return one three-way score object per input pair, in order."""
        raise NotImplementedError


@dataclass(frozen=True)
class NliVerifierConfig:
    verifier_id: str
    verifier_revision: str
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.verifier_id, str) or not self.verifier_id.strip():
            raise ConfigurationError(
                "NliVerifierConfig.verifier_id must be a non-empty string"
            )
        if not isinstance(self.verifier_revision, str) or not self.verifier_revision.strip():
            raise ConfigurationError(
                "NliVerifierConfig.verifier_revision must be a non-empty string"
            )
        threshold = self.decision_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ConfigurationError(
                "NliVerifierConfig.decision_threshold must be a non-bool real number"
            )
        if not math.isfinite(threshold) or not (0.5 <= threshold <= 1.0):
            raise ConfigurationError(
                "NliVerifierConfig.decision_threshold must be finite and within [0.5, 1]"
            )


class NliVerifier(Verifier):
    """Create auditable NLI edges and aggregate them into a claim verdict."""

    def __init__(self, backend: NliBackend, config: NliVerifierConfig) -> None:
        if not isinstance(backend, NliBackend):
            raise ConfigurationError("NliVerifier.backend must implement NliBackend")
        if not isinstance(config, NliVerifierConfig):
            raise ConfigurationError(
                "NliVerifier.config must be an NliVerifierConfig instance"
            )
        self._backend = backend
        self._config = config

    @property
    def config(self) -> NliVerifierConfig:
        return self._config

    def verify(
        self, claim: AtomicClaim, evidence: list[EvidenceCandidate]
    ) -> ClaimVerdict:
        self._validate_inputs(claim, evidence)

        if not evidence:
            return ClaimVerdict(
                claim=claim,
                state=VerificationState.INSUFFICIENT_EVIDENCE,
                evidence_assessments=(),
                raw_score=None,
                calibrated_confidence=None,
                rationale=None,
            )

        pairs = [(candidate.text, claim.text) for candidate in evidence]
        # Backend exceptions propagate unchanged. They are execution failures,
        # not successful return values that violate a contract.
        backend_outputs = self._backend.predict_batch(pairs)
        if not isinstance(backend_outputs, list):
            raise NliBackendContractError(
                "NliBackend.predict_batch must return a list, got "
                f"{type(backend_outputs).__name__}"
            )
        if len(backend_outputs) != len(evidence):
            raise NliBackendContractError(
                "NliBackend.predict_batch must return exactly one LabelScores "
                f"per pair (expected {len(evidence)}, got {len(backend_outputs)})"
            )

        assessments: list[EvidenceAssessment] = []
        for candidate, backend_scores in zip(evidence, backend_outputs):
            probabilities = self._as_probabilities(backend_scores)
            state = self._edge_state(probabilities)
            assessments.append(
                EvidenceAssessment(
                    reference=candidate.reference,
                    label_scores=probabilities,
                    state=state,
                    rationale=None,
                    verifier_id=self._config.verifier_id,
                    verifier_revision=self._config.verifier_revision,
                )
            )

        aggregate_state = self._aggregate_state(assessments)
        return ClaimVerdict(
            claim=claim,
            state=aggregate_state,
            evidence_assessments=tuple(assessments),
            raw_score=None,
            calibrated_confidence=None,
            rationale=None,
        )

    @staticmethod
    def _validate_inputs(claim: AtomicClaim, evidence: list[EvidenceCandidate]) -> None:
        if not isinstance(claim, AtomicClaim):
            raise DomainValidationError("NliVerifier.claim must be an AtomicClaim")
        if not isinstance(evidence, list):
            raise DomainValidationError("NliVerifier.evidence must be a list")

        seen_references: set[tuple[str, int | None, int | None]] = set()
        for candidate in evidence:
            if not isinstance(candidate, EvidenceCandidate):
                raise DomainValidationError(
                    "NliVerifier.evidence must contain only EvidenceCandidate instances"
                )
            reference = candidate.reference
            key = (reference.chunk_id, reference.start_char, reference.end_char)
            if key in seen_references:
                raise DomainValidationError(
                    "NliVerifier.evidence contains a duplicate chunk/span reference "
                    f"{key!r}"
                )
            seen_references.add(key)

    @staticmethod
    def _as_probabilities(scores: LabelScores) -> LabelScores:
        if not isinstance(scores, LabelScores):
            raise NliBackendContractError(
                "NliBackend.predict must return LabelScores, got "
                f"{type(scores).__name__}"
            )
        if scores.score_kind == "probabilities":
            return scores
        if scores.score_kind != "logits":
            raise NliBackendContractError(
                "NliBackend.predict LabelScores.score_kind must be "
                f"'logits' or 'probabilities', got {scores.score_kind!r}"
            )

        raw_values = (
            scores.supported,
            scores.contradicted,
            scores.insufficient,
        )
        maximum = max(raw_values)
        exponentials = tuple(math.exp(value - maximum) for value in raw_values)
        total = sum(exponentials)
        return LabelScores(
            supported=exponentials[0] / total,
            contradicted=exponentials[1] / total,
            insufficient=exponentials[2] / total,
            score_kind="probabilities",
        )

    def _edge_state(self, probabilities: LabelScores) -> VerificationState:
        threshold = self._config.decision_threshold
        if (
            probabilities.supported >= threshold
            and probabilities.supported > probabilities.contradicted
        ):
            return VerificationState.SUPPORTED
        if (
            probabilities.contradicted >= threshold
            and probabilities.contradicted > probabilities.supported
        ):
            return VerificationState.CONTRADICTED
        return VerificationState.INSUFFICIENT_EVIDENCE

    @staticmethod
    def _aggregate_state(
        assessments: list[EvidenceAssessment],
    ) -> VerificationState:
        states = {assessment.state for assessment in assessments}
        has_supported = VerificationState.SUPPORTED in states
        has_contradicted = VerificationState.CONTRADICTED in states
        if has_supported and has_contradicted:
            return VerificationState.CONFLICTING_EVIDENCE
        if has_contradicted:
            return VerificationState.CONTRADICTED
        if has_supported:
            return VerificationState.SUPPORTED
        return VerificationState.INSUFFICIENT_EVIDENCE
