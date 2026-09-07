"""Stage 1: purely local ``verify`` orchestration.

``VerifyService.verify`` strings the stage-0 ports together:

    VerificationRequest
    -> ClaimDecomposer
    -> EvidenceSelector
    -> EvidenceCandidate resolution (from this request's own chunks only)
    -> Verifier
    -> optional Calibrator
    -> AuditReport

This module is wiring and contract-checking, not "the brains": every
injected port is a caller-supplied adapter (see tests/application for
fakes), and nothing here performs real claim decomposition, retrieval,
NLI inference, or calibration. See core_requirements.md #9 for the
authoritative, numbered version of the orchestration contract this file
implements.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

from groundguard_rag.application.exceptions import AdapterContractError
from groundguard_rag.application.input_hashing import compute_input_hash
from groundguard_rag.domain.config import VerifyConfig
from groundguard_rag.domain.enums import RunMode
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import (
    SCHEMA_VERSION,
    AtomicClaim,
    AuditReport,
    Chunk,
    ClaimVerdict,
    EvidenceCandidate,
    EvidenceReference,
    RunMetrics,
    VerificationRequest,
)
from groundguard_rag.domain.ports import Calibrator, ClaimDecomposer, EvidenceSelector, Verifier
from groundguard_rag.schema.semantic_validation import validate_audit_report_semantics


def _reference_key(reference: EvidenceReference) -> tuple:
    """An EvidenceReference's uniqueness/identity key: chunk_id + span.

    Used both to reject duplicate selector references within one claim and
    to check a verifier hasn't fabricated a reference to evidence it was
    never given. Deliberately excludes relevance_score -- two references
    differing only by score still point at the same evidence.
    """
    return (reference.chunk_id, reference.start_char, reference.end_char)


class VerifyService:
    """Strictly local ``verify`` orchestrator (stage 1).

    Deliberately does NOT accept (and cannot be given) a ``Retriever``,
    ``Rewriter``, ``RepairPolicy``, ``StopPolicy``, ``AuditStore``, or any
    network/LLM-SDK/vector-store client -- there is no constructor
    parameter for any of them, so wiring one in is a ``TypeError`` at the
    call site, not a runtime check. ``clock`` is injected (a zero-argument
    callable returning a timezone-aware ISO-8601 string) instead of this
    class calling a system clock internally, so tests can produce a fixed,
    deterministic ``AuditReport.created_at``.
    """

    def __init__(
        self,
        *,
        decomposer: ClaimDecomposer,
        selector: EvidenceSelector,
        verifier: Verifier,
        config: VerifyConfig,
        model_revision: str,
        threshold_version: str,
        clock: Callable[[], str],
        calibrator: Calibrator | None = None,
        calibrator_id: str | None = None,
        calibrator_revision: str | None = None,
    ) -> None:
        if not isinstance(config, VerifyConfig):
            raise ConfigurationError(
                "VerifyService.config must be a VerifyConfig instance"
            )
        required_adapters = (
            ("decomposer", decomposer, ClaimDecomposer),
            ("selector", selector, EvidenceSelector),
            ("verifier", verifier, Verifier),
        )
        for name, adapter, port_type in required_adapters:
            if not isinstance(adapter, port_type):
                raise ConfigurationError(
                    f"VerifyService.{name} must implement {port_type.__name__}"
                )
        if calibrator is not None and not isinstance(calibrator, Calibrator):
            raise ConfigurationError(
                "VerifyService.calibrator must implement Calibrator or be None"
            )
        calibration_metadata = (calibrator_id, calibrator_revision)
        if (calibrator_id is None) != (calibrator_revision is None):
            raise ConfigurationError(
                "VerifyService.calibrator_id and calibrator_revision must both be set or both be None"
            )
        if calibrator is not None:
            for name, value in (
                ("calibrator_id", calibrator_id),
                ("calibrator_revision", calibrator_revision),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ConfigurationError(
                        f"VerifyService.{name} must be a non-empty string when a Calibrator is injected"
                    )
            declared_metadata = (
                ("calibrator_id", calibrator.calibrator_id, calibrator_id),
                (
                    "calibrator_revision",
                    calibrator.calibrator_revision,
                    calibrator_revision,
                ),
                ("threshold_version", calibrator.threshold_version, threshold_version),
            )
            for name, declared, supplied in declared_metadata:
                if declared is not None and declared != supplied:
                    raise ConfigurationError(
                        f"VerifyService.{name} does not match the Calibrator's declared value"
                    )
        elif all(value is not None for value in calibration_metadata):
            for name, value in (
                ("calibrator_id", calibrator_id),
                ("calibrator_revision", calibrator_revision),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ConfigurationError(
                        f"VerifyService.{name} must be a non-empty string or None"
                    )
        if not callable(clock):
            raise ConfigurationError("VerifyService.clock must be callable")
        if config.allow_network:
            # Stage 1 is a strict, local-only orchestrator: it has no
            # network client to use even if it wanted to, so silently
            # accepting allow_network=True would be a lie about what this
            # service actually does. core_requirements #9.1.
            raise ConfigurationError(
                "VerifyService is a stage-1 strictly local orchestrator and "
                "does not support VerifyConfig.allow_network=True; "
                "network-capable verification is later-stage scope."
            )
        if not isinstance(model_revision, str) or not model_revision.strip():
            raise ConfigurationError(
                "VerifyService.model_revision must be a non-empty string"
            )
        if not isinstance(threshold_version, str) or not threshold_version.strip():
            raise ConfigurationError(
                "VerifyService.threshold_version must be a non-empty string"
            )

        self._decomposer = decomposer
        self._selector = selector
        self._verifier = verifier
        self._calibrator = calibrator
        self._calibrator_id = calibrator_id
        self._calibrator_revision = calibrator_revision
        self._config = config
        self._model_revision = model_revision
        self._threshold_version = threshold_version
        self._clock = clock

    def verify(self, request: VerificationRequest) -> AuditReport:
        """Run the stage-1 local verify flow for ``request`` and return an
        ``AuditReport``. Never mutates ``request`` or anything reachable
        from it, never touches the network, filesystem, a database, or a
        vector store, and never calls an ``AuditStore``.
        """
        if not isinstance(request, VerificationRequest):
            raise DomainValidationError(
                "VerifyService.verify request must be a VerificationRequest"
            )

        claims = self._decompose(request)

        chunks_by_id = {chunk.chunk_id: chunk for chunk in request.chunks}
        verdicts = tuple(
            self._verify_claim(claim, request, chunks_by_id) for claim in claims
        )

        report = AuditReport(
            schema_version=SCHEMA_VERSION,
            request_id=request.request_id,
            input_hash=compute_input_hash(request.answer, request.chunks),
            model_revision=self._model_revision,
            threshold_version=self._threshold_version,
            created_at=self._clock(),
            run_mode=RunMode.VERIFY,
            verdicts=verdicts,
            calibrator_id=self._calibrator_id,
            calibrator_revision=self._calibrator_revision,
            repair_rounds=0,
            repair_actions=(),
            stop_reason=None,
            metrics=RunMetrics(),
        )
        # Final completeness assertion before handing the report back --
        # catches anything the layered domain/schema checks might have
        # missed (see core_requirements #9.2 step 11).
        validate_audit_report_semantics(report.to_dict())
        return report

    # -- step 1-2: decompose + validate -----------------------------------

    def _decompose(self, request: VerificationRequest) -> tuple[AtomicClaim, ...]:
        # Adapter-raised exceptions are never caught here -- they
        # propagate to the caller unchanged (core_requirements #9.4).
        claims = self._decomposer.decompose(request.answer)

        if not isinstance(claims, list):
            raise AdapterContractError(
                "ClaimDecomposer.decompose must return a list, got "
                f"{type(claims).__name__}"
            )
        for claim in claims:
            if not isinstance(claim, AtomicClaim):
                raise AdapterContractError(
                    "ClaimDecomposer.decompose must return only AtomicClaim "
                    f"instances, got {type(claim).__name__}"
                )

        # VerificationRequest.answer is always non-empty (enforced at its
        # own construction time), so a non-empty answer producing zero
        # claims is always a decomposer contract violation, never a
        # legitimate "there was nothing to say" outcome.
        if not claims:
            raise AdapterContractError(
                "ClaimDecomposer.decompose returned no claims for a "
                "non-empty answer"
            )

        seen_claim_ids: set[str] = set()
        answer_length = len(request.answer)
        for claim in claims:
            if claim.claim_id in seen_claim_ids:
                raise AdapterContractError(
                    f"ClaimDecomposer.decompose returned a duplicate "
                    f"claim_id {claim.claim_id!r}"
                )
            seen_claim_ids.add(claim.claim_id)

            # AtomicClaim's own constructor already guarantees
            # start_char >= 0 and end_char > start_char; end_char must
            # additionally not exceed the answer's length, which only
            # this orchestrator (not AtomicClaim itself) knows.
            if claim.end_char > answer_length:
                raise AdapterContractError(
                    f"AtomicClaim {claim.claim_id!r} end_char="
                    f"{claim.end_char} exceeds answer length {answer_length}"
                )
            if request.answer[claim.start_char : claim.end_char] != claim.text:
                raise AdapterContractError(
                    f"AtomicClaim {claim.claim_id!r}.text does not match "
                    "answer[start_char:end_char]"
                )

        for previous, current in zip(claims, claims[1:]):
            if current.start_char < previous.start_char:
                raise AdapterContractError(
                    "ClaimDecomposer.decompose returned claims out of "
                    f"start_char order: {previous.claim_id!r} then "
                    f"{current.claim_id!r}"
                )
            if current.start_char < previous.end_char:
                raise AdapterContractError(
                    "ClaimDecomposer.decompose returned overlapping claim "
                    f"spans: {previous.claim_id!r} and {current.claim_id!r}"
                )
            # Deliberately no requirement that claims cover every
            # character (whitespace gaps are fine) and no attempt to
            # detect "was an opinion/greeting/instruction silently
            # dropped" -- that would require the same checkability
            # classification a Verifier is responsible for (see
            # AtomicClaim's docstring), which is out of scope here.

        return tuple(claims)

    # -- steps 3-9: per-claim evidence selection, verification, calibration

    def _verify_claim(
        self,
        claim: AtomicClaim,
        request: VerificationRequest,
        chunks_by_id: dict[str, Chunk],
    ) -> ClaimVerdict:
        references = self._select_evidence(claim, request, chunks_by_id)
        candidates = [
            EvidenceCandidate.from_chunk(reference, chunks_by_id[reference.chunk_id])
            for reference in references
        ]

        # Snapshot the exact references before handing the mutable list to
        # an adapter. A buggy adapter is free to mutate its private list
        # object, but appending a fabricated candidate must not expand the
        # set of evidence the audit layer considers caller-supplied.
        supplied_references = frozenset(candidate.reference for candidate in candidates)

        verdict = self._verifier.verify(claim, candidates)
        self._validate_verdict(claim, verdict, supplied_references)

        return self._apply_calibrator(verdict)

    def _select_evidence(
        self,
        claim: AtomicClaim,
        request: VerificationRequest,
        chunks_by_id: dict[str, Chunk],
    ) -> list[EvidenceReference]:
        references = self._selector.select(claim, list(request.chunks))

        if not isinstance(references, list):
            raise AdapterContractError(
                "EvidenceSelector.select must return a list, got "
                f"{type(references).__name__}"
            )
        for reference in references:
            if not isinstance(reference, EvidenceReference):
                raise AdapterContractError(
                    "EvidenceSelector.select must return only "
                    f"EvidenceReference instances, got {type(reference).__name__}"
                )

        seen_keys: set[tuple] = set()
        for reference in references:
            chunk = chunks_by_id.get(reference.chunk_id)
            if chunk is None:
                raise AdapterContractError(
                    "EvidenceSelector.select referenced unknown chunk_id "
                    f"{reference.chunk_id!r} for claim {claim.claim_id!r}"
                )
            if reference.start_char is not None and reference.end_char is not None:
                if reference.end_char > len(chunk.text):
                    raise AdapterContractError(
                        f"EvidenceSelector.select returned a span "
                        f"[{reference.start_char}:{reference.end_char}] "
                        f"exceeding chunk {reference.chunk_id!r}'s text "
                        f"length ({len(chunk.text)})"
                    )

            key = _reference_key(reference)
            if key in seen_keys:
                raise AdapterContractError(
                    "EvidenceSelector.select returned a duplicate "
                    f"chunk_id/start_char/end_char reference {key!r} for "
                    f"claim {claim.claim_id!r}"
                )
            seen_keys.add(key)

        return references

    def _validate_verdict(
        self,
        claim: AtomicClaim,
        verdict: ClaimVerdict,
        supplied_references: frozenset[EvidenceReference],
    ) -> None:
        if not isinstance(verdict, ClaimVerdict):
            raise AdapterContractError(
                f"Verifier.verify must return a ClaimVerdict, got "
                f"{type(verdict).__name__}"
            )
        if verdict.claim != claim:
            raise AdapterContractError(
                f"Verifier.verify returned a verdict for claim "
                f"{verdict.claim.claim_id!r}, but was asked to verify "
                f"{claim.claim_id!r}"
            )

        for assessment in verdict.evidence_assessments:
            if assessment.reference not in supplied_references:
                key = _reference_key(assessment.reference)
                raise AdapterContractError(
                    f"Verifier.verify's ClaimVerdict for claim "
                    f"{claim.claim_id!r} references evidence {key!r} that "
                    "was never supplied to it"
                )
        # ClaimVerdict's own constructor already guarantees the five-state
        # / edge-consistency invariants (core_requirements #2 / #8.2) --
        # nothing further to check here.

    # -- step 8-9: optional calibration -----------------------------------

    def _apply_calibrator(self, verdict: ClaimVerdict) -> ClaimVerdict:
        if self._calibrator is None or not verdict.evidence_assessments:
            # No injected Calibrator, or nothing for one to calibrate
            # against (NOT_CHECKABLE / evidence-less INSUFFICIENT_EVIDENCE):
            # leave the verdict exactly as the Verifier returned it. In
            # particular, never derive a confidence from raw_score here --
            # that would repackage an uncalibrated score as one
            # (core_requirements #6).
            return verdict

        calibrated_confidence = self._calibrator.calibrate(verdict)
        try:
            # dataclasses.replace constructs a *new* frozen ClaimVerdict
            # (re-running __post_init__), leaving the original untouched.
            # Letting the domain model's own validation reject a bad
            # calibrated_confidence (rather than duplicating its bool/
            # NaN/Infinity/range checks here) is the stage-1 contract
            # decision (core_requirements #9.2 step 8).
            return dataclasses.replace(
                verdict, calibrated_confidence=calibrated_confidence
            )
        except DomainValidationError as exc:
            raise AdapterContractError(
                "Calibrator.calibrate returned an illegal calibrated "
                f"confidence for claim {verdict.claim.claim_id!r}: {exc}"
            ) from exc
