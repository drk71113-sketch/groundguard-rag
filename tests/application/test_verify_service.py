import inspect
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from groundguard_rag.application.exceptions import AdapterContractError
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.config import VerifyConfig
from groundguard_rag.domain.enums import RunMode, VerificationState
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import (
    AtomicClaim,
    Chunk,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceCandidate,
    EvidenceReference,
    LabelScores,
    VerificationRequest,
)
from groundguard_rag.domain.ports import Calibrator, ClaimDecomposer, EvidenceSelector, Verifier
from groundguard_rag.schema.semantic_validation import validate_audit_report_semantics


FIXED_TIME = "2026-09-02T00:00:00+00:00"


class FakeDecomposer(ClaimDecomposer):
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def decompose(self, answer):
        self.calls.append(answer)
        if self.error is not None:
            raise self.error
        return self.result


class FakeSelector(EvidenceSelector):
    def __init__(self, result=None, by_claim=None, mutate_input=False):
        self.result = [] if result is None else result
        self.by_claim = by_claim
        self.mutate_input = mutate_input
        self.calls = []

    def select(self, claim, chunks):
        self.calls.append((claim, list(chunks)))
        if self.mutate_input:
            chunks.clear()
        if self.by_claim is not None:
            return self.by_claim[claim.claim_id]
        return self.result


class FakeVerifier(Verifier):
    def __init__(self, result=None, callback=None, mutate_input=False):
        self.result = result
        self.callback = callback
        self.mutate_input = mutate_input
        self.calls = []

    def verify(self, claim, evidence):
        self.calls.append((claim, list(evidence)))
        if self.callback is not None:
            return self.callback(claim, evidence)
        if self.mutate_input:
            evidence.clear()
        return self.result


class FakeCalibrator(Calibrator):
    def __init__(self, value):
        self.value = value
        self.calls = []

    def calibrate(self, verdict):
        self.calls.append(verdict)
        return self.value


class SelfDescribingCalibrator(FakeCalibrator):
    @property
    def calibrator_id(self):
        return "declared-calibrator"

    @property
    def calibrator_revision(self):
        return "declared-revision"

    @property
    def threshold_version(self):
        return "declared-threshold"


def _claim(claim_id="claim-1", text="Paris is in France.", start=0, end=19):
    return AtomicClaim(claim_id=claim_id, text=text, start_char=start, end_char=end)


def _chunk(
    chunk_id="chunk-1",
    text="Paris is the capital of France.",
    metadata=None,
):
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source="facts.md",
        metadata={} if metadata is None else metadata,
    )


def _request(answer="Paris is in France.", chunks=None):
    if chunks is None:
        chunks = (_chunk(),)
    return VerificationRequest(
        request_id="request-1",
        answer=answer,
        chunks=chunks,
        query="ignored by verify",
    )


def _assessment(reference, state=VerificationState.SUPPORTED):
    if state is VerificationState.SUPPORTED:
        scores = LabelScores(0.9, 0.05, 0.05, "probabilities")
    elif state is VerificationState.CONTRADICTED:
        scores = LabelScores(0.05, 0.9, 0.05, "probabilities")
    else:
        scores = LabelScores(0.05, 0.05, 0.9, "probabilities")
    return EvidenceAssessment(
        reference=reference,
        label_scores=scores,
        state=state,
        rationale="fake test assessment",
        verifier_id="fake-verifier",
        verifier_revision="v1",
    )


def _supported_verdict(claim, reference, calibrated_confidence=None):
    return ClaimVerdict(
        claim=claim,
        state=VerificationState.SUPPORTED,
        evidence_assessments=(_assessment(reference),),
        raw_score=3.2,
        calibrated_confidence=calibrated_confidence,
        rationale="supported by test evidence",
    )


def _insufficient_verdict(claim):
    return ClaimVerdict(
        claim=claim,
        state=VerificationState.INSUFFICIENT_EVIDENCE,
        evidence_assessments=(),
    )


def _not_checkable_verdict(claim):
    return ClaimVerdict(
        claim=claim,
        state=VerificationState.NOT_CHECKABLE,
        evidence_assessments=(),
    )


def _service(
    *,
    claims=None,
    selector=None,
    verifier=None,
    calibrator=None,
    calibrator_id=None,
    calibrator_revision=None,
    config=None,
):
    if claims is None:
        claims = [_claim()]
    if selector is None:
        selector = FakeSelector(result=[])
    if verifier is None:
        verifier = FakeVerifier(result=_insufficient_verdict(claims[0]))
    if calibrator is not None and calibrator_id is None and calibrator_revision is None:
        calibrator_id = "fake-calibrator"
        calibrator_revision = "fake-calibrator-v1"
    return VerifyService(
        decomposer=FakeDecomposer(result=claims),
        selector=selector,
        verifier=verifier,
        calibrator=calibrator,
        calibrator_id=calibrator_id,
        calibrator_revision=calibrator_revision,
        config=VerifyConfig() if config is None else config,
        model_revision="pipeline-v1",
        threshold_version="thresholds-v1",
        clock=lambda: FIXED_TIME,
    )


def _validate_against_json_schema(payload):
    schema_path = (
        Path(__file__).parents[2]
        / "src"
        / "groundguard_rag"
        / "schema"
        / "audit_report.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(payload)


def test_happy_path_builds_a_valid_verify_audit_report():
    answer = "Paris is in France. Hello!"
    factual = _claim("factual", "Paris is in France.", 0, 19)
    greeting = _claim("greeting", "Hello!", 20, 26)
    chunk = _chunk()
    reference = EvidenceReference(chunk_id=chunk.chunk_id, relevance_score=0.8)
    selector = FakeSelector(by_claim={"factual": [reference], "greeting": []})

    def verify(claim, candidates):
        if claim.claim_id == "factual":
            assert candidates[0].text == chunk.text
            return _supported_verdict(claim, candidates[0].reference)
        assert candidates == []
        return _not_checkable_verdict(claim)

    verifier = FakeVerifier(callback=verify)
    calibrator = FakeCalibrator(0.93)
    service = _service(
        claims=[factual, greeting],
        selector=selector,
        verifier=verifier,
        calibrator=calibrator,
    )

    report = service.verify(_request(answer=answer, chunks=(chunk,)))

    assert report.run_mode is RunMode.VERIFY
    assert report.repair_rounds == 0
    assert report.repair_actions == ()
    assert report.stop_reason is None
    assert report.created_at == FIXED_TIME
    assert report.model_revision == "pipeline-v1"
    assert report.threshold_version == "thresholds-v1"
    assert report.calibrator_id == "fake-calibrator"
    assert report.calibrator_revision == "fake-calibrator-v1"
    assert [item.claim.claim_id for item in report.verdicts] == ["factual", "greeting"]
    assert report.verdicts[0].calibrated_confidence == 0.93
    assert report.verdicts[1].calibrated_confidence is None
    assert len(calibrator.calls) == 1

    payload = report.to_dict()
    _validate_against_json_schema(payload)
    validate_audit_report_semantics(payload)


def test_empty_chunks_can_produce_insufficient_evidence_without_calibration():
    claim = _claim()
    calibrator = FakeCalibrator(0.5)
    service = _service(
        claims=[claim],
        selector=FakeSelector(result=[]),
        verifier=FakeVerifier(result=_insufficient_verdict(claim)),
        calibrator=calibrator,
    )

    report = service.verify(_request(chunks=()))

    assert report.verdicts[0].state is VerificationState.INSUFFICIENT_EVIDENCE
    assert report.verdicts[0].calibrated_confidence is None
    assert calibrator.calls == []


def test_not_checkable_without_edges_does_not_call_calibrator():
    claim = _claim()
    calibrator = FakeCalibrator(0.99)
    service = _service(
        claims=[claim],
        verifier=FakeVerifier(result=_not_checkable_verdict(claim)),
        calibrator=calibrator,
    )

    report = service.verify(_request())

    assert report.verdicts[0].state is VerificationState.NOT_CHECKABLE
    assert report.verdicts[0].calibrated_confidence is None
    assert calibrator.calls == []


def test_no_calibrator_never_repackages_raw_score_as_confidence():
    claim = _claim()
    reference = EvidenceReference(chunk_id="chunk-1")
    raw = _supported_verdict(claim, reference)
    service = _service(
        claims=[claim],
        selector=FakeSelector(result=[reference]),
        verifier=FakeVerifier(result=raw),
    )

    report = service.verify(_request())

    assert report.verdicts[0].raw_score == 3.2
    assert report.verdicts[0].calibrated_confidence is None


def test_without_calibrator_preserves_a_precalibrated_verdict():
    claim = _claim()
    reference = EvidenceReference(chunk_id="chunk-1")
    raw = _supported_verdict(claim, reference, calibrated_confidence=0.81)
    service = _service(
        claims=[claim],
        selector=FakeSelector(result=[reference]),
        verifier=FakeVerifier(result=raw),
        calibrator_id="upstream-calibrator",
        calibrator_revision="upstream-calibrator-v1",
    )

    assert service.verify(_request()).verdicts[0].calibrated_confidence == 0.81


def test_adapter_list_mutation_cannot_mutate_request_or_nested_metadata():
    claim = _claim()
    chunk = _chunk(metadata={"nested": {"labels": ["a", "b"]}})
    request = _request(chunks=(chunk,))
    original_chunks = request.chunks
    original_metadata = dict(chunk.metadata)
    selector = FakeSelector(result=[], mutate_input=True)
    verifier = FakeVerifier(result=_insufficient_verdict(claim), mutate_input=True)
    service = _service(claims=[claim], selector=selector, verifier=verifier)

    service.verify(request)

    assert request.chunks == original_chunks
    assert len(request.chunks) == 1
    assert dict(request.chunks[0].metadata) == original_metadata
    assert request.chunks[0].metadata["nested"]["labels"] == ("a", "b")


@pytest.mark.parametrize(
    "claims",
    [
        (),
        "not-a-list",
        ["not-a-claim"],
    ],
)
def test_decomposer_rejects_empty_non_list_or_wrong_item_types(claims):
    service = VerifyService(
        decomposer=FakeDecomposer(result=claims),
        selector=FakeSelector(),
        verifier=FakeVerifier(),
        config=VerifyConfig(),
        model_revision="m",
        threshold_version="t",
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(AdapterContractError):
        service.verify(_request())


def test_decomposer_rejects_duplicate_claim_ids():
    claims = [
        _claim("same", "Paris", 0, 5),
        _claim("same", "is", 6, 8),
    ]
    with pytest.raises(AdapterContractError, match="duplicate claim_id"):
        _service(claims=claims).verify(_request())


def test_decomposer_rejects_claim_span_beyond_answer():
    claim = _claim(text="Paris is in France.", start=0, end=99)
    with pytest.raises(AdapterContractError, match="exceeds answer length"):
        _service(claims=[claim]).verify(_request())


def test_decomposer_rejects_claim_text_that_does_not_match_answer_slice():
    claim = _claim(text="London", start=0, end=6)
    with pytest.raises(AdapterContractError, match="does not match"):
        _service(claims=[claim]).verify(_request())


def test_decomposer_rejects_out_of_order_claims():
    answer = "alpha beta"
    claims = [
        _claim("second", "beta", 6, 10),
        _claim("first", "alpha", 0, 5),
    ]
    with pytest.raises(AdapterContractError, match="out of start_char order"):
        _service(claims=claims).verify(_request(answer=answer))


def test_decomposer_rejects_overlapping_claim_spans():
    answer = "abcdef"
    claims = [
        _claim("left", "abcd", 0, 4),
        _claim("right", "def", 3, 6),
    ]
    with pytest.raises(AdapterContractError, match="overlapping"):
        _service(claims=claims).verify(_request(answer=answer))


@pytest.mark.parametrize("references", [(), "not-a-list", ["not-a-reference"]])
def test_selector_rejects_non_list_or_wrong_item_types(references):
    service = _service(selector=FakeSelector(result=references))
    with pytest.raises(AdapterContractError):
        service.verify(_request())


def test_selector_rejects_unknown_chunk_id():
    reference = EvidenceReference(chunk_id="unknown")
    with pytest.raises(AdapterContractError, match="unknown chunk_id"):
        _service(selector=FakeSelector(result=[reference])).verify(_request())


def test_selector_rejects_span_beyond_chunk_text():
    reference = EvidenceReference(chunk_id="chunk-1", start_char=0, end_char=999)
    with pytest.raises(AdapterContractError, match="exceeding chunk"):
        _service(selector=FakeSelector(result=[reference])).verify(_request())


def test_selector_rejects_duplicate_reference_even_if_score_differs():
    references = [
        EvidenceReference(chunk_id="chunk-1", relevance_score=0.8),
        EvidenceReference(chunk_id="chunk-1", relevance_score=0.7),
    ]
    with pytest.raises(AdapterContractError, match="duplicate"):
        _service(selector=FakeSelector(result=references)).verify(_request())


def test_verifier_rejects_wrong_return_type():
    service = _service(verifier=FakeVerifier(result="not-a-verdict"))
    with pytest.raises(AdapterContractError, match="must return a ClaimVerdict"):
        service.verify(_request())


def test_verifier_rejects_verdict_for_a_different_claim():
    other = _claim("other", "Paris", 0, 5)
    verdict = _not_checkable_verdict(other)
    with pytest.raises(AdapterContractError, match="was asked to verify"):
        _service(verifier=FakeVerifier(result=verdict)).verify(_request())


def test_verifier_rejects_fabricated_evidence_reference():
    claim = _claim()
    selected = EvidenceReference(chunk_id="chunk-1")
    fabricated = EvidenceReference(chunk_id="never-supplied")
    verdict = _supported_verdict(claim, fabricated)
    service = _service(
        claims=[claim],
        selector=FakeSelector(result=[selected]),
        verifier=FakeVerifier(result=verdict),
    )

    with pytest.raises(AdapterContractError, match="never supplied"):
        service.verify(_request())


def test_verifier_cannot_expand_allowed_evidence_by_mutating_candidate_list():
    claim = _claim()
    selected = EvidenceReference(chunk_id="chunk-1")
    fabricated_chunk = _chunk("fabricated", "fabricated evidence")
    fabricated_ref = EvidenceReference(chunk_id="fabricated")

    def mutate_then_verify(current_claim, candidates):
        candidates.append(EvidenceCandidate.from_chunk(fabricated_ref, fabricated_chunk))
        return _supported_verdict(current_claim, fabricated_ref)

    service = _service(
        claims=[claim],
        selector=FakeSelector(result=[selected]),
        verifier=FakeVerifier(callback=mutate_then_verify),
    )

    with pytest.raises(AdapterContractError, match="never supplied"):
        service.verify(_request())


def test_verifier_cannot_change_selector_relevance_score_in_audit_edge():
    claim = _claim()
    selected = EvidenceReference(chunk_id="chunk-1", relevance_score=0.8)
    altered = EvidenceReference(chunk_id="chunk-1", relevance_score=0.7)
    service = _service(
        claims=[claim],
        selector=FakeSelector(result=[selected]),
        verifier=FakeVerifier(result=_supported_verdict(claim, altered)),
    )

    with pytest.raises(AdapterContractError, match="never supplied"):
        service.verify(_request())


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_calibrator_illegal_values_are_wrapped_with_original_cause(bad_value):
    claim = _claim()
    reference = EvidenceReference(chunk_id="chunk-1")
    raw_verdict = _supported_verdict(claim, reference)
    calibrator = FakeCalibrator(bad_value)
    service = _service(
        claims=[claim],
        selector=FakeSelector(result=[reference]),
        verifier=FakeVerifier(result=raw_verdict),
        calibrator=calibrator,
    )

    with pytest.raises(AdapterContractError) as captured:
        service.verify(_request())

    assert isinstance(captured.value.__cause__, DomainValidationError)
    assert calibrator.calls[0] is raw_verdict
    assert raw_verdict.calibrated_confidence is None


def test_adapter_raised_exception_propagates_unchanged():
    sentinel = RuntimeError("adapter failed")
    service = VerifyService(
        decomposer=FakeDecomposer(error=sentinel),
        selector=FakeSelector(),
        verifier=FakeVerifier(),
        config=VerifyConfig(),
        model_revision="m",
        threshold_version="t",
        clock=lambda: FIXED_TIME,
    )

    with pytest.raises(RuntimeError) as captured:
        service.verify(_request())

    assert captured.value is sentinel


def test_allow_network_true_is_rejected_at_construction():
    with pytest.raises(ConfigurationError, match="allow_network=True"):
        _service(config=VerifyConfig(allow_network=True))


def test_calibrator_requires_complete_audit_identity():
    calibrator = FakeCalibrator(0.5)
    kwargs = dict(
        decomposer=FakeDecomposer(result=[_claim()]),
        selector=FakeSelector(),
        verifier=FakeVerifier(),
        calibrator=calibrator,
        config=VerifyConfig(),
        model_revision="m",
        threshold_version="t",
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(ConfigurationError, match="calibrator_id"):
        VerifyService(**kwargs)

    kwargs["calibrator_id"] = "cal"
    with pytest.raises(ConfigurationError, match="both be set"):
        VerifyService(**kwargs)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("calibrator_id", "wrong-id"),
        ("calibrator_revision", "wrong-revision"),
        ("threshold_version", "wrong-threshold"),
    ],
)
def test_self_describing_calibrator_rejects_mismatched_wiring(field, wrong_value):
    kwargs = dict(
        decomposer=FakeDecomposer(result=[_claim()]),
        selector=FakeSelector(),
        verifier=FakeVerifier(),
        calibrator=SelfDescribingCalibrator(0.5),
        calibrator_id="declared-calibrator",
        calibrator_revision="declared-revision",
        config=VerifyConfig(),
        model_revision="m",
        threshold_version="declared-threshold",
        clock=lambda: FIXED_TIME,
    )
    kwargs[field] = wrong_value
    with pytest.raises(ConfigurationError, match="does not match"):
        VerifyService(**kwargs)

def test_upstream_precalibrated_metadata_is_allowed_without_local_calibrator():
    service = _service(
        calibrator=None,
        calibrator_id="upstream-cal",
        calibrator_revision="upstream-v1",
    )
    report = service.verify(_request())
    assert report.calibrator_id == "upstream-cal"
    assert report.calibrator_revision == "upstream-v1"


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("decomposer", object()),
        ("selector", object()),
        ("verifier", object()),
        ("calibrator", object()),
        ("config", object()),
        ("clock", "not-callable"),
    ],
)
def test_constructor_rejects_dependencies_that_do_not_implement_contract(
    field_name, bad_value
):
    kwargs = dict(
        decomposer=FakeDecomposer(result=[_claim()]),
        selector=FakeSelector(),
        verifier=FakeVerifier(),
        calibrator=None,
        config=VerifyConfig(),
        model_revision="m",
        threshold_version="t",
        clock=lambda: FIXED_TIME,
    )
    kwargs[field_name] = bad_value

    with pytest.raises(ConfigurationError):
        VerifyService(**kwargs)


def test_verify_rejects_non_verification_request_with_domain_error():
    with pytest.raises(DomainValidationError, match="VerificationRequest"):
        _service().verify({"answer": "Paris is in France."})


@pytest.mark.parametrize("field_name", ["model_revision", "threshold_version"])
def test_empty_run_metadata_is_rejected_at_construction(field_name):
    kwargs = dict(
        decomposer=FakeDecomposer(result=[_claim()]),
        selector=FakeSelector(),
        verifier=FakeVerifier(),
        config=VerifyConfig(),
        model_revision="m",
        threshold_version="t",
        clock=lambda: FIXED_TIME,
    )
    kwargs[field_name] = "   "
    with pytest.raises(ConfigurationError):
        VerifyService(**kwargs)


def test_constructor_has_no_external_or_heal_dependencies():
    parameter_names = set(inspect.signature(VerifyService).parameters)
    forbidden = {
        "retriever",
        "rewriter",
        "repair_policy",
        "stop_policy",
        "audit_store",
        "network_client",
        "llm_client",
        "vector_store",
    }
    assert parameter_names.isdisjoint(forbidden)
