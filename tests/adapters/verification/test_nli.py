import math

import pytest

from groundguard_rag.adapters.decomposition import RuleBasedClaimDecomposer
from groundguard_rag.adapters.evidence_selection import (
    LexicalEvidenceSelector,
    LexicalSelectorConfig,
)
from groundguard_rag.adapters.verification import (
    NliBackend,
    NliBackendContractError,
    NliVerifier,
    NliVerifierConfig,
)
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.config import VerifyConfig
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import (
    AtomicClaim,
    Chunk,
    EvidenceCandidate,
    EvidenceReference,
    LabelScores,
    VerificationRequest,
)


class QueueBackend(NliBackend):
    def __init__(self, outputs=None, error=None):
        self.outputs = list(outputs or [])
        self.error = error
        self.calls = []

    def predict_batch(self, pairs):
        self.calls.append(list(pairs))
        if self.error is not None:
            raise self.error
        return list(self.outputs)


def _claim(text="Paris is in France."):
    return AtomicClaim("claim-1", text, 0, len(text))


def _candidate(chunk_id="chunk-1", text="Paris is a city in France.", score=0.8):
    return EvidenceCandidate(
        reference=EvidenceReference(chunk_id=chunk_id, relevance_score=score),
        text=text,
    )


def _probabilities(supported, contradicted, insufficient):
    return LabelScores(
        supported=supported,
        contradicted=contradicted,
        insufficient=insufficient,
        score_kind="probabilities",
    )


def _config(threshold=0.5):
    return NliVerifierConfig(
        verifier_id="test-nli",
        verifier_revision="revision-1",
        decision_threshold=threshold,
    )


def _verify(outputs, evidence=None, threshold=0.5):
    if evidence is None:
        evidence = [_candidate()]
    backend = QueueBackend(outputs=outputs)
    verdict = NliVerifier(backend, _config(threshold)).verify(_claim(), evidence)
    return verdict, backend


def test_nli_backend_is_abstract():
    with pytest.raises(TypeError):
        NliBackend()


def test_no_evidence_returns_insufficient_without_backend_call():
    backend = QueueBackend()
    verdict = NliVerifier(backend, _config()).verify(_claim(), [])

    assert verdict.state is VerificationState.INSUFFICIENT_EVIDENCE
    assert verdict.evidence_assessments == ()
    assert verdict.raw_score is None
    assert verdict.calibrated_confidence is None
    assert backend.calls == []


def test_backend_receives_evidence_as_premise_and_claim_as_hypothesis():
    claim = _claim("The sky is blue.")
    candidate = _candidate(text="On clear days, the sky appears blue.")
    backend = QueueBackend([_probabilities(0.8, 0.1, 0.1)])

    NliVerifier(backend, _config()).verify(claim, [candidate])

    assert backend.calls == [[(candidate.text, claim.text)]]


def test_logits_are_converted_with_numerically_stable_softmax():
    logits = LabelScores(1000.0, -1000.0, 0.0, "logits")
    verdict, _ = _verify([logits])
    scores = verdict.evidence_assessments[0].label_scores

    assert verdict.state is VerificationState.SUPPORTED
    assert scores.score_kind == "probabilities"
    assert math.isclose(
        scores.supported + scores.contradicted + scores.insufficient,
        1.0,
        abs_tol=1e-12,
    )
    assert scores.supported > 0.999
    assert all(
        math.isfinite(value)
        for value in (scores.supported, scores.contradicted, scores.insufficient)
    )


def test_probability_output_is_recorded_as_full_three_way_edge_score():
    output = _probabilities(0.7, 0.1, 0.2)
    verdict, _ = _verify([output])
    assessment = verdict.evidence_assessments[0]

    assert assessment.label_scores == output
    assert assessment.verifier_id == "test-nli"
    assert assessment.verifier_revision == "revision-1"
    assert assessment.rationale is None


def test_supported_probability_at_threshold_is_supported():
    verdict, _ = _verify([_probabilities(0.6, 0.2, 0.2)], threshold=0.6)
    assert verdict.state is VerificationState.SUPPORTED


def test_contradicted_probability_at_threshold_is_contradicted():
    verdict, _ = _verify([_probabilities(0.2, 0.6, 0.2)], threshold=0.6)
    assert verdict.state is VerificationState.CONTRADICTED


def test_neutral_dominance_is_insufficient_evidence():
    verdict, _ = _verify([_probabilities(0.1, 0.1, 0.8)])
    assert verdict.state is VerificationState.INSUFFICIENT_EVIDENCE


def test_below_threshold_is_insufficient_even_when_entailment_is_argmax():
    verdict, _ = _verify([_probabilities(0.7, 0.1, 0.2)], threshold=0.8)
    assert verdict.state is VerificationState.INSUFFICIENT_EVIDENCE


def test_equal_support_and_contradiction_does_not_choose_arbitrarily():
    verdict, _ = _verify([_probabilities(0.5, 0.5, 0.0)])
    assert verdict.state is VerificationState.INSUFFICIENT_EVIDENCE


def test_supported_and_contradicted_edges_aggregate_to_conflicting():
    evidence = [_candidate("support"), _candidate("contradict")]
    outputs = [_probabilities(0.8, 0.1, 0.1), _probabilities(0.1, 0.8, 0.1)]
    verdict, _ = _verify(outputs, evidence=evidence)

    assert verdict.state is VerificationState.CONFLICTING_EVIDENCE
    assert [edge.state for edge in verdict.evidence_assessments] == [
        VerificationState.SUPPORTED,
        VerificationState.CONTRADICTED,
    ]


def test_supported_plus_insufficient_aggregates_to_supported():
    evidence = [_candidate("support"), _candidate("neutral")]
    outputs = [_probabilities(0.8, 0.1, 0.1), _probabilities(0.1, 0.1, 0.8)]
    verdict, _ = _verify(outputs, evidence=evidence)
    assert verdict.state is VerificationState.SUPPORTED


def test_contradicted_plus_insufficient_aggregates_to_contradicted():
    evidence = [_candidate("contradict"), _candidate("neutral")]
    outputs = [_probabilities(0.1, 0.8, 0.1), _probabilities(0.1, 0.1, 0.8)]
    verdict, _ = _verify(outputs, evidence=evidence)
    assert verdict.state is VerificationState.CONTRADICTED


def test_all_insufficient_edges_aggregate_to_insufficient():
    evidence = [_candidate("a"), _candidate("b")]
    outputs = [_probabilities(0.2, 0.2, 0.6), _probabilities(0.1, 0.2, 0.7)]
    verdict, _ = _verify(outputs, evidence=evidence)
    assert verdict.state is VerificationState.INSUFFICIENT_EVIDENCE


def test_assessments_keep_input_order_and_exact_references():
    evidence = [
        _candidate("first", score=0.9),
        _candidate("second", score=0.7),
    ]
    outputs = [_probabilities(0.8, 0.1, 0.1), _probabilities(0.1, 0.1, 0.8)]
    verdict, _ = _verify(outputs, evidence=evidence)

    assert [edge.reference for edge in verdict.evidence_assessments] == [
        candidate.reference for candidate in evidence
    ]


def test_multiple_evidence_edges_use_one_backend_batch_call():
    evidence = [_candidate("a"), _candidate("b"), _candidate("c")]
    outputs = [
        _probabilities(0.8, 0.1, 0.1),
        _probabilities(0.1, 0.8, 0.1),
        _probabilities(0.1, 0.1, 0.8),
    ]
    _, backend = _verify(outputs, evidence=evidence)

    assert len(backend.calls) == 1
    assert backend.calls[0] == [
        (candidate.text, _claim().text) for candidate in evidence
    ]


def test_model_probabilities_are_not_claim_confidence_or_raw_claim_score():
    verdict, _ = _verify([_probabilities(0.99, 0.005, 0.005)])
    assert verdict.raw_score is None
    assert verdict.calibrated_confidence is None


@pytest.mark.parametrize("bad_output", [None, {"LABEL_0": 0.9}, [0.1, 0.2, 0.7]])
def test_backend_wrong_output_type_is_rejected(bad_output):
    backend = QueueBackend([bad_output])
    verifier = NliVerifier(backend, _config())
    with pytest.raises(NliBackendContractError, match="must return LabelScores"):
        verifier.verify(_claim(), [_candidate()])


def test_unknown_score_kind_is_rejected():
    backend = QueueBackend([LabelScores(1.0, 2.0, 3.0, "mystery-scale")])
    with pytest.raises(NliBackendContractError, match="score_kind"):
        NliVerifier(backend, _config()).verify(_claim(), [_candidate()])


class WrongBatchBackend(NliBackend):
    def __init__(self, result):
        self.result = result

    def predict_batch(self, pairs):
        return self.result


@pytest.mark.parametrize("bad_result", [None, (), "not-a-list", {"scores": []}])
def test_backend_batch_must_return_a_list(bad_result):
    with pytest.raises(NliBackendContractError, match="must return a list"):
        NliVerifier(WrongBatchBackend(bad_result), _config()).verify(
            _claim(), [_candidate()]
        )


@pytest.mark.parametrize(
    "wrong_length",
    [[], [_probabilities(0.8, 0.1, 0.1), _probabilities(0.7, 0.2, 0.1)]],
)
def test_backend_batch_length_must_match_input_pairs(wrong_length):
    with pytest.raises(NliBackendContractError, match="exactly one"):
        NliVerifier(WrongBatchBackend(wrong_length), _config()).verify(
            _claim(), [_candidate()]
        )


def test_backend_exception_propagates_unchanged():
    sentinel = RuntimeError("model failed")
    backend = QueueBackend(error=sentinel)

    with pytest.raises(RuntimeError) as captured:
        NliVerifier(backend, _config()).verify(_claim(), [_candidate()])

    assert captured.value is sentinel


def test_verifier_does_not_mutate_evidence_list_or_candidates():
    evidence = [_candidate()]
    before = list(evidence)
    NliVerifier(QueueBackend([_probabilities(0.8, 0.1, 0.1)]), _config()).verify(
        _claim(), evidence
    )
    assert evidence == before
    assert evidence[0] is before[0]


def test_duplicate_chunk_span_reference_is_rejected_before_backend_call():
    backend = QueueBackend([_probabilities(0.8, 0.1, 0.1)])
    evidence = [_candidate("same", score=0.9), _candidate("same", score=0.7)]

    with pytest.raises(DomainValidationError, match="duplicate"):
        NliVerifier(backend, _config()).verify(_claim(), evidence)

    assert backend.calls == []


def test_wrong_claim_type_is_rejected():
    with pytest.raises(DomainValidationError):
        NliVerifier(QueueBackend(), _config()).verify("not-a-claim", [])


@pytest.mark.parametrize("evidence", [(), "not-a-list", ["not-a-candidate"]])
def test_wrong_evidence_container_or_items_are_rejected(evidence):
    with pytest.raises(DomainValidationError):
        NliVerifier(QueueBackend(), _config()).verify(_claim(), evidence)


@pytest.mark.parametrize("field_name", ["verifier_id", "verifier_revision"])
@pytest.mark.parametrize("bad_value", ["", "   ", None, 42])
def test_config_rejects_invalid_identity_fields(field_name, bad_value):
    kwargs = dict(
        verifier_id="id",
        verifier_revision="revision",
        decision_threshold=0.5,
    )
    kwargs[field_name] = bad_value
    with pytest.raises(ConfigurationError):
        NliVerifierConfig(**kwargs)


@pytest.mark.parametrize(
    "threshold",
    [True, False, 0.49, 1.01, float("nan"), float("inf"), "0.5"],
)
def test_config_rejects_invalid_threshold(threshold):
    with pytest.raises(ConfigurationError):
        _config(threshold)


def test_threshold_boundaries_are_allowed():
    assert _config(0.5).decision_threshold == 0.5
    assert _config(1.0).decision_threshold == 1.0


def test_constructor_rejects_wrong_backend_or_config_type():
    with pytest.raises(ConfigurationError):
        NliVerifier(object(), _config())
    with pytest.raises(ConfigurationError):
        NliVerifier(QueueBackend(), object())


def test_full_local_pipeline_uses_nli_for_decision_not_lexical_score():
    backend = QueueBackend([LabelScores(4.0, -2.0, 0.0, "logits")])
    verifier = NliVerifier(backend, _config(0.6))
    service = VerifyService(
        decomposer=RuleBasedClaimDecomposer(),
        selector=LexicalEvidenceSelector(LexicalSelectorConfig(top_k=1)),
        verifier=verifier,
        calibrator=None,
        config=VerifyConfig(),
        model_revision="nli-core-v1",
        threshold_version="decision-0.6-v1",
        clock=lambda: "2026-09-02T00:00:00+00:00",
    )
    request = VerificationRequest(
        request_id="request-1",
        answer="Paris is the capital of France.",
        chunks=(
            Chunk("candidate", "France has Paris as its capital."),
            Chunk("unrelated", "Tokyo is in Japan."),
        ),
    )

    report = service.verify(request)

    verdict = report.verdicts[0]
    assert verdict.state is VerificationState.SUPPORTED
    assert verdict.calibrated_confidence is None
    assert len(verdict.evidence_assessments) == 1
    assert verdict.evidence_assessments[0].reference.chunk_id == "candidate"
    assert backend.calls == [[
        ("France has Paris as its capital.", "Paris is the capital of France.")
    ]]
