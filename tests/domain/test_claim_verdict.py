import pytest

from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import (
    AtomicClaim,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceReference,
    LabelScores,
)


def _claim():
    return AtomicClaim(claim_id="c1", text="Paris is in France.", start_char=0, end_char=20)


def _scores():
    return LabelScores(supported=2.0, contradicted=-2.0, insufficient=-0.5, score_kind="logits")


def _assessment(state=VerificationState.SUPPORTED, chunk_id="chunk-1", verifier_id="stub-verifier"):
    return EvidenceAssessment(
        reference=EvidenceReference(chunk_id=chunk_id, relevance_score=0.9),
        label_scores=_scores(),
        state=state,
        rationale="says so directly",
        verifier_id=verifier_id,
        verifier_revision="v0",
    )


def test_supported_verdict_constructs():
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.SUPPORTED,
        evidence_assessments=[_assessment(VerificationState.SUPPORTED)],
        raw_score=12.34,
        calibrated_confidence=0.87,
        rationale="chunk-1 states this directly",
    )
    assert verdict.state is VerificationState.SUPPORTED
    assert verdict.calibrated_confidence == 0.87
    # raw_score is verifier-specific and deliberately unbounded.
    assert verdict.raw_score == 12.34
    assert len(verdict.evidence_assessments) == 1


def test_contradicted_verdict_constructs():
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.CONTRADICTED,
        evidence_assessments=[_assessment(VerificationState.CONTRADICTED)],
    )
    assert verdict.state is VerificationState.CONTRADICTED


def test_insufficient_evidence_verdict_constructs_with_no_evidence():
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.INSUFFICIENT_EVIDENCE,
        evidence_assessments=[],
    )
    assert verdict.evidence_assessments == ()


def test_insufficient_evidence_verdict_constructs_with_all_insufficient_edges():
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.INSUFFICIENT_EVIDENCE,
        evidence_assessments=[
            _assessment(VerificationState.INSUFFICIENT_EVIDENCE, chunk_id="chunk-1"),
            _assessment(VerificationState.INSUFFICIENT_EVIDENCE, chunk_id="chunk-2"),
        ],
    )
    assert len(verdict.evidence_assessments) == 2


def test_insufficient_evidence_verdict_with_a_supported_edge_rejected():
    # A supporting edge present means the claim isn't actually
    # "insufficient evidence" -- that combination should be SUPPORTED.
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.INSUFFICIENT_EVIDENCE,
            evidence_assessments=[_assessment(VerificationState.SUPPORTED)],
        )


def test_conflicting_evidence_verdict_requires_supported_and_contradicted():
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.CONFLICTING_EVIDENCE,
        evidence_assessments=[
            _assessment(VerificationState.SUPPORTED, chunk_id="chunk-1"),
            _assessment(VerificationState.CONTRADICTED, chunk_id="chunk-2"),
        ],
    )
    assert len(verdict.evidence_assessments) == 2


def test_conflicting_evidence_with_two_supported_edges_rejected():
    # Two SUPPORTED edges don't conflict with each other -- this must not
    # pass as CONFLICTING_EVIDENCE just because there are >= 2 edges.
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.CONFLICTING_EVIDENCE,
            evidence_assessments=[
                _assessment(VerificationState.SUPPORTED, chunk_id="chunk-1"),
                _assessment(VerificationState.SUPPORTED, chunk_id="chunk-2"),
            ],
        )


def test_conflicting_evidence_with_only_one_assessment_rejected():
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.CONFLICTING_EVIDENCE,
            evidence_assessments=[_assessment(VerificationState.SUPPORTED)],
        )


@pytest.mark.parametrize("state", [VerificationState.SUPPORTED, VerificationState.CONTRADICTED])
def test_supported_or_contradicted_verdict_with_no_evidence_rejected(state):
    with pytest.raises(DomainValidationError):
        ClaimVerdict(claim=_claim(), state=state, evidence_assessments=[])


def test_supported_verdict_with_only_contradicted_edge_rejected():
    # Direct regression test for stage 0.2 item 2: SUPPORTED must have at
    # least one SUPPORTED edge -- an all-CONTRADICTED edge set cannot back
    # a SUPPORTED verdict.
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.SUPPORTED,
            evidence_assessments=[_assessment(VerificationState.CONTRADICTED)],
        )


def test_supported_verdict_with_a_contradicted_edge_present_rejected():
    # Even with a SUPPORTED edge present, a coexisting CONTRADICTED edge
    # means the claim's true state is CONFLICTING_EVIDENCE, not SUPPORTED.
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.SUPPORTED,
            evidence_assessments=[
                _assessment(VerificationState.SUPPORTED, chunk_id="chunk-1"),
                _assessment(VerificationState.CONTRADICTED, chunk_id="chunk-2"),
            ],
        )


def test_contradicted_verdict_with_only_supported_edge_rejected():
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.CONTRADICTED,
            evidence_assessments=[_assessment(VerificationState.SUPPORTED)],
        )


def test_supported_verdict_may_coexist_with_insufficient_evidence_edges():
    # SUPPORTED only forbids a coexisting CONTRADICTED edge; a mix of
    # SUPPORTED + INSUFFICIENT_EVIDENCE edges is fine.
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.SUPPORTED,
        evidence_assessments=[
            _assessment(VerificationState.SUPPORTED, chunk_id="chunk-1"),
            _assessment(VerificationState.INSUFFICIENT_EVIDENCE, chunk_id="chunk-2"),
        ],
    )
    assert len(verdict.evidence_assessments) == 2


def test_duplicate_evidence_edge_rejected():
    # Same chunk_id + span + verifier_id + verifier_revision twice.
    duplicate = _assessment(VerificationState.SUPPORTED, chunk_id="chunk-1")
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.SUPPORTED,
            evidence_assessments=[duplicate, duplicate],
        )


def test_same_chunk_different_verifier_is_not_a_duplicate():
    # The uniqueness key includes verifier_id, so the same chunk assessed
    # by two different verifiers is legitimately two distinct edges.
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.SUPPORTED,
        evidence_assessments=[
            _assessment(VerificationState.SUPPORTED, chunk_id="chunk-1", verifier_id="verifier-a"),
            _assessment(VerificationState.SUPPORTED, chunk_id="chunk-1", verifier_id="verifier-b"),
        ],
    )
    assert len(verdict.evidence_assessments) == 2


def test_evidence_assessments_list_is_coerced_to_tuple():
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.SUPPORTED,
        evidence_assessments=[_assessment()],
    )
    assert isinstance(verdict.evidence_assessments, tuple)


def test_evidence_assessments_must_be_evidence_assessment_instances():
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.SUPPORTED,
            evidence_assessments=[EvidenceReference(chunk_id="chunk-1")],
        )


def test_claim_must_be_atomic_claim_instance():
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim={"claim_id": "c1"},
            state=VerificationState.INSUFFICIENT_EVIDENCE,
            evidence_assessments=[],
        )


def test_not_checkable_with_no_evidence_constructs():
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.NOT_CHECKABLE,
        evidence_assessments=[],
    )
    assert verdict.evidence_assessments == ()


def test_not_checkable_with_evidence_rejected():
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.NOT_CHECKABLE,
            evidence_assessments=[_assessment()],
        )


def test_non_enum_state_rejected():
    with pytest.raises(DomainValidationError):
        ClaimVerdict(claim=_claim(), state="SUPPORTED", evidence_assessments=[])


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_calibrated_confidence_out_of_range_rejected(confidence):
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.INSUFFICIENT_EVIDENCE,
            evidence_assessments=[],
            calibrated_confidence=confidence,
        )


def test_calibrated_confidence_nan_rejected():
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.INSUFFICIENT_EVIDENCE,
            evidence_assessments=[],
            calibrated_confidence=float("nan"),
        )


def test_raw_score_infinite_rejected():
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.INSUFFICIENT_EVIDENCE,
            evidence_assessments=[],
            raw_score=float("inf"),
        )


def test_rationale_must_be_string_or_none():
    with pytest.raises(DomainValidationError):
        ClaimVerdict(
            claim=_claim(),
            state=VerificationState.INSUFFICIENT_EVIDENCE,
            evidence_assessments=[],
            rationale=42,
        )


def test_all_five_verification_states_are_constructible_on_a_claim_verdict():
    # Direct regression test for "确保五种状态在模型层都实际可表达": AtomicClaim
    # represents a decomposed answer unit, not an assertion that it is
    # checkable -- whether it is checkable is exactly what this
    # ClaimVerdict.state records (see AtomicClaim's docstring). A
    # greeting/opinion span is a perfectly ordinary AtomicClaim that simply
    # resolves to NOT_CHECKABLE once examined.
    greeting = AtomicClaim(claim_id="c0", text="Hi there!", start_char=0, end_char=9)

    supported = ClaimVerdict(
        claim=_claim(), state=VerificationState.SUPPORTED,
        evidence_assessments=[_assessment(VerificationState.SUPPORTED)],
    )
    contradicted = ClaimVerdict(
        claim=_claim(), state=VerificationState.CONTRADICTED,
        evidence_assessments=[_assessment(VerificationState.CONTRADICTED)],
    )
    insufficient = ClaimVerdict(
        claim=_claim(), state=VerificationState.INSUFFICIENT_EVIDENCE,
        evidence_assessments=[],
    )
    conflicting = ClaimVerdict(
        claim=_claim(), state=VerificationState.CONFLICTING_EVIDENCE,
        evidence_assessments=[
            _assessment(VerificationState.SUPPORTED, chunk_id="chunk-1"),
            _assessment(VerificationState.CONTRADICTED, chunk_id="chunk-2"),
        ],
    )
    not_checkable = ClaimVerdict(
        claim=greeting, state=VerificationState.NOT_CHECKABLE, evidence_assessments=[]
    )

    constructed_states = {
        supported.state, contradicted.state, insufficient.state,
        conflicting.state, not_checkable.state,
    }
    assert constructed_states == set(VerificationState)


def test_to_dict_shape():
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.INSUFFICIENT_EVIDENCE,
        evidence_assessments=[],
        raw_score=None,
        calibrated_confidence=None,
        rationale=None,
    )
    payload = verdict.to_dict()
    assert payload["state"] == "INSUFFICIENT_EVIDENCE"
    assert payload["evidence_assessments"] == []
    assert payload["claim"]["claim_id"] == "c1"


def test_to_dict_includes_evidence_assessment_details():
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.SUPPORTED,
        evidence_assessments=[_assessment(VerificationState.SUPPORTED)],
    )
    payload = verdict.to_dict()
    assessment_payload = payload["evidence_assessments"][0]
    assert assessment_payload["state"] == "SUPPORTED"
    assert assessment_payload["verifier_id"] == "stub-verifier"
    assert assessment_payload["reference"]["chunk_id"] == "chunk-1"
    assert assessment_payload["label_scores"]["score_kind"] == "logits"
