import pytest

from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import EvidenceAssessment, EvidenceReference, LabelScores


def _ref():
    return EvidenceReference(chunk_id="chunk-1", relevance_score=0.9)


def _scores(supported=2.1, contradicted=-1.5, insufficient=-0.3, score_kind="logits"):
    return LabelScores(
        supported=supported,
        contradicted=contradicted,
        insufficient=insufficient,
        score_kind=score_kind,
    )


@pytest.mark.parametrize(
    "state",
    [
        VerificationState.SUPPORTED,
        VerificationState.CONTRADICTED,
        VerificationState.INSUFFICIENT_EVIDENCE,
    ],
)
def test_edge_level_states_construct(state):
    assessment = EvidenceAssessment(
        reference=_ref(),
        label_scores=_scores(),
        state=state,
        rationale="because",
        verifier_id="verifier-a",
        verifier_revision="v1",
    )
    assert assessment.state is state


@pytest.mark.parametrize(
    "state", [VerificationState.NOT_CHECKABLE, VerificationState.CONFLICTING_EVIDENCE]
)
def test_claim_level_only_states_rejected(state):
    # A single edge cannot be NOT_CHECKABLE (that classification happens
    # before evidence is sought) or CONFLICTING_EVIDENCE (which only
    # emerges from comparing multiple edges).
    with pytest.raises(DomainValidationError):
        EvidenceAssessment(
            reference=_ref(),
            label_scores=_scores(),
            state=state,
            rationale=None,
            verifier_id="verifier-a",
            verifier_revision="v1",
        )


def test_non_enum_state_rejected():
    with pytest.raises(DomainValidationError):
        EvidenceAssessment(
            reference=_ref(),
            label_scores=_scores(),
            state="SUPPORTED",
            rationale=None,
            verifier_id="verifier-a",
            verifier_revision="v1",
        )


def test_reference_must_be_evidence_reference_instance():
    with pytest.raises(DomainValidationError):
        EvidenceAssessment(
            reference="chunk-1",
            label_scores=_scores(),
            state=VerificationState.SUPPORTED,
            rationale=None,
            verifier_id="verifier-a",
            verifier_revision="v1",
        )


def test_label_scores_must_be_label_scores_instance():
    with pytest.raises(DomainValidationError):
        EvidenceAssessment(
            reference=_ref(),
            label_scores={"supported": 1.0, "contradicted": 0.0, "insufficient": 0.0},
            state=VerificationState.SUPPORTED,
            rationale=None,
            verifier_id="verifier-a",
            verifier_revision="v1",
        )


def test_rationale_must_be_string_or_none():
    with pytest.raises(DomainValidationError):
        EvidenceAssessment(
            reference=_ref(),
            label_scores=_scores(),
            state=VerificationState.SUPPORTED,
            rationale=123,
            verifier_id="verifier-a",
            verifier_revision="v1",
        )


@pytest.mark.parametrize("field_name", ["verifier_id", "verifier_revision"])
def test_empty_verifier_identity_fields_rejected(field_name):
    kwargs = dict(
        reference=_ref(),
        label_scores=_scores(),
        state=VerificationState.SUPPORTED,
        rationale=None,
        verifier_id="verifier-a",
        verifier_revision="v1",
    )
    kwargs[field_name] = ""
    with pytest.raises(DomainValidationError):
        EvidenceAssessment(**kwargs)


def test_edge_key_composition():
    ref = EvidenceReference(chunk_id="chunk-1", start_char=0, end_char=5)
    assessment = EvidenceAssessment(
        reference=ref,
        label_scores=_scores(),
        state=VerificationState.SUPPORTED,
        rationale=None,
        verifier_id="verifier-a",
        verifier_revision="v1",
    )
    assert assessment.edge_key() == ("chunk-1", 0, 5, "verifier-a", "v1")


def test_to_dict_shape():
    assessment = EvidenceAssessment(
        reference=_ref(),
        label_scores=_scores(supported=2.1, contradicted=-1.5, insufficient=-0.3, score_kind="logits"),
        state=VerificationState.SUPPORTED,
        rationale="because",
        verifier_id="verifier-a",
        verifier_revision="v1",
    )
    assert assessment.to_dict() == {
        "reference": {
            "chunk_id": "chunk-1",
            "start_char": None,
            "end_char": None,
            "relevance_score": 0.9,
        },
        "label_scores": {
            "supported": 2.1,
            "contradicted": -1.5,
            "insufficient": -0.3,
            "score_kind": "logits",
        },
        "state": "SUPPORTED",
        "rationale": "because",
        "verifier_id": "verifier-a",
        "verifier_revision": "v1",
    }
