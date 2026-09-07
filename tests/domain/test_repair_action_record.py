import pytest

from groundguard_rag.domain.enums import RepairAction, VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import RepairActionRecord


def _minimal(**overrides):
    fields = dict(
        round=1,
        claim_id="c1",
        action=RepairAction.REWRITE,
        state_before=VerificationState.INSUFFICIENT_EVIDENCE,
        state_after=VerificationState.SUPPORTED,
        claim_id_after="c1",
        claim_text_before="old claim",
        claim_text_after="new claim",
    )
    fields.update(overrides)
    return RepairActionRecord(**fields)


def test_valid_record_constructs():
    record = _minimal(note="rewrote using chunk-2")
    assert record.round == 1
    assert record.note == "rewrote using chunk-2"


def test_remaining_optional_fields_default_to_empty_or_none():
    record = _minimal()
    assert record.claim_id_after == "c1"
    assert record.claim_text_before == "old claim"
    assert record.claim_text_after == "new claim"
    assert record.committed is True
    assert record.note is None
    assert record.evidence_ids_added == ()
    assert record.evidence_ids_removed == ()
    assert record.answer_hash_before is None
    assert record.answer_hash_after is None
    assert record.calibrated_confidence_before is None
    assert record.calibrated_confidence_after is None
    assert record.elapsed_ms is None
    assert record.estimated_cost is None
    assert record.failure_or_stop_reason is None


def test_fully_populated_record_constructs():
    record = _minimal(
        evidence_ids_added=("chunk-2",),
        evidence_ids_removed=("chunk-1",),
        answer_hash_before="hash-before",
        answer_hash_after="hash-after",
        calibrated_confidence_before=0.3,
        calibrated_confidence_after=0.8,
        elapsed_ms=125.5,
        estimated_cost=0.001,
        failure_or_stop_reason=None,
    )
    assert record.evidence_ids_added == ("chunk-2",)
    assert record.evidence_ids_removed == ("chunk-1",)
    assert record.answer_hash_before == "hash-before"
    assert record.calibrated_confidence_before == 0.3
    assert record.elapsed_ms == 125.5


def test_evidence_ids_lists_are_coerced_to_tuples():
    record = _minimal(evidence_ids_added=["chunk-2", "chunk-3"])
    assert isinstance(record.evidence_ids_added, tuple)
    assert record.evidence_ids_added == ("chunk-2", "chunk-3")


def test_evidence_ids_added_rejects_empty_string_entries():
    with pytest.raises(DomainValidationError):
        _minimal(evidence_ids_added=("",))


def test_evidence_ids_removed_rejects_empty_string_entries():
    with pytest.raises(DomainValidationError):
        _minimal(evidence_ids_removed=("",))


@pytest.mark.parametrize(
    "field_name", ["answer_hash_before", "answer_hash_after", "note", "failure_or_stop_reason"]
)
def test_string_or_none_fields_reject_non_string_non_none(field_name):
    with pytest.raises(DomainValidationError):
        _minimal(**{field_name: 42})


@pytest.mark.parametrize(
    "field_name", ["calibrated_confidence_before", "calibrated_confidence_after"]
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_calibrated_confidence_fields_reject_non_finite(field_name, bad_value):
    with pytest.raises(DomainValidationError):
        _minimal(**{field_name: bad_value})


@pytest.mark.parametrize(
    "field_name", ["calibrated_confidence_before", "calibrated_confidence_after"]
)
@pytest.mark.parametrize("bad_value", [-0.01, 1.01, -5.0, 3.0])
def test_calibrated_confidence_fields_reject_out_of_unit_range(field_name, bad_value):
    # Stage 0.3: these are the *calibrated* confidence, not a raw score --
    # unlike ClaimVerdict.raw_score, they are bounded to [0, 1].
    with pytest.raises(DomainValidationError):
        _minimal(**{field_name: bad_value})


@pytest.mark.parametrize(
    "field_name", ["calibrated_confidence_before", "calibrated_confidence_after"]
)
@pytest.mark.parametrize("boundary_value", [0.0, 1.0])
def test_calibrated_confidence_fields_accept_unit_interval_boundaries(field_name, boundary_value):
    record = _minimal(**{field_name: boundary_value})
    assert getattr(record, field_name) == boundary_value


@pytest.mark.parametrize(
    "field_name", ["calibrated_confidence_before", "calibrated_confidence_after"]
)
def test_calibrated_confidence_fields_reject_bool(field_name):
    # bool is a subclass of int/float-comparable in Python -- True must not
    # silently pass as a valid confidence of 1.0.
    with pytest.raises(DomainValidationError):
        _minimal(**{field_name: True})


@pytest.mark.parametrize("field_name", ["elapsed_ms", "estimated_cost"])
def test_nonneg_fields_reject_negative_values(field_name):
    with pytest.raises(DomainValidationError):
        _minimal(**{field_name: -0.01})


@pytest.mark.parametrize("field_name", ["elapsed_ms", "estimated_cost"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonneg_fields_reject_non_finite(field_name, bad_value):
    with pytest.raises(DomainValidationError):
        _minimal(**{field_name: bad_value})


@pytest.mark.parametrize("round_number", [0, -1, True, False, 1.5, "1"])
def test_round_rejects_non_positive_bool_and_non_int(round_number):
    with pytest.raises(DomainValidationError):
        _minimal(round=round_number)


def test_empty_claim_id_rejected():
    with pytest.raises(DomainValidationError):
        _minimal(claim_id="")


def test_non_enum_action_rejected():
    with pytest.raises(DomainValidationError):
        _minimal(action="")


@pytest.mark.parametrize("field_name", ["state_before", "state_after"])
def test_non_enum_state_rejected(field_name):
    with pytest.raises(DomainValidationError):
        _minimal(**{field_name: "SUPPORTED"})


def test_to_dict_shape():
    record = _minimal()
    assert record.to_dict() == {
        "round": 1,
        "claim_id": "c1",
        "action": "rewrite",
        "state_before": "INSUFFICIENT_EVIDENCE",
        "state_after": "SUPPORTED",
        "claim_id_after": "c1",
        "claim_text_before": "old claim",
        "claim_text_after": "new claim",
        "committed": True,
        "note": None,
        "evidence_ids_added": [],
        "evidence_ids_removed": [],
        "answer_hash_before": None,
        "answer_hash_after": None,
        "calibrated_confidence_before": None,
        "calibrated_confidence_after": None,
        "elapsed_ms": None,
        "estimated_cost": None,
        "failure_or_stop_reason": None,
    }


def test_committed_delete_is_the_only_action_allowed_without_after_values():
    record = _minimal(
        action=RepairAction.DELETE,
        state_after=None,
        claim_id_after=None,
        claim_text_after=None,
    )
    assert record.committed is True
    assert record.state_after is None


@pytest.mark.parametrize("committed", [False, 0, 1, "yes"])
def test_delete_null_after_values_require_literal_true(committed):
    with pytest.raises(DomainValidationError):
        _minimal(
            action=RepairAction.DELETE,
            state_after=None,
            claim_id_after=None,
            claim_text_after=None,
            committed=committed,
        )


def test_uncommitted_action_keeps_source_lineage():
    record = _minimal(committed=False, failure_or_stop_reason="min_improvement")
    assert record.claim_id_after == record.claim_id
