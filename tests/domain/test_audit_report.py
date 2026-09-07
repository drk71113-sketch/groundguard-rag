import pytest

from groundguard_rag.domain.enums import RepairAction, RunMode, VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import (
    SCHEMA_VERSION,
    AtomicClaim,
    AuditReport,
    ClaimVerdict,
    RepairActionRecord,
    RunMetrics,
)


def _minimal_report(**overrides):
    fields = dict(
        schema_version=SCHEMA_VERSION,
        request_id="req-1",
        input_hash="deadbeef",
        model_revision="verifier-v0",
        threshold_version="thresholds-v0",
        created_at="2026-09-01T00:00:00+00:00",
        run_mode=RunMode.VERIFY,
        verdicts=(),
    )
    fields.update(overrides)
    if fields["run_mode"] is RunMode.HEAL:
        if not fields["verdicts"]:
            fields["verdicts"] = (_verdict("c1"),)
        fields.setdefault("initial_verdicts", fields["verdicts"])
    return AuditReport(**fields)


def _action(round_number=1, action=RepairAction.REWRITE, claim_id="c1"):
    return RepairActionRecord(
        round=round_number,
        claim_id=claim_id,
        action=action,
        state_before=VerificationState.INSUFFICIENT_EVIDENCE,
        state_after=VerificationState.SUPPORTED,
        claim_id_after=claim_id,
        claim_text_before="old claim",
        claim_text_after="new claim",
    )


def _verdict(claim_id="c1"):
    return ClaimVerdict(
        claim=AtomicClaim(claim_id=claim_id, text="x", start_char=0, end_char=1),
        state=VerificationState.INSUFFICIENT_EVIDENCE,
        evidence_assessments=[],
    )


def test_minimal_verify_only_report_constructs():
    report = _minimal_report()
    assert report.repair_rounds == 0
    assert report.repair_actions == ()
    assert report.stop_reason is None
    assert report.run_mode is RunMode.VERIFY
    assert isinstance(report.metrics, RunMetrics)
    assert report.metrics.total_latency_ms is None
    assert report.calibrator_id is None
    assert report.calibrator_revision is None


def test_calibrator_metadata_must_be_a_complete_nonempty_pair():
    with pytest.raises(DomainValidationError, match="both be set"):
        _minimal_report(calibrator_id="cal", calibrator_revision=None)
    with pytest.raises(DomainValidationError, match="non-empty"):
        _minimal_report(calibrator_id="cal", calibrator_revision="   ")


def test_calibrated_verdict_requires_traceable_calibrator_metadata():
    base = _verdict("c1")
    calibrated = ClaimVerdict(
        claim=base.claim,
        state=base.state,
        evidence_assessments=base.evidence_assessments,
        calibrated_confidence=0.8,
    )
    with pytest.raises(DomainValidationError, match="requires calibrator metadata"):
        _minimal_report(verdicts=(calibrated,))

    report = _minimal_report(
        verdicts=(calibrated,),
        calibrator_id="claim-platt",
        calibrator_revision="sha256:revision",
    )
    assert report.calibrator_id == "claim-platt"
    assert report.to_dict()["calibrator_revision"] == "sha256:revision"


def test_heal_run_mode_with_no_repair_activity_constructs():
    # HEAL explicitly allows repair_rounds == 0 -- e.g. every claim already
    # passed and nothing needed repairing.
    report = _minimal_report(run_mode=RunMode.HEAL, stop_reason="nothing_to_repair")
    assert report.run_mode is RunMode.HEAL
    assert report.repair_rounds == 0


def test_heal_run_mode_with_repair_activity_constructs():
    report = _minimal_report(
        run_mode=RunMode.HEAL,
        verdicts=(_verdict("c1"),),
        repair_rounds=1,
        repair_actions=(_action(round_number=1),),
        stop_reason="max_rounds_exhausted",
    )
    assert report.repair_rounds == 1
    assert len(report.repair_actions) == 1


def test_non_enum_run_mode_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(run_mode="VERIFY")


# --- RunMode <-> repair-field invariant (core_requirements #8.1) ----------


def test_verify_with_nonzero_repair_rounds_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(run_mode=RunMode.VERIFY, repair_rounds=1)


def test_verify_with_repair_actions_rejected():
    # This is the exact case the stage 0.3 review flagged: a previous round
    # of tests treated VERIFY + a repair action as legitimate input.
    with pytest.raises(DomainValidationError):
        _minimal_report(
            run_mode=RunMode.VERIFY,
            verdicts=(_verdict("c1"),),
            repair_rounds=1,
            repair_actions=(_action(round_number=1),),
        )


def test_verify_with_stop_reason_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(run_mode=RunMode.VERIFY, stop_reason="some reason")


def test_verify_with_all_heal_fields_at_default_accepted():
    report = _minimal_report(run_mode=RunMode.VERIFY)
    assert report.repair_rounds == 0
    assert report.repair_actions == ()
    assert report.stop_reason is None


def test_heal_with_none_stop_reason_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(run_mode=RunMode.HEAL, stop_reason=None)


def test_heal_with_empty_stop_reason_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(run_mode=RunMode.HEAL, stop_reason="")


def test_heal_with_whitespace_only_stop_reason_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(run_mode=RunMode.HEAL, stop_reason="   ")


def test_heal_repair_action_round_still_bounded_by_repair_rounds():
    with pytest.raises(DomainValidationError):
        _minimal_report(
            run_mode=RunMode.HEAL,
            verdicts=(_verdict("c1"),),
            repair_rounds=1,
            repair_actions=(_action(round_number=2),),
            stop_reason="max_rounds_exhausted",
        )


def test_repair_action_round_exactly_at_repair_rounds_accepted():
    report = _minimal_report(
        run_mode=RunMode.HEAL,
        verdicts=(_verdict("c1"),),
        repair_rounds=1,
        repair_actions=(_action(round_number=1),),
        stop_reason="max_rounds",
    )
    assert len(report.repair_actions) == 1
    assert report.stop_reason == "max_rounds"


def test_repair_action_round_beyond_repair_rounds_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(
            run_mode=RunMode.HEAL,
            verdicts=(_verdict("c1"),),
            repair_rounds=2,
            repair_actions=(_action(round_number=3),),
            stop_reason="x",
        )


def test_multiple_repair_actions_across_rounds_accepted():
    report = _minimal_report(
        run_mode=RunMode.HEAL,
        verdicts=(_verdict("c1"),),
        repair_rounds=2,
        repair_actions=(
            _action(round_number=1),
            _action(round_number=2, action=RepairAction.RETRIEVE),
        ),
        stop_reason="x",
    )
    assert len(report.repair_actions) == 2


def test_created_at_accepts_trailing_z():
    report = _minimal_report(created_at="2026-09-01T00:00:00Z")
    assert report.created_at == "2026-09-01T00:00:00Z"


def test_created_at_accepts_explicit_positive_offset():
    report = _minimal_report(created_at="2026-09-01T08:00:00+08:00")
    assert report.created_at == "2026-09-01T08:00:00+08:00"


def test_invalid_created_at_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(created_at="not-a-timestamp")


def test_created_at_without_timezone_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(created_at="2026-09-01T00:00:00")


@pytest.mark.parametrize(
    "schema_version", ["1.0", "v1.1.0", "", "1.1.0.0", "1.0.0", "2.0.0"]
)
def test_schema_version_not_exactly_equal_to_current_rejected(schema_version):
    with pytest.raises(DomainValidationError):
        _minimal_report(schema_version=schema_version)


def test_schema_version_matching_constant_accepted():
    report = _minimal_report(schema_version=SCHEMA_VERSION)
    assert report.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize(
    "field_name",
    ["request_id", "input_hash", "model_revision", "threshold_version"],
)
def test_empty_required_string_fields_rejected(field_name):
    with pytest.raises(DomainValidationError):
        _minimal_report(**{field_name: ""})


def test_stop_reason_must_be_string_or_none():
    with pytest.raises(DomainValidationError):
        _minimal_report(run_mode=RunMode.HEAL, stop_reason=42)


@pytest.mark.parametrize("repair_rounds", [True, False, 1.5, "1"])
def test_repair_rounds_rejects_bool_and_non_int(repair_rounds):
    with pytest.raises(DomainValidationError):
        _minimal_report(repair_rounds=repair_rounds)


def test_negative_repair_rounds_rejected():
    with pytest.raises(DomainValidationError):
        _minimal_report(repair_rounds=-1)


def test_repair_actions_must_be_repair_action_record_instances():
    with pytest.raises(DomainValidationError):
        _minimal_report(
            run_mode=RunMode.HEAL,
            repair_rounds=1,
            repair_actions=("rewrite",),
            stop_reason="x",
        )


def test_repair_actions_list_is_coerced_to_tuple():
    report = _minimal_report(
        run_mode=RunMode.HEAL,
        verdicts=(_verdict("c1"),),
        repair_rounds=1,
        repair_actions=[_action()],
        stop_reason="x",
    )
    assert isinstance(report.repair_actions, tuple)


def test_verdicts_list_is_coerced_to_tuple():
    report = _minimal_report(verdicts=[])
    assert isinstance(report.verdicts, tuple)


def test_verdicts_must_contain_only_claim_verdict_instances():
    with pytest.raises(DomainValidationError):
        _minimal_report(verdicts=[{"state": "SUPPORTED"}])


def test_verdicts_containing_atomic_claim_instead_of_claim_verdict_rejected():
    claim = AtomicClaim(claim_id="c1", text="x", start_char=0, end_char=1)
    with pytest.raises(DomainValidationError):
        _minimal_report(verdicts=[claim])


# --- audit integrity: duplicate claim_id / unknown repair-action claim ----


def test_duplicate_claim_id_across_verdicts_rejected_at_construction():
    with pytest.raises(DomainValidationError):
        _minimal_report(verdicts=(_verdict("c1"), _verdict("c1")))


def test_distinct_claim_ids_across_verdicts_accepted():
    report = _minimal_report(verdicts=(_verdict("c1"), _verdict("c2")))
    assert len(report.verdicts) == 2


def test_repair_action_referencing_unknown_claim_id_rejected_at_construction():
    with pytest.raises(DomainValidationError):
        _minimal_report(
            run_mode=RunMode.HEAL,
            verdicts=(_verdict("c1"),),
            repair_rounds=1,
            repair_actions=(_action(round_number=1, claim_id="does-not-exist"),),
            stop_reason="x",
        )


def test_repair_action_referencing_existing_claim_id_accepted():
    report = _minimal_report(
        run_mode=RunMode.HEAL,
        verdicts=(_verdict("c1"),),
        repair_rounds=1,
        repair_actions=(_action(round_number=1, claim_id="c1"),),
        stop_reason="x",
    )
    assert report.repair_actions[0].claim_id == "c1"


def test_metrics_must_be_run_metrics_instance():
    with pytest.raises(DomainValidationError):
        _minimal_report(metrics={"total_latency_ms": 1.0})


def test_metrics_defaults_to_empty_run_metrics():
    report = _minimal_report()
    assert report.metrics.to_dict() == RunMetrics().to_dict()


def test_metrics_can_be_supplied():
    metrics = RunMetrics(total_latency_ms=42.0, model_call_count=2)
    report = _minimal_report(metrics=metrics)
    assert report.metrics.total_latency_ms == 42.0
    assert report.metrics.model_call_count == 2


def test_to_dict_contains_all_required_top_level_keys():
    report = _minimal_report()
    payload = report.to_dict()
    assert set(payload.keys()) == {
        "schema_version",
        "request_id",
        "input_hash",
        "model_revision",
        "threshold_version",
        "created_at",
        "run_mode",
        "verdicts",
        "initial_verdicts",
        "calibrator_id",
        "calibrator_revision",
        "heal_progress_calibrator_id",
        "heal_progress_calibrator_revision",
        "repair_rounds",
        "repair_actions",
        "stop_reason",
        "metrics",
    }


def test_to_dict_serializes_run_mode_and_metrics():
    report = _minimal_report(
        run_mode=RunMode.HEAL, stop_reason="x", metrics=RunMetrics(model_call_count=5)
    )
    payload = report.to_dict()
    assert payload["run_mode"] == "HEAL"
    assert payload["metrics"]["model_call_count"] == 5


def test_to_dict_serializes_repair_actions():
    report = _minimal_report(
        run_mode=RunMode.HEAL,
        verdicts=(_verdict("c1"),),
        repair_rounds=1,
        repair_actions=(_action(),),
        stop_reason="x",
    )
    payload = report.to_dict()
    assert payload["repair_actions"] == [
        {
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
    ]
