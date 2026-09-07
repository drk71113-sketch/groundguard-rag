import json

import pytest

from groundguard_rag.domain.enums import RepairAction, RunMode, VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import (
    SCHEMA_VERSION,
    AtomicClaim,
    AuditReport,
    ClaimVerdict,
    EvidenceAssessment,
    EvidenceReference,
    LabelScores,
    RepairActionRecord,
    RunMetrics,
)
from groundguard_rag.schema.semantic_validation import validate_audit_report_semantics


def _label_scores(**overrides):
    fields = dict(supported=2.0, contradicted=-2.0, insufficient=-0.5, score_kind="logits")
    fields.update(overrides)
    return LabelScores(**fields)


def _assessment(state=VerificationState.SUPPORTED, chunk_id="chunk-1", verifier_id="stub-verifier"):
    return EvidenceAssessment(
        reference=EvidenceReference(chunk_id=chunk_id, start_char=0, end_char=10, relevance_score=0.9),
        label_scores=_label_scores(),
        state=state,
        rationale="chunk states this directly",
        verifier_id=verifier_id,
        verifier_revision="v0",
    )


def _claim(claim_id="c1"):
    return AtomicClaim(claim_id=claim_id, text="Paris is in France.", start_char=0, end_char=20)


def _sample_report(**overrides) -> AuditReport:
    verdict = ClaimVerdict(
        claim=_claim(),
        state=VerificationState.SUPPORTED,
        evidence_assessments=[_assessment(VerificationState.SUPPORTED)],
        raw_score=12.3,
        calibrated_confidence=0.91,
        rationale="supported by chunk-1",
    )
    fields = dict(
        schema_version=SCHEMA_VERSION,
        request_id="req-1",
        input_hash="deadbeef",
        model_revision="verifier-v0",
        threshold_version="thresholds-v0",
        created_at="2026-09-01T00:00:00+00:00",
        run_mode=RunMode.VERIFY,
        verdicts=(verdict,),
        calibrator_id="test-calibrator",
        calibrator_revision="test-calibrator-v1",
    )
    fields.update(overrides)
    if fields["run_mode"] is RunMode.HEAL:
        fields.setdefault("initial_verdicts", fields["verdicts"])
        fields.setdefault("heal_progress_calibrator_id", "test-progress-calibrator")
        fields.setdefault(
            "heal_progress_calibrator_revision", "test-progress-calibrator-v1"
        )
    return AuditReport(**fields)


def test_valid_report_passes_semantic_validation():
    validate_audit_report_semantics(_sample_report().to_dict())


def test_semantics_rejects_untraceable_or_partial_calibration_metadata():
    payload = _sample_report().to_dict()
    payload["calibrator_id"] = None
    payload["calibrator_revision"] = None
    with pytest.raises(DomainValidationError, match="calibrated verdicts require"):
        validate_audit_report_semantics(payload)

    payload = _sample_report().to_dict()
    payload["calibrator_revision"] = None
    with pytest.raises(DomainValidationError, match="both be set"):
        validate_audit_report_semantics(payload)


def test_valid_report_with_repair_activity_and_metrics_passes():
    report = _sample_report(
        run_mode=RunMode.HEAL,
        repair_rounds=2,
        repair_actions=(
            RepairActionRecord(
                round=2,
                claim_id="c1",
                action=RepairAction.REWRITE,
                state_before=VerificationState.INSUFFICIENT_EVIDENCE,
                state_after=VerificationState.SUPPORTED,
                claim_id_after="c1",
                claim_text_before="old claim",
                claim_text_after="Paris is in France.",
                calibrated_confidence_before=0.2,
                calibrated_confidence_after=0.8,
                elapsed_ms=42.0,
                estimated_cost=0.001,
            ),
        ),
        stop_reason="max_rounds_exhausted",
        metrics=RunMetrics(
            total_latency_ms=500.0,
            stage_latencies_ms={"decompose": 50.0, "verify": 450.0},
        ),
    )
    validate_audit_report_semantics(report.to_dict())


# --- RunMode <-> repair-field invariant (core_requirements #8.1) ----------


def test_verify_with_repair_rounds_rejected_on_foreign_document():
    # AuditReport's own constructor already forbids this combination, so
    # build a valid report and mutate the dict directly to simulate a
    # foreign document that bypassed the Python constructor.
    payload = _sample_report().to_dict()
    payload["repair_rounds"] = 1
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_verify_with_repair_action_rejected_on_foreign_document():
    payload = _sample_report().to_dict()
    payload["repair_rounds"] = 1
    payload["repair_actions"] = [
        RepairActionRecord(
            round=1,
            claim_id="c1",
            action=RepairAction.REWRITE,
            state_before=VerificationState.INSUFFICIENT_EVIDENCE,
            state_after=VerificationState.SUPPORTED,
            claim_id_after="c1",
            claim_text_before="old claim",
            claim_text_after="Paris is in France.",
        ).to_dict()
    ]
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_verify_with_stop_reason_rejected_on_foreign_document():
    payload = _sample_report().to_dict()
    payload["stop_reason"] = "some reason"
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_heal_with_null_stop_reason_rejected_on_foreign_document():
    payload = _sample_report(run_mode=RunMode.HEAL, stop_reason="x").to_dict()
    payload["stop_reason"] = None
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_heal_with_empty_stop_reason_rejected_on_foreign_document():
    payload = _sample_report(run_mode=RunMode.HEAL, stop_reason="x").to_dict()
    payload["stop_reason"] = ""
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_heal_with_zero_repair_rounds_accepted():
    payload = _sample_report(run_mode=RunMode.HEAL, stop_reason="nothing_to_repair").to_dict()
    validate_audit_report_semantics(payload)


def test_out_of_order_claim_span_rejected():
    # This exact shape passes jsonschema.validate (see
    # test_schema.py::test_schema_alone_does_not_catch_out_of_order_span)
    # because JSON Schema cannot compare two sibling properties.
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["claim"]["start_char"] = 5
    payload["verdicts"][0]["claim"]["end_char"] = 0
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_out_of_order_evidence_span_rejected():
    payload = _sample_report().to_dict()
    ref = payload["verdicts"][0]["evidence_assessments"][0]["reference"]
    ref["start_char"] = 10
    ref["end_char"] = 3
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_duplicate_claim_id_across_verdicts_rejected_on_foreign_document():
    # AuditReport's own constructor already rejects this at construction
    # time (stage 0.3), so build the payload by mutating a valid dict
    # rather than via the constructor.
    payload = _sample_report().to_dict()
    duplicate_verdict = json.loads(json.dumps(payload["verdicts"][0]))
    payload["verdicts"].append(duplicate_verdict)
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_duplicate_evidence_edge_within_one_verdict_rejected():
    # ClaimVerdict's own constructor already rejects this, so bypass it by
    # editing the dict directly -- proving the check also holds for a
    # foreign document that never went through the Python constructor.
    payload = _sample_report().to_dict()
    duplicate = payload["verdicts"][0]["evidence_assessments"][0]
    payload["verdicts"][0]["evidence_assessments"] = [duplicate, dict(duplicate)]
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_supported_verdict_with_only_contradicted_edge_rejected_on_foreign_document():
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["evidence_assessments"][0]["state"] = "CONTRADICTED"
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_conflicting_evidence_without_both_sides_rejected_on_foreign_document():
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["state"] = "CONFLICTING_EVIDENCE"
    # only a SUPPORTED edge present
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_repair_action_referencing_unknown_claim_id_rejected_on_foreign_document():
    # AuditReport's own constructor already rejects this at construction
    # time (stage 0.3), so mutate a valid payload's dict directly.
    payload = _sample_report(
        run_mode=RunMode.HEAL,
        repair_rounds=1,
        repair_actions=(
            RepairActionRecord(
                round=1,
                claim_id="c1",
                action=RepairAction.REWRITE,
                state_before=VerificationState.INSUFFICIENT_EVIDENCE,
                state_after=VerificationState.SUPPORTED,
                claim_id_after="c1",
                claim_text_before="old claim",
                claim_text_after="Paris is in France.",
            ),
        ),
        stop_reason="x",
    ).to_dict()
    payload["repair_actions"][0]["claim_id"] = "does-not-exist"
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_repair_action_referencing_existing_claim_id_accepted():
    payload = _sample_report(
        run_mode=RunMode.HEAL,
        repair_rounds=1,
        repair_actions=(
            RepairActionRecord(
                round=1,
                claim_id="c1",
                action=RepairAction.REWRITE,
                state_before=VerificationState.INSUFFICIENT_EVIDENCE,
                state_after=VerificationState.SUPPORTED,
                claim_id_after="c1",
                claim_text_before="old claim",
                claim_text_after="Paris is in France.",
            ),
        ),
        stop_reason="x",
    ).to_dict()
    validate_audit_report_semantics(payload)


def test_nan_raw_score_rejected():
    # json.dumps allows the non-standard NaN token by default, and
    # json.loads parses it back into a real (non-finite) float that still
    # satisfies a plain "type": "number" schema check.
    raw_text = json.dumps(_sample_report().to_dict()).replace('"raw_score": 12.3', '"raw_score": NaN')
    payload = json.loads(raw_text)
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_infinite_calibrated_confidence_rejected():
    raw_text = json.dumps(_sample_report().to_dict()).replace(
        '"calibrated_confidence": 0.91', '"calibrated_confidence": Infinity'
    )
    payload = json.loads(raw_text)
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_nan_relevance_score_rejected():
    raw_text = json.dumps(_sample_report().to_dict()).replace(
        '"relevance_score": 0.9', '"relevance_score": NaN'
    )
    payload = json.loads(raw_text)
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_nan_label_score_rejected():
    raw_text = json.dumps(_sample_report().to_dict()).replace(
        '"supported": 2.0', '"supported": NaN'
    )
    payload = json.loads(raw_text)
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_probabilities_summing_to_one_accepted():
    report = _sample_report(
        verdicts=[
            ClaimVerdict(
                claim=_claim("c1"),
                state=VerificationState.SUPPORTED,
                evidence_assessments=[
                    EvidenceAssessment(
                        reference=EvidenceReference(chunk_id="chunk-1"),
                        label_scores=LabelScores(
                            supported=0.7, contradicted=0.2, insufficient=0.1, score_kind="probabilities"
                        ),
                        state=VerificationState.SUPPORTED,
                        rationale=None,
                        verifier_id="v",
                        verifier_revision="0",
                    )
                ],
            )
        ]
    )
    validate_audit_report_semantics(report.to_dict())


def test_probabilities_not_summing_to_one_rejected_on_foreign_document():
    # LabelScores' own constructor already rejects this (stage 0.3), so
    # mutate a valid payload's dict directly to simulate a document that
    # bypassed the Python constructor -- e.g. a hand-edited or foreign-
    # client JSON file.
    payload = _sample_report().to_dict()
    label_scores = payload["verdicts"][0]["evidence_assessments"][0]["label_scores"]
    label_scores["score_kind"] = "probabilities"
    label_scores["supported"] = 0.9
    label_scores["contradicted"] = 0.9
    label_scores["insufficient"] = 0.9
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_infinite_metrics_total_latency_rejected():
    payload = _sample_report(metrics=RunMetrics(total_latency_ms=1.0)).to_dict()
    raw_text = json.dumps(payload).replace('"total_latency_ms": 1.0', '"total_latency_ms": Infinity')
    payload = json.loads(raw_text)
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_nan_stage_latency_rejected():
    payload = _sample_report(
        metrics=RunMetrics(stage_latencies_ms={"decompose": 1.0})
    ).to_dict()
    raw_text = json.dumps(payload).replace('"decompose": 1.0', '"decompose": NaN')
    payload = json.loads(raw_text)
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_infinite_repair_action_calibrated_confidence_rejected():
    payload = _sample_report(
        run_mode=RunMode.HEAL,
        repair_rounds=1,
        repair_actions=(
            RepairActionRecord(
                round=1,
                claim_id="c1",
                action=RepairAction.REWRITE,
                state_before=VerificationState.INSUFFICIENT_EVIDENCE,
                state_after=VerificationState.SUPPORTED,
                claim_id_after="c1",
                claim_text_before="old claim",
                claim_text_after="Paris is in France.",
                calibrated_confidence_before=0.2,
            ),
        ),
        stop_reason="x",
    ).to_dict()
    raw_text = json.dumps(payload).replace(
        '"calibrated_confidence_before": 0.2', '"calibrated_confidence_before": -Infinity'
    )
    payload = json.loads(raw_text)
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_repair_action_round_exceeding_repair_rounds_rejected():
    payload = _sample_report(run_mode=RunMode.HEAL, stop_reason="x").to_dict()
    payload["repair_rounds"] = 1
    payload["repair_actions"] = [
        {
            "round": 2,
            "claim_id": "c1",
            "action": "rewrite",
                "state_before": "INSUFFICIENT_EVIDENCE",
                "state_after": "SUPPORTED",
                "claim_id_after": "c1",
                "claim_text_before": "old claim",
                "claim_text_after": "Paris is in France.",
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
    with pytest.raises(DomainValidationError):
        validate_audit_report_semantics(payload)


def test_repair_action_round_within_bounds_accepted():
    payload = _sample_report(run_mode=RunMode.HEAL, stop_reason="x").to_dict()
    payload["repair_rounds"] = 3
    payload["repair_actions"] = [
        {
            "round": 3,
            "claim_id": "c1",
            "action": "rewrite",
            "state_before": "INSUFFICIENT_EVIDENCE",
            "state_after": "SUPPORTED",
            "claim_id_after": "c1",
            "claim_text_before": "old claim",
            "claim_text_after": "Paris is in France.",
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
    validate_audit_report_semantics(payload)
