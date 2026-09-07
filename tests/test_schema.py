import json
import math
from pathlib import Path

import jsonschema
import pytest

from groundguard_rag.domain.enums import RepairAction, RunMode, VerificationState
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

SCHEMA_PATH = (
    Path(__file__).parent.parent
    / "src"
    / "groundguard_rag"
    / "schema"
    / "audit_report.v1.schema.json"
)

# Format assertions ("date-time" on created_at) are only actually checked
# by jsonschema when a FormatChecker with a registered "date-time" checker
# is passed in -- see the stage 0.1 fix note below.
FORMAT_CHECKER = jsonschema.FormatChecker()


def _validate(instance: dict, schema: dict) -> None:
    jsonschema.validate(instance=instance, schema=schema, format_checker=FORMAT_CHECKER)


@pytest.fixture(scope="module")
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_file_is_a_valid_json_schema(schema):
    # Raises if the schema document itself is malformed.
    jsonschema.Draft202012Validator.check_schema(schema)


def test_format_checker_actually_has_a_date_time_checker_registered():
    # Guards the fix itself: jsonschema.FormatChecker() only validates
    # "format": "date-time" if a checker for it is registered (which
    # requires the rfc3339-validator package to be installed). If this
    # assertion ever fails, every date-time check in this file is
    # silently a no-op again.
    assert "date-time" in FORMAT_CHECKER.checkers


def test_schema_declared_version_matches_the_code_constant_exactly(schema):
    # Locking the schema to a literal "const" and asserting exact string
    # equality here catches drift: changing SCHEMA_VERSION without
    # updating the schema file's "const" now fails this test (a regex
    # pattern match alone would not).
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


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


def _sample_report(**overrides) -> AuditReport:
    claim = AtomicClaim(claim_id="c1", text="Paris is in France.", start_char=0, end_char=20)
    verdict = ClaimVerdict(
        claim=claim,
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


def test_minimal_verify_only_report_validates_against_schema(schema):
    report = AuditReport(
        schema_version=SCHEMA_VERSION,
        request_id="req-1",
        input_hash="deadbeef",
        model_revision="verifier-v0",
        threshold_version="thresholds-v0",
        created_at="2026-09-01T00:00:00+00:00",
        run_mode=RunMode.VERIFY,
        verdicts=(),
    )
    _validate(report.to_dict(), schema)


def test_populated_report_with_a_verdict_validates_against_schema(schema):
    _validate(_sample_report().to_dict(), schema)


def test_schema_rejects_partial_or_missing_calibration_metadata(schema):
    payload = _sample_report().to_dict()
    payload["calibrator_revision"] = None
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)

    payload = _sample_report().to_dict()
    payload.pop("calibrator_id")
    payload.pop("calibrator_revision")
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_allows_absent_calibration_metadata_when_no_confidence(schema):
    report = AuditReport(
        schema_version=SCHEMA_VERSION,
        request_id="req-no-cal",
        input_hash="hash",
        model_revision="model",
        threshold_version="thresholds",
        created_at="2026-09-01T00:00:00+00:00",
        run_mode=RunMode.VERIFY,
        verdicts=(),
    ).to_dict()
    report.pop("calibrator_id")
    report.pop("calibrator_revision")
    _validate(report, schema)


def test_report_with_repair_activity_and_metrics_validates_against_schema(schema):
    report = _sample_report(
        run_mode=RunMode.HEAL,
        repair_rounds=2,
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
                note="rewrote using chunk-1",
                evidence_ids_added=("chunk-2",),
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
            model_call_count=4,
            input_tokens=1024,
            output_tokens=256,
            estimated_cost=0.01,
        ),
    )
    _validate(report.to_dict(), schema)


def test_schema_rejects_unknown_top_level_field(schema):
    payload = _sample_report().to_dict()
    payload["unexpected_field"] = "should not be here"
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_invalid_state_value(schema):
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["state"] = "BASELESS"
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_wrong_schema_version(schema):
    payload = _sample_report().to_dict()
    payload["schema_version"] = "1.0.1"
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_invalid_run_mode(schema):
    payload = _sample_report().to_dict()
    payload["run_mode"] = "REPAIR"
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


# --- RunMode <-> repair-field invariant, expressed directly in schema -----


def test_schema_rejects_verify_with_nonzero_repair_rounds(schema):
    payload = _sample_report().to_dict()  # run_mode is VERIFY by default
    payload["repair_rounds"] = 1
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_verify_with_a_repair_action(schema):
    # The exact case the stage 0.3 review flagged as previously accepted.
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
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_verify_with_a_stop_reason(schema):
    payload = _sample_report().to_dict()
    payload["stop_reason"] = "some reason"
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_heal_with_null_stop_reason(schema):
    payload = _sample_report(run_mode=RunMode.HEAL, stop_reason="x").to_dict()
    payload["stop_reason"] = None
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_heal_with_empty_stop_reason(schema):
    payload = _sample_report(run_mode=RunMode.HEAL, stop_reason="x").to_dict()
    payload["stop_reason"] = ""
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_accepts_heal_with_zero_repair_rounds(schema):
    # HEAL explicitly allows repair_rounds == 0 (nothing needed repairing).
    payload = _sample_report(run_mode=RunMode.HEAL, stop_reason="nothing_to_repair").to_dict()
    _validate(payload, schema)


def test_schema_rejects_partial_evidence_span(schema):
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["evidence_assessments"][0]["reference"]["end_char"] = None
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_not_checkable_verdict_with_evidence(schema):
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["state"] = "NOT_CHECKABLE"
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_supported_verdict_with_only_contradicted_edge(schema):
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["evidence_assessments"][0]["state"] = "CONTRADICTED"
    # verdict.state is still SUPPORTED, but its only edge is CONTRADICTED.
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_supported_verdict_with_coexisting_contradicted_edge(schema):
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["evidence_assessments"].append(
        _assessment(VerificationState.CONTRADICTED, chunk_id="chunk-2").to_dict()
    )
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_conflicting_evidence_without_both_supported_and_contradicted(schema):
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["state"] = "CONFLICTING_EVIDENCE"
    # only a SUPPORTED edge present, no CONTRADICTED edge
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_accepts_conflicting_evidence_with_both_supported_and_contradicted(schema):
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["state"] = "CONFLICTING_EVIDENCE"
    payload["verdicts"][0]["evidence_assessments"].append(
        _assessment(VerificationState.CONTRADICTED, chunk_id="chunk-2").to_dict()
    )
    _validate(payload, schema)


def test_schema_rejects_insufficient_evidence_verdict_with_a_supported_edge(schema):
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["state"] = "INSUFFICIENT_EVIDENCE"
    # evidence_assessments[0] is still SUPPORTED -- inconsistent.
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_edge_level_not_checkable_state(schema):
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["evidence_assessments"][0]["state"] = "NOT_CHECKABLE"
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_missing_label_scores_field(schema):
    payload = _sample_report().to_dict()
    del payload["verdicts"][0]["evidence_assessments"][0]["label_scores"]
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_schema_rejects_metrics_with_negative_value(schema):
    payload = _sample_report(metrics=RunMetrics(total_latency_ms=1.0)).to_dict()
    payload["metrics"]["total_latency_ms"] = -1.0
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_naive_created_at_rejected_by_format_checker(schema):
    payload = _sample_report().to_dict()
    # Bypass the domain constructor (which already forbids this) to prove
    # the *schema's* format assertion independently rejects a naive
    # (timezone-less) timestamp on a foreign document.
    payload["created_at"] = "2026-09-01T00:00:00"
    with pytest.raises(jsonschema.ValidationError):
        _validate(payload, schema)


def test_naive_created_at_passes_without_a_format_checker(schema):
    # Demonstrates exactly why FormatChecker must be passed explicitly:
    # draft 2020-12 treats "format" as an annotation, not an assertion,
    # unless a format checker is supplied. Without one, this same
    # malformed payload passes silently.
    payload = _sample_report().to_dict()
    payload["created_at"] = "2026-09-01T00:00:00"
    jsonschema.validate(instance=payload, schema=schema)  # no format_checker


def test_schema_alone_does_not_catch_out_of_order_span(schema):
    # end_char <= start_char is a numeric comparison between two sibling
    # properties, which JSON Schema cannot express -- the schema passes
    # this invalid span. groundguard_rag.schema.semantic_validation is
    # what actually catches it (see test_semantic_validation.py).
    payload = _sample_report().to_dict()
    payload["verdicts"][0]["claim"]["end_char"] = 0
    payload["verdicts"][0]["claim"]["start_char"] = 5
    _validate(payload, schema)  # does not raise


def test_schema_alone_does_not_catch_nan_raw_score(schema):
    # Python's json module parses the non-standard "NaN" token into a real
    # (non-finite) float, which still satisfies "type": "number" -- the
    # schema alone cannot reject it. semantic_validation does.
    raw_text = json.dumps(_sample_report().to_dict()).replace('"raw_score": 12.3', '"raw_score": NaN')
    payload = json.loads(raw_text)
    assert math.isnan(payload["verdicts"][0]["raw_score"])
    _validate(payload, schema)  # does not raise


def test_schema_alone_does_not_catch_duplicate_evidence_edges(schema):
    # Two identical edges (same chunk_id/span/verifier_id/verifier_revision)
    # is not expressible via uniqueItems (items differ by rationale text) or
    # any other plain-schema keyword -- semantic_validation catches it.
    payload = _sample_report().to_dict()
    duplicate = payload["verdicts"][0]["evidence_assessments"][0]
    payload["verdicts"][0]["evidence_assessments"] = [duplicate, dict(duplicate)]
    _validate(payload, schema)  # does not raise


def test_schema_alone_does_not_catch_repair_action_referencing_unknown_claim(schema):
    # claim_id existing among verdicts is a cross-object reference check
    # this schema cannot express -- semantic_validation catches it. This
    # can no longer be constructed via AuditReport (stage 0.3 rejects it
    # directly), so the payload is built valid and then mutated to
    # simulate a foreign document that bypassed the Python constructor.
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
        stop_reason="max_rounds_exhausted",
    ).to_dict()
    payload["repair_actions"][0]["claim_id"] = "does-not-exist"
    _validate(payload, schema)  # does not raise


def test_to_dict_passes_schema_format_checker_and_semantic_validator(schema):
    # A single AuditReport.to_dict() output must satisfy all three layers
    # of validation at once.
    from groundguard_rag.schema.semantic_validation import validate_audit_report_semantics

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
        stop_reason="max_rounds_exhausted",
        metrics=RunMetrics(total_latency_ms=10.0),
    ).to_dict()
    _validate(payload, schema)
    validate_audit_report_semantics(payload)
