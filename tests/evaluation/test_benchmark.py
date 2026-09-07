from __future__ import annotations

import dataclasses
import json

import pytest

from groundguard_rag import Chunk, GroundGuard, VerificationRequest
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.evaluation import (
    BenchmarkCase,
    DatasetFormatError,
    EvaluationContractError,
    evaluate_verification,
    load_jsonl_dataset,
)

from tests.stage8._helpers import make_verify_service


def _case(case_id: str, answer: str, expected: VerificationState) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        request=VerificationRequest(
            request_id=case_id,
            answer=answer,
            chunks=(Chunk(f"{case_id}-chunk", "Paris is in France."),),
        ),
        expected_states=(expected,),
    )


def test_evaluate_verification_builds_complete_five_state_matrix():
    report = evaluate_verification(
        GroundGuard(verify_service=make_verify_service()),
        [
            _case("supported", "Paris is in France.", VerificationState.SUPPORTED),
            _case(
                "contradicted",
                "Paris is in Germany.",
                VerificationState.CONTRADICTED,
            ),
        ],
        monotonic=iter([1.0, 1.025]).__next__,
    )

    assert report.case_count == 2
    assert report.claim_count == 2
    assert report.accuracy == 1.0
    assert report.macro_f1 == pytest.approx(0.4)
    assert report.confusion_matrix["SUPPORTED"]["SUPPORTED"] == 1
    assert set(report.confusion_matrix) == {state.value for state in VerificationState}
    assert report.calibration is None
    assert report.total_latency_ms == pytest.approx(25.0)


def test_calibration_metrics_use_only_traceable_calibrated_confidence():
    service = make_verify_service()
    original_verify = service.verify

    def calibrated_verify(request):
        report = original_verify(request)
        verdict = dataclasses.replace(
            report.verdicts[0], calibrated_confidence=0.8
        )
        return dataclasses.replace(
            report,
            verdicts=(verdict,),
            calibrator_id="held-out-calibrator",
            calibrator_revision="sha256:test",
        )

    service.verify = calibrated_verify
    report = evaluate_verification(
        GroundGuard(verify_service=service),
        [_case("one", "Paris is in France.", VerificationState.SUPPORTED)],
        calibration_bins=5,
        monotonic=iter([0.0, 0.1]).__next__,
    )
    assert report.calibration is not None
    assert report.calibration.count == 1
    assert report.calibration.coverage == 1.0
    assert report.calibration.brier_score == pytest.approx(0.04)
    assert report.calibration.expected_calibration_error == pytest.approx(0.2)


def test_mismatched_claim_count_and_duplicate_cases_fail_explicitly():
    guard = GroundGuard(verify_service=make_verify_service())
    mismatch = dataclasses.replace(
        _case("mismatch", "Paris is in France.", VerificationState.SUPPORTED),
        expected_states=(VerificationState.SUPPORTED, VerificationState.SUPPORTED),
    )
    with pytest.raises(EvaluationContractError):
        evaluate_verification(guard, [mismatch])
    duplicate = _case("duplicate", "Paris is in France.", VerificationState.SUPPORTED)
    with pytest.raises(DomainValidationError):
        evaluate_verification(guard, [duplicate, duplicate])


def test_invalid_clock_and_arguments_are_rejected():
    guard = GroundGuard(verify_service=make_verify_service())
    case = _case("one", "Paris is in France.", VerificationState.SUPPORTED)
    with pytest.raises(EvaluationContractError):
        evaluate_verification(guard, [case], monotonic=lambda: float("nan"))
    with pytest.raises(DomainValidationError):
        evaluate_verification(guard, [], calibration_bins=10)
    with pytest.raises(DomainValidationError):
        evaluate_verification(guard, [case], calibration_bins=True)


def test_jsonl_loader_round_trips_strict_labeled_rows(tmp_path):
    path = tmp_path / "dataset.jsonl"
    row = {
        "case_id": "case-1",
        "request": {
            "request_id": "request-1",
            "answer": "Paris is in France.",
            "chunks": [{"chunk_id": "chunk-1", "text": "Paris is in France."}],
        },
        "expected_states": ["SUPPORTED"],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    cases = load_jsonl_dataset(path)
    assert cases[0].case_id == "case-1"
    assert cases[0].expected_states == (VerificationState.SUPPORTED,)


@pytest.mark.parametrize(
    "contents",
    [
        "not-json\n",
        "{}\n",
        json.dumps(
            {
                "case_id": "bad",
                "request": {"request_id": "r", "answer": "a", "chunks": []},
                "expected_states": ["UNKNOWN"],
            }
        ),
    ],
)
def test_jsonl_loader_reports_line_without_echoing_row(contents, tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(DatasetFormatError, match="line 1") as captured:
        load_jsonl_dataset(path)
    assert contents.strip() not in str(captured.value)
