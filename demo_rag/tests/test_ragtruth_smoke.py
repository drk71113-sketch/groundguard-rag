from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from demo_rag.app.datasets.ragtruth import RagTruthAnnotation, RagTruthExample
from demo_rag.app.evaluation.ragtruth_smoke import (
    BinaryMetrics,
    PredictedClaim,
    RagTruthSmokeReport,
    align_claims,
    default_demo_root,
    evaluate_smoke_subset,
    write_report,
)
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.models import Chunk, VerificationRequest


def _annotation(start: int = 4, end: int = 9) -> RagTruthAnnotation:
    return RagTruthAnnotation(
        start=start,
        end=end,
        text="xxxxx"[: end - start],
        label_type="Evident Baseless Info",
        meta=None,
        implicit_true=False,
        due_to_null=False,
    )


def _claim(
    state: VerificationState,
    *,
    start: int = 0,
    end: int = 10,
    claim_id: str = "claim-1",
) -> PredictedClaim:
    return PredictedClaim(
        claim_id=claim_id,
        text="x" * (end - start),
        start=start,
        end=end,
        state=state,
    )


@pytest.mark.parametrize(
    ("state", "expected_risk"),
    [
        (VerificationState.SUPPORTED, False),
        (VerificationState.NOT_CHECKABLE, False),
        (VerificationState.CONTRADICTED, True),
        (VerificationState.INSUFFICIENT_EVIDENCE, True),
        (VerificationState.CONFLICTING_EVIDENCE, True),
    ],
)
def test_align_claims_uses_registered_risk_state_mapping(state, expected_risk):
    comparisons = align_claims([_claim(state)], [_annotation()])

    assert comparisons[0].gold_risk is True
    assert comparisons[0].predicted_risk is expected_risk


def test_half_open_spans_touching_at_boundary_do_not_overlap():
    comparisons = align_claims(
        [
            _claim(VerificationState.SUPPORTED, start=0, end=4, claim_id="first"),
            _claim(
                VerificationState.INSUFFICIENT_EVIDENCE,
                start=4,
                end=9,
                claim_id="second",
            ),
        ],
        [_annotation(start=4, end=9)],
    )

    assert comparisons[0].gold_risk is False
    assert comparisons[1].gold_risk is True


def test_align_claims_rejects_annotation_outside_all_claims():
    with pytest.raises(ValueError, match="not covered"):
        align_claims(
            [_claim(VerificationState.SUPPORTED, start=0, end=4)],
            [_annotation(start=5, end=9)],
        )


def test_binary_metrics_are_derived_from_explicit_counts():
    metrics = BinaryMetrics(
        true_positive=8,
        false_positive=2,
        false_negative=2,
        true_negative=8,
    )

    assert metrics.count == 20
    assert metrics.precision == pytest.approx(0.8)
    assert metrics.recall == pytest.approx(0.8)
    assert metrics.f1 == pytest.approx(0.8)
    assert metrics.accuracy == pytest.approx(0.8)


def test_binary_metrics_handle_empty_denominators():
    metrics = BinaryMetrics(0, 0, 0, 0)

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0
    assert metrics.accuracy == 0.0


class _FakeGuard:
    def verify(self, request):
        state = (
            VerificationState.INSUFFICIENT_EVIDENCE
            if request.request_id.endswith("risk")
            else VerificationState.SUPPORTED
        )
        claim = SimpleNamespace(
            claim_id="claim-0",
            text=request.answer,
            start_char=0,
            end_char=len(request.answer),
        )
        verdict = SimpleNamespace(claim=claim, state=state)
        report = SimpleNamespace(verdicts=(verdict,))
        return SimpleNamespace(
            audit_report=report,
            to_dict=lambda: {"audit_report": {"verdicts": []}},
        )


def _example(response_id: str, *, annotated: bool) -> RagTruthExample:
    answer = "xxxxx"
    annotations = (_annotation(0, 5),) if annotated else ()
    return RagTruthExample(
        response_id=response_id,
        source_id=f"source-{response_id}",
        model="fixture-model",
        split="test",
        quality="good",
        request=VerificationRequest(
            request_id=f"request-{response_id}",
            answer=answer,
            chunks=(Chunk(chunk_id="chunk-1", text="evidence"),),
            query="question",
        ),
        annotations=annotations,
    )


def test_evaluate_smoke_subset_computes_claim_and_response_counts():
    examples = (
        _example("risk", annotated=True),
        _example("clean", annotated=False),
    )
    ticks = iter([10.0, 10.5])

    report = evaluate_smoke_subset(
        _FakeGuard(), examples, monotonic=lambda: next(ticks)
    )

    assert report.claim_metrics.true_positive == 1
    assert report.claim_metrics.true_negative == 1
    assert report.response_metrics.true_positive == 1
    assert report.response_metrics.true_negative == 1
    assert report.total_latency_ms == pytest.approx(500.0)


def test_write_report_creates_summary_and_one_jsonl_row(tmp_path):
    report = RagTruthSmokeReport(
        claim_metrics=BinaryMetrics(1, 0, 0, 0),
        response_metrics=BinaryMetrics(1, 0, 0, 0),
        state_counts={"INSUFFICIENT_EVIDENCE": 1},
        model_counts={"fixture": 1},
        cases=({"response_id": "r1"},),
        total_latency_ms=12.5,
    )

    write_report(report, tmp_path)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    cases = (tmp_path / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["response_count"] == 1
    assert json.loads(cases[0]) == {"response_id": "r1"}


def test_default_demo_root_points_at_reference_application():
    root = default_demo_root()

    assert root.name == "demo_rag"
    assert (root / "app" / "evaluation" / "ragtruth_smoke.py").is_file()
