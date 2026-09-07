from __future__ import annotations

import json

import pytest

from demo_rag.app.evaluation.error_analysis import (
    analyze_cases,
    load_and_validate_manual_review,
    load_frozen_cases,
    select_review_queue,
    write_analysis,
)


def _case(
    *,
    response_id: str = "r1",
    outcome: str = "false_positive",
    state: str = "INSUFFICIENT_EVIDENCE",
    annotated: bool = False,
    with_edge: bool = True,
) -> dict:
    annotation = {
        "start": 2,
        "end": 4,
        "text": "ai",
        "label_type": "Evident Baseless Info",
        "meta": None,
        "implicit_true": False,
        "due_to_null": False,
    }
    edges = []
    if with_edge:
        edges = [
            {
                "reference": {
                    "chunk_id": "chunk-1",
                    "start_char": None,
                    "end_char": None,
                    "relevance_score": 0.5,
                },
                "label_scores": {
                    "supported": 0.7,
                    "contradicted": 0.1,
                    "insufficient": 0.2,
                    "score_kind": "probabilities",
                },
                "state": state,
            }
        ]
    return {
        "response_id": response_id,
        "source_id": "s1",
        "upstream_model": "fixture",
        "question": "question",
        "answer": "claim text",
        "chunks": [
            {
                "chunk_id": "chunk-1",
                "text": "evidence",
                "source": "fixture",
                "metadata": {},
            }
        ],
        "ragtruth_annotations": [annotation] if annotated else [],
        "claim_comparisons": [
            {
                "claim_id": "claim-1",
                "text": "claim text",
                "start": 0,
                "end": 10,
                "state": state,
                "gold_risk": annotated,
                "predicted_risk": outcome in {"true_positive", "false_positive"},
                "outcome": outcome,
                "overlapping_annotation_indexes": [0] if annotated else [],
            }
        ],
        "groundguard_output": {
            "audit_report": {
                "verdicts": [
                    {
                        "claim": {"claim_id": "claim-1"},
                        "evidence_assessments": edges,
                    }
                ]
            }
        },
    }


def test_load_frozen_cases_rejects_duplicate_response_ids(tmp_path):
    path = tmp_path / "cases.jsonl"
    row = _case()
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="duplicate response_id"):
        load_frozen_cases(path)


def test_analyze_cases_reports_near_threshold_and_token_limit_signals():
    summary, rows = analyze_cases([_case()], pair_token_count=lambda _a, _b: 300)

    assert rows[0]["max_label_scores"]["supported"] == 0.7
    assert "max_support_between_0.5_and_0.8" in rows[0]["signals"]
    assert "all_selected_pairs_exceed_256_tokens" in rows[0]["signals"]
    assert summary["error_claim_count"] == 1


def test_analyze_cases_reports_zero_evidence_without_calling_tokenizer():
    def should_not_run(_premise, _hypothesis):
        raise AssertionError("token counter must not run without evidence")

    _, rows = analyze_cases(
        [_case(with_edge=False)], pair_token_count=should_not_run
    )

    assert rows[0]["signals"] == ["zero_evidence_edges"]


def test_small_annotated_span_is_recorded_as_granularity_signal():
    _, rows = analyze_cases(
        [
            _case(
                outcome="false_negative",
                state="SUPPORTED",
                annotated=True,
            )
        ],
        pair_token_count=lambda _a, _b: 20,
    )

    assert rows[0]["annotation_to_claim_ratio"] == pytest.approx(0.2)
    assert "annotated_span_is_at_most_25pct_of_claim" in rows[0]["signals"]


def test_review_queue_keeps_all_fn_and_registered_strata_deterministically():
    rows = []
    for index in range(5):
        rows.append(
            {
                "response_id": f"fn-{index}",
                "claim_id": "c",
                "outcome": "false_negative",
                "state": "SUPPORTED",
            }
        )
    for state, count in (
        ("INSUFFICIENT_EVIDENCE", 10),
        ("CONTRADICTED", 10),
        ("CONFLICTING_EVIDENCE", 10),
    ):
        for index in range(count):
            rows.append(
                {
                    "response_id": f"fp-{state}-{index}",
                    "claim_id": "c",
                    "outcome": "false_positive",
                    "state": state,
                }
            )
    for outcome in ("true_positive", "true_negative"):
        for index in range(10):
            rows.append(
                {
                    "response_id": f"{outcome}-{index}",
                    "claim_id": "c",
                    "outcome": outcome,
                    "state": "SUPPORTED",
                }
            )

    forward = select_review_queue(rows)
    reverse = select_review_queue(list(reversed(rows)))

    assert [(x["response_id"], x["claim_id"]) for x in forward] == [
        (x["response_id"], x["claim_id"]) for x in reverse
    ]
    assert sum(x["outcome"] == "false_negative" for x in forward) == 5
    assert sum(x["outcome"] == "false_positive" for x in forward) == 12
    assert sum(x["outcome"] == "true_positive" for x in forward) == 4
    assert sum(x["outcome"] == "true_negative" for x in forward) == 4


def test_write_analysis_creates_all_three_artifacts(tmp_path):
    summary = {"claim_count": 1}
    rows = ({"response_id": "r1"},)

    write_analysis(summary, rows, rows, tmp_path)

    assert json.loads((tmp_path / "diagnostics.json").read_text())["claim_count"] == 1
    assert len((tmp_path / "claim_diagnostics.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "review_queue.jsonl").read_text().splitlines()) == 1


def test_manual_review_must_cover_the_exact_queue(tmp_path):
    queue = (
        {
            "response_id": "r1",
            "claim_id": "c1",
            "outcome": "false_positive",
        },
    )
    review = {
        "response_id": "r1",
        "claim_id": "c1",
        "outcome": "false_positive",
        "primary_observation_layer": "checkability",
        "confidence": "high",
        "note": "A factual-looking discourse heading reached NLI.",
    }
    path = tmp_path / "manual_review.jsonl"
    path.write_text(json.dumps(review) + "\n", encoding="utf-8")

    assert load_and_validate_manual_review(path, queue) == (review,)


def test_manual_review_rejects_missing_queue_item(tmp_path):
    path = tmp_path / "manual_review.jsonl"
    path.write_text("", encoding="utf-8")
    queue = ({"response_id": "r1", "claim_id": "c1", "outcome": "false_positive"},)

    with pytest.raises(ValueError, match="missing queued claims"):
        load_and_validate_manual_review(path, queue)
