from __future__ import annotations

import json
from pathlib import Path

import pytest

from demo_rag.app.datasets.ragtruth import (
    RagTruthExample,
    RagTruthFormatError,
    load_qa_test_examples,
    parse_qa_passages,
    select_smoke_subset,
)


def _source(source_id: str = "source-1") -> dict:
    return {
        "source_id": source_id,
        "task_type": "QA",
        "source": "MARCO",
        "source_info": {
            "question": "Where is the office?",
            "passages": (
                "passage 1:The office is in Chicago.\n\n"
                "passage 2:The factory is in Austin.\n\n"
                "passage 3:The company was founded in 2020.\n\n"
            ),
        },
        "prompt": "unused upstream prompt",
    }


def _response(
    response_id: str,
    *,
    source_id: str = "source-1",
    answer: str = "The office is in Chicago.",
    annotated: bool = False,
    split: str = "test",
    quality: str = "good",
) -> dict:
    labels = []
    if annotated:
        start = answer.index("Chicago")
        labels = [
            {
                "start": start,
                "end": start + len("Chicago"),
                "text": "Chicago",
                "meta": None,
                "label_type": "Evident Baseless Info",
                "implicit_true": False,
                "due_to_null": False,
            }
        ]
    return {
        "id": response_id,
        "source_id": source_id,
        "model": "test-model",
        "temperature": 0.0,
        "labels": labels,
        "split": split,
        "quality": quality,
        "response": answer,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _load(tmp_path: Path, sources: list[dict], responses: list[dict]):
    source_path = tmp_path / "source_info.jsonl"
    response_path = tmp_path / "response.jsonl"
    _write_jsonl(source_path, sources)
    _write_jsonl(response_path, responses)
    return load_qa_test_examples(source_path, response_path)


def test_parse_qa_passages_returns_three_bodies_without_headers():
    passages = parse_qa_passages(
        "passage 1:Alpha.\n\npassage 2:Beta.\n\npassage 3:Gamma.\n",
        source_id="s1",
    )

    assert passages == ("Alpha.", "Beta.", "Gamma.")


@pytest.mark.parametrize(
    "raw",
    [
        "there are no headers",
        "prefix\npassage 1:Alpha",
        "passage 1:Alpha\npassage 3:Gamma",
        "passage 1:\npassage 2:Beta",
    ],
)
def test_parse_qa_passages_rejects_malformed_input(raw: str):
    with pytest.raises(RagTruthFormatError):
        parse_qa_passages(raw, source_id="s1")


def test_loader_builds_groundguard_request_and_keeps_labels_separate(tmp_path: Path):
    examples = _load(tmp_path, [_source()], [_response("r1", annotated=True)])

    assert len(examples) == 1
    example = examples[0]
    assert example.response_id == "r1"
    assert example.request.request_id == "ragtruth-response-r1"
    assert example.request.answer == "The office is in Chicago."
    assert example.request.query == "Where is the office?"
    assert [chunk.text for chunk in example.request.chunks] == [
        "The office is in Chicago.",
        "The factory is in Austin.",
        "The company was founded in 2020.",
    ]
    assert example.request.chunks[0].metadata["upstream_source_id"] == "source-1"
    assert example.annotations[0].text == "Chicago"
    assert example.has_hallucination is True


def test_loader_filters_non_qa_train_and_bad_quality_rows(tmp_path: Path):
    non_qa = _source("summary-source")
    non_qa["task_type"] = "Summary"
    non_qa["source_info"] = {}
    examples = _load(
        tmp_path,
        [_source(), non_qa],
        [
            _response("good"),
            _response("train", split="train"),
            _response("refusal", quality="incorrect_refusal"),
            _response("summary", source_id="summary-source"),
        ],
    )

    assert [item.response_id for item in examples] == ["good"]


def test_loader_rejects_annotation_text_that_disagrees_with_offsets(tmp_path: Path):
    row = _response("r1", annotated=True)
    row["labels"][0]["text"] = "wrong"

    with pytest.raises(RagTruthFormatError, match="does not match offsets"):
        _load(tmp_path, [_source()], [row])


def test_loader_reports_invalid_json_line(tmp_path: Path):
    source_path = tmp_path / "source_info.jsonl"
    response_path = tmp_path / "response.jsonl"
    source_path.write_text("not-json\n", encoding="utf-8")
    _write_jsonl(response_path, [_response("r1")])

    with pytest.raises(RagTruthFormatError, match="line 1: invalid JSON"):
        load_qa_test_examples(source_path, response_path)


def test_loader_rejects_duplicate_eligible_response_ids(tmp_path: Path):
    with pytest.raises(RagTruthFormatError, match="duplicate response id"):
        _load(tmp_path, [_source()], [_response("same"), _response("same")])


def test_smoke_subset_is_balanced_deterministic_and_order_independent(
    tmp_path: Path,
):
    rows = [
        _response(f"clean-{index}", annotated=False) for index in range(5)
    ] + [_response(f"marked-{index}", annotated=True) for index in range(5)]
    examples = _load(tmp_path, [_source()], rows)

    forward = select_smoke_subset(examples, per_class=3, salt="fixed")
    reverse = select_smoke_subset(reversed(examples), per_class=3, salt="fixed")

    assert [item.response_id for item in forward] == [
        item.response_id for item in reverse
    ]
    assert sum(item.has_hallucination for item in forward) == 3
    assert sum(not item.has_hallucination for item in forward) == 3


def test_smoke_subset_rejects_insufficient_class_examples(tmp_path: Path):
    examples: tuple[RagTruthExample, ...] = _load(
        tmp_path,
        [_source()],
        [_response("clean"), _response("marked", annotated=True)],
    )

    with pytest.raises(ValueError, match="not enough examples"):
        select_smoke_subset(examples, per_class=2)
