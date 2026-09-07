"""Strict adapter from pinned RAGTruth QA data to GroundGuard requests.

This module only converts data.  It never loads an NLI model, calls a provider,
runs GroundGuard, or interprets RAGTruth labels as GroundGuard five-state truth.
Keeping human annotations beside (not inside) the request prevents label
leakage into the verifier input.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groundguard_rag.domain.models import Chunk, VerificationRequest


_PASSAGE_HEADER = re.compile(r"(?m)^passage\s+(\d+):")
_REQUIRED_SOURCE_FIELDS = frozenset(
    {"source_id", "task_type", "source", "source_info", "prompt"}
)
_REQUIRED_RESPONSE_FIELDS = frozenset(
    {"id", "source_id", "model", "temperature", "labels", "split", "quality", "response"}
)
_REQUIRED_ANNOTATION_FIELDS = frozenset(
    {"start", "end", "text", "meta", "label_type", "implicit_true", "due_to_null"}
)


class RagTruthFormatError(ValueError):
    """Pinned RAGTruth input does not satisfy the documented data contract."""


@dataclass(frozen=True)
class RagTruthAnnotation:
    """One human-annotated hallucination span in a RAGTruth response."""

    start: int
    end: int
    text: str
    label_type: str
    meta: str | None
    implicit_true: bool
    due_to_null: bool

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise RagTruthFormatError("annotation.start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise RagTruthFormatError("annotation.end must be an integer")
        if self.start < 0 or self.end <= self.start:
            raise RagTruthFormatError("annotation offsets must satisfy 0 <= start < end")
        if not isinstance(self.text, str) or not self.text:
            raise RagTruthFormatError("annotation.text must be a non-empty string")
        if not isinstance(self.label_type, str) or not self.label_type.strip():
            raise RagTruthFormatError("annotation.label_type must be non-empty")
        if self.meta is not None and not isinstance(self.meta, str):
            raise RagTruthFormatError("annotation.meta must be a string or null")
        if not isinstance(self.implicit_true, bool):
            raise RagTruthFormatError("annotation.implicit_true must be boolean")
        if not isinstance(self.due_to_null, bool):
            raise RagTruthFormatError("annotation.due_to_null must be boolean")


@dataclass(frozen=True)
class RagTruthExample:
    """A GroundGuard request paired with evaluation-only upstream labels."""

    response_id: str
    source_id: str
    model: str
    split: str
    quality: str
    request: VerificationRequest
    annotations: tuple[RagTruthAnnotation, ...]

    @property
    def has_hallucination(self) -> bool:
        """Whether RAGTruth annotators marked at least one response span."""

        return bool(self.annotations)


def parse_qa_passages(raw: str, *, source_id: str) -> tuple[str, ...]:
    """Parse RAGTruth's ``passage N:`` QA string without losing body text."""

    if not isinstance(raw, str) or not raw.strip():
        raise RagTruthFormatError(f"source {source_id!r} passages must be non-empty")
    matches = list(_PASSAGE_HEADER.finditer(raw))
    if not matches:
        raise RagTruthFormatError(f"source {source_id!r} has no passage headers")

    numbers = [int(match.group(1)) for match in matches]
    expected = list(range(1, len(matches) + 1))
    if numbers != expected:
        raise RagTruthFormatError(
            f"source {source_id!r} passage numbers must be consecutive from 1"
        )
    if raw[: matches[0].start()].strip():
        raise RagTruthFormatError(
            f"source {source_id!r} contains text before its first passage"
        )

    passages: list[str] = []
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        body = raw[match.end() : body_end].strip()
        if not body:
            raise RagTruthFormatError(
                f"source {source_id!r} passage {numbers[index]} is empty"
            )
        passages.append(body)
    return tuple(passages)


def load_qa_test_examples(
    source_path: str | Path,
    response_path: str | Path,
) -> tuple[RagTruthExample, ...]:
    """Load eligible QA/test/good rows and join them by upstream source ID."""

    sources = _load_qa_sources(Path(source_path))
    examples: list[RagTruthExample] = []
    response_ids: set[str] = set()

    for line_number, row in _iter_json_objects(Path(response_path)):
        _require_fields(row, _REQUIRED_RESPONSE_FIELDS, "response", line_number)
        if row["split"] != "test" or row["quality"] != "good":
            continue

        response_id = _nonempty_string(row["id"], "response.id", line_number)
        if response_id in response_ids:
            raise RagTruthFormatError(
                f"response line {line_number}: duplicate response id {response_id!r}"
            )
        response_ids.add(response_id)

        source_id = _nonempty_string(
            row["source_id"], "response.source_id", line_number
        )
        source = sources.get(source_id)
        # Non-QA rows legitimately point at sources absent from the QA index.
        if source is None:
            continue
        examples.append(_convert_response(row, line_number, source))

    if not examples:
        raise RagTruthFormatError("no eligible QA/test/good responses were found")
    return tuple(examples)


def select_smoke_subset(
    examples: Iterable[RagTruthExample],
    *,
    per_class: int = 25,
    salt: str = "groundguard-ragtruth-smoke-v1",
) -> tuple[RagTruthExample, ...]:
    """Select equal clean/annotated samples by stable hash, never input order."""

    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 1:
        raise ValueError("per_class must be an integer >= 1")
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")

    values = tuple(examples)
    if any(not isinstance(item, RagTruthExample) for item in values):
        raise TypeError("examples must contain only RagTruthExample instances")
    ids = [item.response_id for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError("examples must not contain duplicate response IDs")

    clean = [item for item in values if not item.has_hallucination]
    annotated = [item for item in values if item.has_hallucination]
    if len(clean) < per_class or len(annotated) < per_class:
        raise ValueError(
            "not enough examples to select the requested number from both classes"
        )

    def rank(item: RagTruthExample) -> tuple[str, str]:
        digest = hashlib.sha256(f"{salt}:{item.response_id}".encode()).hexdigest()
        return digest, item.response_id

    selected = sorted(clean, key=rank)[:per_class]
    selected.extend(sorted(annotated, key=rank)[:per_class])
    # Final stable ordering makes serialized manifests reproducible as well.
    return tuple(sorted(selected, key=rank))


@dataclass(frozen=True)
class _QaSource:
    source_id: str
    source_name: str
    question: str
    passages: tuple[str, ...]


def _load_qa_sources(path: Path) -> dict[str, _QaSource]:
    sources: dict[str, _QaSource] = {}
    for line_number, row in _iter_json_objects(path):
        _require_fields(row, _REQUIRED_SOURCE_FIELDS, "source", line_number)
        if row["task_type"] != "QA":
            continue
        source_id = _nonempty_string(row["source_id"], "source.source_id", line_number)
        if source_id in sources:
            raise RagTruthFormatError(
                f"source line {line_number}: duplicate source id {source_id!r}"
            )
        info = row["source_info"]
        if not isinstance(info, Mapping):
            raise RagTruthFormatError(
                f"source line {line_number}: source_info must be an object"
            )
        if set(info) != {"question", "passages"}:
            raise RagTruthFormatError(
                f"source line {line_number}: source_info must contain exactly "
                "question and passages"
            )
        question = _nonempty_string(info["question"], "question", line_number)
        source_name = _nonempty_string(row["source"], "source.source", line_number)
        passages = parse_qa_passages(info["passages"], source_id=source_id)
        sources[source_id] = _QaSource(
            source_id=source_id,
            source_name=source_name,
            question=question,
            passages=passages,
        )
    if not sources:
        raise RagTruthFormatError(f"{path}: no QA sources were found")
    return sources


def _convert_response(
    row: Mapping[str, Any], line_number: int, source: _QaSource
) -> RagTruthExample:
    response_id = _nonempty_string(row["id"], "response.id", line_number)
    answer = _nonempty_string(row["response"], "response.response", line_number)
    model = _nonempty_string(row["model"], "response.model", line_number)
    labels = row["labels"]
    if not isinstance(labels, list):
        raise RagTruthFormatError(f"response line {line_number}: labels must be an array")

    annotations = tuple(
        _parse_annotation(value, answer, line_number, index)
        for index, value in enumerate(labels)
    )
    chunks = tuple(
        Chunk(
            chunk_id=f"ragtruth-{source.source_id}-passage-{index}",
            text=text,
            source=f"RAGTruth:{source.source_name}",
            metadata={
                "dataset": "RAGTruth",
                "upstream_source_id": source.source_id,
                "passage_index": index,
            },
        )
        for index, text in enumerate(source.passages, start=1)
    )
    request = VerificationRequest(
        request_id=f"ragtruth-response-{response_id}",
        answer=answer,
        chunks=chunks,
        query=source.question,
    )
    return RagTruthExample(
        response_id=response_id,
        source_id=source.source_id,
        model=model,
        split="test",
        quality="good",
        request=request,
        annotations=annotations,
    )


def _parse_annotation(
    value: Any, answer: str, line_number: int, index: int
) -> RagTruthAnnotation:
    if not isinstance(value, Mapping):
        raise RagTruthFormatError(
            f"response line {line_number}: labels[{index}] must be an object"
        )
    if set(value) != _REQUIRED_ANNOTATION_FIELDS:
        raise RagTruthFormatError(
            f"response line {line_number}: labels[{index}] has unexpected fields"
        )
    annotation = RagTruthAnnotation(
        start=value["start"],
        end=value["end"],
        text=value["text"],
        label_type=value["label_type"],
        meta=value["meta"],
        implicit_true=value["implicit_true"],
        due_to_null=value["due_to_null"],
    )
    if annotation.end > len(answer):
        raise RagTruthFormatError(
            f"response line {line_number}: labels[{index}] exceeds response length"
        )
    if answer[annotation.start : annotation.end] != annotation.text:
        raise RagTruthFormatError(
            f"response line {line_number}: labels[{index}] text does not match offsets"
        )
    return annotation


def _iter_json_objects(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise RagTruthFormatError(f"cannot open {path}: {type(exc).__name__}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RagTruthFormatError(
                    f"{path} line {line_number}: invalid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise RagTruthFormatError(
                    f"{path} line {line_number}: each row must be an object"
                )
            yield line_number, value


def _require_fields(
    row: Mapping[str, Any], required: frozenset[str], kind: str, line_number: int
) -> None:
    missing = required - set(row)
    if missing:
        raise RagTruthFormatError(
            f"{kind} line {line_number}: missing fields {sorted(missing)!r}"
        )


def _nonempty_string(value: Any, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagTruthFormatError(
            f"line {line_number}: {field} must be a non-empty string"
        )
    return value
