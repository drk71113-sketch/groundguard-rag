"""Deterministic five-state and calibration evaluation on labeled requests.

The evaluator reads only caller-provided data and invokes ``GroundGuard.verify``.
It does not download a dataset, treat synthetic fixtures as evidence of model
quality, or reinterpret ``INSUFFICIENT_EVIDENCE`` as real-world falsity.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groundguard_rag.api import GroundGuard
from groundguard_rag.domain.enums import VerificationState
from groundguard_rag.domain.exceptions import DomainValidationError, GroundGuardError
from groundguard_rag.domain.models import VerificationRequest
from groundguard_rag.integrations.serialization import request_from_json


BENCHMARK_SCHEMA_VERSION = "1.0.0"
_STATES = tuple(VerificationState)
_CASE_FIELDS = frozenset({"case_id", "request", "expected_states"})
_REQUEST_FIELDS = frozenset({"request_id", "answer", "chunks", "query"})


class DatasetFormatError(GroundGuardError):
    """A labeled JSONL row does not satisfy the benchmark input contract."""


class EvaluationContractError(GroundGuardError):
    """A guard output cannot be aligned with the supplied ground truth."""


@dataclass(frozen=True)
class BenchmarkCase:
    """One request and the expected state of each decomposed claim, in order."""

    case_id: str
    request: VerificationRequest
    expected_states: tuple[VerificationState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise DomainValidationError("BenchmarkCase.case_id must be non-empty")
        if not isinstance(self.request, VerificationRequest):
            raise DomainValidationError(
                "BenchmarkCase.request must be a VerificationRequest"
            )
        states = tuple(self.expected_states)
        if not states or any(not isinstance(item, VerificationState) for item in states):
            raise DomainValidationError(
                "BenchmarkCase.expected_states must contain VerificationState members"
            )
        object.__setattr__(self, "expected_states", states)


@dataclass(frozen=True)
class CalibrationMetrics:
    """Correctness calibration metrics for claims that carry confidence."""

    count: int
    coverage: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    bins: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "coverage": self.coverage,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "expected_calibration_error": self.expected_calibration_error,
            "bins": self.bins,
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregate verification metrics with a complete five-state matrix."""

    schema_version: str
    case_count: int
    claim_count: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: Mapping[str, Mapping[str, int]]
    per_state: Mapping[str, Mapping[str, float | int]]
    calibration: CalibrationMetrics | None
    total_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_count": self.case_count,
            "claim_count": self.claim_count,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "confusion_matrix": {
                expected: dict(predicted)
                for expected, predicted in self.confusion_matrix.items()
            },
            "per_state": {
                state: dict(metrics) for state, metrics in self.per_state.items()
            },
            "calibration": (
                None if self.calibration is None else self.calibration.to_dict()
            ),
            "total_latency_ms": self.total_latency_ms,
        }


def evaluate_verification(
    guard: GroundGuard,
    cases: Sequence[BenchmarkCase],
    *,
    calibration_bins: int = 10,
    monotonic: Callable[[], float] = time.perf_counter,
) -> BenchmarkReport:
    """Evaluate one explicitly configured guard without network side effects."""

    if not isinstance(guard, GroundGuard):
        raise DomainValidationError("guard must be a GroundGuard")
    if isinstance(cases, (str, bytes, bytearray)) or not isinstance(cases, Sequence):
        raise DomainValidationError("cases must be a sequence of BenchmarkCase")
    if not cases:
        raise DomainValidationError("cases must not be empty")
    if isinstance(calibration_bins, bool) or not isinstance(calibration_bins, int):
        raise DomainValidationError("calibration_bins must be an integer")
    if calibration_bins < 1:
        raise DomainValidationError("calibration_bins must be >= 1")
    if not callable(monotonic):
        raise DomainValidationError("monotonic must be callable")

    matrix = {
        expected.value: {predicted.value: 0 for predicted in _STATES}
        for expected in _STATES
    }
    seen_case_ids: set[str] = set()
    confidence_labels: list[tuple[float, int]] = []
    claim_count = 0
    correct_count = 0
    started = _read_monotonic(monotonic)

    for case in cases:
        if not isinstance(case, BenchmarkCase):
            raise DomainValidationError(
                "cases must contain only BenchmarkCase instances"
            )
        if case.case_id in seen_case_ids:
            raise DomainValidationError(f"duplicate benchmark case_id {case.case_id!r}")
        seen_case_ids.add(case.case_id)
        output = guard.verify(case.request)
        verdicts = output.audit_report.verdicts
        if len(verdicts) != len(case.expected_states):
            raise EvaluationContractError(
                f"case {case.case_id!r} expected {len(case.expected_states)} claims "
                f"but verifier produced {len(verdicts)}"
            )
        for expected, verdict in zip(case.expected_states, verdicts, strict=True):
            predicted = verdict.state
            matrix[expected.value][predicted.value] += 1
            is_correct = int(predicted is expected)
            correct_count += is_correct
            claim_count += 1
            if verdict.calibrated_confidence is not None:
                confidence_labels.append(
                    (verdict.calibrated_confidence, is_correct)
                )

    elapsed_ms = (_read_monotonic(monotonic) - started) * 1000.0
    if elapsed_ms < 0:
        raise EvaluationContractError("monotonic clock moved backwards")

    per_state: dict[str, dict[str, float | int]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for state in _STATES:
        name = state.value
        true_positive = matrix[name][name]
        predicted_total = sum(matrix[row.value][name] for row in _STATES)
        expected_total = sum(matrix[name].values())
        precision = true_positive / predicted_total if predicted_total else 0.0
        recall = true_positive / expected_total if expected_total else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_state[name] = {
            "support": expected_total,
            "predicted": predicted_total,
            "true_positive": true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    calibration = _calibration_metrics(
        confidence_labels,
        total_claims=claim_count,
        bins=calibration_bins,
    )
    state_count = len(_STATES)
    return BenchmarkReport(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        case_count=len(cases),
        claim_count=claim_count,
        accuracy=correct_count / claim_count,
        macro_precision=sum(precision_values) / state_count,
        macro_recall=sum(recall_values) / state_count,
        macro_f1=sum(f1_values) / state_count,
        confusion_matrix=matrix,
        per_state=per_state,
        calibration=calibration,
        total_latency_ms=elapsed_ms,
    )


def _calibration_metrics(
    values: Sequence[tuple[float, int]], *, total_claims: int, bins: int
) -> CalibrationMetrics | None:
    if not values:
        return None
    epsilon = 1e-15
    brier = sum((confidence - label) ** 2 for confidence, label in values) / len(
        values
    )
    log_loss = -sum(
        label * math.log(min(max(confidence, epsilon), 1.0 - epsilon))
        + (1 - label)
        * math.log(min(max(1.0 - confidence, epsilon), 1.0 - epsilon))
        for confidence, label in values
    ) / len(values)

    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            item
            for item in values
            if lower <= item[0] <= upper
            and (index == bins - 1 or item[0] < upper)
        ]
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        ece += len(bucket) / len(values) * abs(mean_confidence - accuracy)

    return CalibrationMetrics(
        count=len(values),
        coverage=len(values) / total_claims,
        brier_score=brier,
        log_loss=log_loss,
        expected_calibration_error=ece,
        bins=bins,
    )


def _read_monotonic(clock: Callable[[], float]) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationContractError("monotonic must return a real number")
    if not math.isfinite(value):
        raise EvaluationContractError("monotonic must return a finite number")
    return float(value)


def load_jsonl_dataset(path: str | Path) -> tuple[BenchmarkCase, ...]:
    """Load strict local JSONL; no URL or implicit dataset resolution is allowed."""

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetFormatError("dataset path must identify a local file")
    cases: list[BenchmarkCase] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
                cases.append(_case_from_document(document))
            except (TypeError, ValueError, KeyError, DomainValidationError) as exc:
                raise DatasetFormatError(
                    f"invalid benchmark row at line {line_number}: {type(exc).__name__}"
                ) from exc
    if not cases:
        raise DatasetFormatError("dataset must contain at least one non-empty row")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise DatasetFormatError("dataset contains duplicate case_id values")
    return tuple(cases)


def _case_from_document(document: Any) -> BenchmarkCase:
    if not isinstance(document, Mapping):
        raise TypeError("row must be an object")
    if set(document) != _CASE_FIELDS:
        raise KeyError("row fields mismatch")
    request_document = document["request"]
    if not isinstance(request_document, Mapping):
        raise TypeError("request must be an object")
    if set(request_document) - _REQUEST_FIELDS:
        raise KeyError("request contains unknown fields")
    missing = {"request_id", "answer", "chunks"} - set(request_document)
    if missing:
        raise KeyError("request fields missing")
    states_document = document["expected_states"]
    if not isinstance(states_document, list) or not states_document:
        raise TypeError("expected_states must be a non-empty array")
    states = tuple(VerificationState(item) for item in states_document)
    request = request_from_json(
        request_id=request_document["request_id"],
        answer=request_document["answer"],
        chunks=request_document["chunks"],
        query=request_document.get("query"),
    )
    return BenchmarkCase(
        case_id=document["case_id"], request=request, expected_states=states
    )
