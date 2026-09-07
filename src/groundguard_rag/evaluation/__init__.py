"""Offline evaluation helpers for caller-supplied labeled datasets."""

from groundguard_rag.evaluation.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkCase,
    BenchmarkReport,
    CalibrationMetrics,
    DatasetFormatError,
    EvaluationContractError,
    evaluate_verification,
    load_jsonl_dataset,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkCase",
    "BenchmarkReport",
    "CalibrationMetrics",
    "DatasetFormatError",
    "EvaluationContractError",
    "evaluate_verification",
    "load_jsonl_dataset",
]
