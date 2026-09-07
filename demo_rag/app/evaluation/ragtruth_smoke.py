"""Run the fixed RAGTruth QA smoke subset against a local GroundGuard.

The evaluation is deliberately binary and claim-aligned.  RAGTruth annotates
hallucination spans; it does not provide GroundGuard's complete five-state gold
labels.  Consequently this module reports only risk-detection metrics and keeps
the predicted five-state values available for error analysis.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from demo_rag.app.datasets.ragtruth import (
    RagTruthAnnotation,
    RagTruthExample,
    load_qa_test_examples,
    select_smoke_subset,
)
from groundguard_rag import GroundGuard, VerifyConfig
from groundguard_rag.adapters.decomposition import RuleBasedClaimDecomposer
from groundguard_rag.adapters.evidence_selection import (
    LexicalEvidenceSelector,
    LexicalSelectorConfig,
)
from groundguard_rag.adapters.verification import (
    NliLabelMapping,
    NliVerifier,
    NliVerifierConfig,
    RuleBasedCheckabilityVerifier,
    TransformersNliBackend,
)
from groundguard_rag.application.verify_service import VerifyService
from groundguard_rag.domain.enums import VerificationState


MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
MODEL_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
SMOKE_SALT = "groundguard-ragtruth-smoke-v1"
DECISION_THRESHOLD = 0.8
RISK_STATES = frozenset(
    {
        VerificationState.CONTRADICTED,
        VerificationState.INSUFFICIENT_EVIDENCE,
        VerificationState.CONFLICTING_EVIDENCE,
    }
)


@dataclass(frozen=True)
class PredictedClaim:
    """Minimal GroundGuard claim prediction used for span alignment."""

    claim_id: str
    text: str
    start: int
    end: int
    state: VerificationState


@dataclass(frozen=True)
class ClaimComparison:
    """One claim-level binary comparison against RAGTruth span annotations."""

    claim: PredictedClaim
    gold_risk: bool
    predicted_risk: bool
    outcome: str
    overlapping_annotation_indexes: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim.claim_id,
            "text": self.claim.text,
            "start": self.claim.start,
            "end": self.claim.end,
            "state": self.claim.state.value,
            "gold_risk": self.gold_risk,
            "predicted_risk": self.predicted_risk,
            "outcome": self.outcome,
            "overlapping_annotation_indexes": list(
                self.overlapping_annotation_indexes
            ),
        }


@dataclass(frozen=True)
class BinaryMetrics:
    """Transparent binary confusion counts and derived metrics."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def count(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def accuracy(self) -> float:
        return (
            (self.true_positive + self.true_negative) / self.count
            if self.count
            else 0.0
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class RagTruthSmokeReport:
    """Aggregate metrics plus serializable per-response evidence."""

    claim_metrics: BinaryMetrics
    response_metrics: BinaryMetrics
    state_counts: dict[str, int]
    model_counts: dict[str, int]
    cases: tuple[dict[str, Any], ...]
    total_latency_ms: float

    def summary_dict(self) -> dict[str, Any]:
        return {
            "evaluation": "ragtruth-qa-smoke-v1",
            "scope": (
                "Fixed 50-response QA/test/good smoke subset; binary grounding-risk "
                "detection only, not five-state gold evaluation."
            ),
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "decision_threshold": DECISION_THRESHOLD,
                "local_files_only": True,
                "calibrated_confidence": "not_provided",
            },
            "response_count": len(self.cases),
            "claim_metrics": self.claim_metrics.to_dict(),
            "response_metrics": self.response_metrics.to_dict(),
            "state_counts": self.state_counts,
            "model_counts": self.model_counts,
            "total_latency_ms": self.total_latency_ms,
        }


def build_local_groundguard() -> GroundGuard:
    """Assemble the pre-registered offline baseline without hidden providers."""

    backend = TransformersNliBackend.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        label_mapping=NliLabelMapping(
            entailment_index=1,
            contradiction_index=0,
            neutral_index=2,
        ),
        local_files_only=True,
        trust_remote_code=False,
        batch_size=16,
        max_length=256,
        device="cpu",
    )
    verifier = RuleBasedCheckabilityVerifier(
        NliVerifier(
            backend,
            NliVerifierConfig(
                verifier_id="ragtruth-smoke-transformers-nli",
                verifier_revision=f"{MODEL_ID}@{MODEL_REVISION}",
                decision_threshold=DECISION_THRESHOLD,
            ),
        )
    )
    service = VerifyService(
        decomposer=RuleBasedClaimDecomposer(),
        selector=LexicalEvidenceSelector(
            LexicalSelectorConfig(top_k=3, min_score=0.1)
        ),
        verifier=verifier,
        config=VerifyConfig(allow_network=False),
        model_revision=f"{MODEL_ID}@{MODEL_REVISION}",
        threshold_version="ragtruth-smoke-v1/threshold-0.8",
        clock=lambda: datetime.now(timezone.utc).isoformat(),
    )
    return GroundGuard(verify_service=service)


def align_claims(
    predictions: Sequence[PredictedClaim],
    annotations: Sequence[RagTruthAnnotation],
) -> tuple[ClaimComparison, ...]:
    """Align human spans to claim spans using half-open interval overlap."""

    covered_annotation_indexes: set[int] = set()
    comparisons: list[ClaimComparison] = []
    for claim in predictions:
        overlapping = tuple(
            index
            for index, annotation in enumerate(annotations)
            if claim.start < annotation.end and annotation.start < claim.end
        )
        covered_annotation_indexes.update(overlapping)
        gold_risk = bool(overlapping)
        predicted_risk = claim.state in RISK_STATES
        comparisons.append(
            ClaimComparison(
                claim=claim,
                gold_risk=gold_risk,
                predicted_risk=predicted_risk,
                outcome=_outcome(gold_risk, predicted_risk),
                overlapping_annotation_indexes=overlapping,
            )
        )

    expected_indexes = set(range(len(annotations)))
    if covered_annotation_indexes != expected_indexes:
        missing = sorted(expected_indexes - covered_annotation_indexes)
        raise ValueError(
            "RAGTruth annotations were not covered by any decomposed claim: "
            f"{missing!r}"
        )
    return tuple(comparisons)


def evaluate_smoke_subset(
    guard: GroundGuard,
    examples: Sequence[RagTruthExample],
    *,
    progress: Callable[[int, int, str], None] | None = None,
    monotonic: Callable[[], float] = time.perf_counter,
) -> RagTruthSmokeReport:
    """Verify a preselected subset and retain every auditable case outcome."""

    claim_outcomes: list[str] = []
    response_outcomes: list[str] = []
    cases: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    started = monotonic()

    for position, example in enumerate(examples, start=1):
        output = guard.verify(example.request)
        predictions = tuple(
            PredictedClaim(
                claim_id=verdict.claim.claim_id,
                text=verdict.claim.text,
                start=verdict.claim.start_char,
                end=verdict.claim.end_char,
                state=verdict.state,
            )
            for verdict in output.audit_report.verdicts
        )
        comparisons = align_claims(predictions, example.annotations)
        claim_outcomes.extend(item.outcome for item in comparisons)
        state_counts.update(item.claim.state.value for item in comparisons)
        model_counts[example.model] += 1

        gold_response_risk = example.has_hallucination
        predicted_response_risk = any(item.predicted_risk for item in comparisons)
        response_outcome = _outcome(gold_response_risk, predicted_response_risk)
        response_outcomes.append(response_outcome)
        cases.append(
            _case_dict(
                example,
                comparisons,
                response_outcome,
                output.to_dict(),
            )
        )
        if progress is not None:
            progress(position, len(examples), example.response_id)

    elapsed_ms = (monotonic() - started) * 1000.0
    return RagTruthSmokeReport(
        claim_metrics=_metrics(claim_outcomes),
        response_metrics=_metrics(response_outcomes),
        state_counts=dict(sorted(state_counts.items())),
        model_counts=dict(sorted(model_counts.items())),
        cases=tuple(cases),
        total_latency_ms=elapsed_ms,
    )


def write_report(report: RagTruthSmokeReport, output_dir: str | Path) -> None:
    """Write public, relative-path-safe JSON artifacts for manual inspection."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(report.summary_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (destination / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in report.cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _case_dict(
    example: RagTruthExample,
    comparisons: Sequence[ClaimComparison],
    response_outcome: str,
    groundguard_output: dict[str, Any],
) -> dict[str, Any]:
    return {
        "response_id": example.response_id,
        "source_id": example.source_id,
        "upstream_model": example.model,
        "question": example.request.query,
        "answer": example.request.answer,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "metadata": dict(chunk.metadata),
            }
            for chunk in example.request.chunks
        ],
        "ragtruth_annotations": [
            {
                "start": item.start,
                "end": item.end,
                "text": item.text,
                "label_type": item.label_type,
                "meta": item.meta,
                "implicit_true": item.implicit_true,
                "due_to_null": item.due_to_null,
            }
            for item in example.annotations
        ],
        "response_outcome": response_outcome,
        "claim_comparisons": [item.to_dict() for item in comparisons],
        "groundguard_output": groundguard_output,
    }


def _outcome(gold_risk: bool, predicted_risk: bool) -> str:
    if gold_risk and predicted_risk:
        return "true_positive"
    if not gold_risk and predicted_risk:
        return "false_positive"
    if gold_risk and not predicted_risk:
        return "false_negative"
    return "true_negative"


def _metrics(outcomes: Iterable[str]) -> BinaryMetrics:
    counts = Counter(outcomes)
    return BinaryMetrics(
        true_positive=counts["true_positive"],
        false_positive=counts["false_positive"],
        false_negative=counts["false_negative"],
        true_negative=counts["true_negative"],
    )


def default_demo_root() -> Path:
    """Return the ``demo_rag`` directory independently of the current cwd."""

    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = default_demo_root()
    raw = root / "data" / "raw" / "ragtruth"
    output = root / "results" / "samples" / "ragtruth_smoke_v1"
    examples = load_qa_test_examples(
        raw / "source_info.jsonl", raw / "response.jsonl"
    )
    subset = select_smoke_subset(examples, per_class=25, salt=SMOKE_SALT)
    guard = build_local_groundguard()

    def show_progress(position: int, total: int, response_id: str) -> None:
        if position == 1 or position % 5 == 0 or position == total:
            print(f"verified {position}/{total} response_id={response_id}", flush=True)

    report = evaluate_smoke_subset(guard, subset, progress=show_progress)
    write_report(report, output)
    print(json.dumps(report.summary_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
