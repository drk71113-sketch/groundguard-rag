"""Frozen-result diagnostics for the RAGTruth smoke evaluation.

Every category produced here is an observable signal, not a causal claim.  The
module reads the existing case artifact and never reruns inference or changes
the registered sample, model, selector, or threshold.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


PAIR_TOKEN_LIMIT = 256
REVIEW_SALT = "ragtruth-error-review-v1"
_CLAUSE_MARKERS = re.compile(r",|;|\b(?:and|but|while|which|that)\b", re.IGNORECASE)
_REVIEW_LIMITS = {
    ("false_positive", "INSUFFICIENT_EVIDENCE"): 6,
    ("false_positive", "CONTRADICTED"): 3,
    ("false_positive", "CONFLICTING_EVIDENCE"): 3,
    ("true_positive", "*"): 4,
    ("true_negative", "*"): 4,
}
_MANUAL_REVIEW_FIELDS = frozenset(
    {
        "response_id",
        "claim_id",
        "outcome",
        "primary_observation_layer",
        "confidence",
        "note",
    }
)


def load_frozen_cases(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read the stage-3 JSONL artifact with minimal structural validation."""

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"cases line {line_number}: invalid JSON") from exc
            if not isinstance(case, dict):
                raise ValueError(f"cases line {line_number}: row must be an object")
            response_id = case.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                raise ValueError(
                    f"cases line {line_number}: response_id must be non-empty"
                )
            if response_id in seen_ids:
                raise ValueError(
                    f"cases line {line_number}: duplicate response_id {response_id!r}"
                )
            seen_ids.add(response_id)
            cases.append(case)
    if not cases:
        raise ValueError("frozen cases file must not be empty")
    return tuple(cases)


def analyze_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    pair_token_count: Callable[[str, str], int],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Return aggregate signal counts and one diagnostic row per claim."""

    rows: list[dict[str, Any]] = []
    for case in cases:
        chunks = {item["chunk_id"]: item["text"] for item in case["chunks"]}
        verdicts = {
            item["claim"]["claim_id"]: item
            for item in case["groundguard_output"]["audit_report"]["verdicts"]
        }
        annotations = case["ragtruth_annotations"]
        for comparison in case["claim_comparisons"]:
            verdict = verdicts[comparison["claim_id"]]
            row = _diagnostic_row(
                case,
                comparison,
                verdict,
                chunks,
                annotations,
                pair_token_count,
            )
            rows.append(row)

    summary = _aggregate(rows, response_count=len(cases))
    return summary, tuple(rows)


def select_review_queue(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Apply the pre-registered deterministic review strata."""

    selected: list[Mapping[str, Any]] = [
        row for row in rows if row["outcome"] == "false_negative"
    ]
    for (outcome, state), limit in _REVIEW_LIMITS.items():
        candidates = [
            row
            for row in rows
            if row["outcome"] == outcome and (state == "*" or row["state"] == state)
        ]
        selected.extend(sorted(candidates, key=_stable_rank)[:limit])

    unique = {(row["response_id"], row["claim_id"]): row for row in selected}
    return tuple(sorted((dict(row) for row in unique.values()), key=_stable_rank))


def write_analysis(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    review_queue: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "diagnostics.json", summary)
    _write_jsonl(destination / "claim_diagnostics.jsonl", rows)
    _write_jsonl(destination / "review_queue.jsonl", review_queue)


def load_and_validate_manual_review(
    path: str | Path, review_queue: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Require exactly one structured human observation per queued claim."""

    expected = {
        (row["response_id"], row["claim_id"]): row for row in review_queue
    }
    reviews: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manual review line {line_number}: invalid JSON") from exc
            if not isinstance(row, dict) or set(row) != _MANUAL_REVIEW_FIELDS:
                raise ValueError(
                    f"manual review line {line_number}: unexpected fields"
                )
            identity = (row["response_id"], row["claim_id"])
            if identity not in expected:
                raise ValueError(
                    f"manual review line {line_number}: claim is not in review queue"
                )
            if identity in seen:
                raise ValueError(
                    f"manual review line {line_number}: duplicate claim review"
                )
            seen.add(identity)
            if row["outcome"] != expected[identity]["outcome"]:
                raise ValueError(
                    f"manual review line {line_number}: outcome does not match queue"
                )
            if row["confidence"] not in {"low", "medium", "high"}:
                raise ValueError(
                    f"manual review line {line_number}: invalid confidence"
                )
            for field in ("primary_observation_layer", "note"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(
                        f"manual review line {line_number}: {field} must be non-empty"
                    )
            reviews.append(row)
    if seen != set(expected):
        missing = sorted(set(expected) - seen)
        raise ValueError(f"manual review is missing queued claims: {missing!r}")
    return tuple(reviews)


def build_local_pair_token_counter() -> Callable[[str, str], int]:
    """Load only the pinned tokenizer locally; no model or network is used."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "cross-encoder/nli-deberta-v3-xsmall",
        revision="a150876415327c80daeff35ca6f68f5ed8cf5c24",
        local_files_only=True,
        trust_remote_code=False,
    )

    def count(premise: str, hypothesis: str) -> int:
        encoded = tokenizer(
            premise,
            text_pair=hypothesis,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        return len(encoded["input_ids"])

    return count


def _diagnostic_row(
    case: Mapping[str, Any],
    comparison: Mapping[str, Any],
    verdict: Mapping[str, Any],
    chunks: Mapping[str, str],
    annotations: Sequence[Mapping[str, Any]],
    pair_token_count: Callable[[str, str], int],
) -> dict[str, Any]:
    claim_text = comparison["text"]
    overlapping = [
        annotations[index]
        for index in comparison["overlapping_annotation_indexes"]
    ]
    overlap_characters = sum(
        max(
            0,
            min(comparison["end"], item["end"])
            - max(comparison["start"], item["start"]),
        )
        for item in overlapping
    )
    annotation_ratio = min(1.0, overlap_characters / len(claim_text))

    edges: list[dict[str, Any]] = []
    for edge in verdict["evidence_assessments"]:
        chunk_id = edge["reference"]["chunk_id"]
        evidence_text = chunks[chunk_id]
        token_count = pair_token_count(evidence_text, claim_text)
        edges.append(
            {
                "chunk_id": chunk_id,
                "evidence_text": evidence_text,
                "relevance_score": edge["reference"]["relevance_score"],
                "pair_token_count": token_count,
                "exceeds_256_tokens": token_count > PAIR_TOKEN_LIMIT,
                "state": edge["state"],
                "label_scores": edge["label_scores"],
            }
        )

    max_scores = {
        label: max(
            (edge["label_scores"][label] for edge in edges), default=None
        )
        for label in ("supported", "contradicted", "insufficient")
    }
    clause_marker_count = len(_CLAUSE_MARKERS.findall(claim_text))
    signals: list[str] = []
    if not edges:
        signals.append("zero_evidence_edges")
    if edges and any(edge["exceeds_256_tokens"] for edge in edges):
        signals.append("some_selected_pairs_exceed_256_tokens")
    if edges and all(edge["exceeds_256_tokens"] for edge in edges):
        signals.append("all_selected_pairs_exceed_256_tokens")
    if comparison["gold_risk"] and annotation_ratio <= 0.25:
        signals.append("annotated_span_is_at_most_25pct_of_claim")
    if clause_marker_count >= 2:
        signals.append("multiple_clause_markers")
    supported = max_scores["supported"]
    if (
        comparison["outcome"] == "false_positive"
        and comparison["state"] == "INSUFFICIENT_EVIDENCE"
        and supported is not None
        and 0.5 <= supported < 0.8
    ):
        signals.append("max_support_between_0.5_and_0.8")

    return {
        "response_id": case["response_id"],
        "source_id": case["source_id"],
        "upstream_model": case["upstream_model"],
        "question": case["question"],
        "answer": case["answer"],
        "claim_id": comparison["claim_id"],
        "claim_text": claim_text,
        "claim_start": comparison["start"],
        "claim_end": comparison["end"],
        "state": comparison["state"],
        "gold_risk": comparison["gold_risk"],
        "predicted_risk": comparison["predicted_risk"],
        "outcome": comparison["outcome"],
        "annotation_to_claim_ratio": annotation_ratio,
        "clause_marker_count": clause_marker_count,
        "ragtruth_annotations": overlapping,
        "label_types": sorted({item["label_type"] for item in overlapping}),
        "has_implicit_true_annotation": any(
            item["implicit_true"] for item in overlapping
        ),
        "has_due_to_null_annotation": any(item["due_to_null"] for item in overlapping),
        "evidence_edge_count": len(edges),
        "max_label_scores": max_scores,
        "signals": signals,
        "selected_evidence": edges,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], *, response_count: int) -> dict[str, Any]:
    error_rows = [
        row for row in rows if row["outcome"] in {"false_positive", "false_negative"}
    ]
    signal_counts = Counter(
        signal for row in error_rows for signal in row["signals"]
    )
    return {
        "analysis": "ragtruth-smoke-v1-frozen-error-analysis",
        "scope": (
            "Observable diagnostic signals over frozen stage-3 results; counts "
            "are not proven causal attribution."
        ),
        "response_count": response_count,
        "claim_count": len(rows),
        "error_claim_count": len(error_rows),
        "outcome_state_counts": _nested_counts(
            (row["outcome"], row["state"]) for row in rows
        ),
        "error_signal_counts": dict(sorted(signal_counts.items())),
        "false_positive_edge_count_distribution": dict(
            sorted(
                Counter(
                    str(row["evidence_edge_count"])
                    for row in rows
                    if row["outcome"] == "false_positive"
                ).items()
            )
        ),
        "false_negative_label_type_counts": dict(
            sorted(
                Counter(
                    label
                    for row in rows
                    if row["outcome"] == "false_negative"
                    for label in row["label_types"]
                ).items()
            )
        ),
    }


def _nested_counts(values: Iterable[tuple[str, str]]) -> dict[str, dict[str, int]]:
    counts = Counter(values)
    result: dict[str, dict[str, int]] = {}
    for (outer, inner), count in sorted(counts.items()):
        result.setdefault(outer, {})[inner] = count
    return result


def _stable_rank(row: Mapping[str, Any]) -> tuple[str, str, str]:
    identity = f"{row['response_id']}:{row['claim_id']}"
    digest = hashlib.sha256(f"{REVIEW_SALT}:{identity}".encode()).hexdigest()
    return digest, row["response_id"], row["claim_id"]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> None:
    demo_root = Path(__file__).resolve().parents[2]
    output = demo_root / "results" / "samples" / "ragtruth_smoke_v1"
    cases = load_frozen_cases(output / "cases.jsonl")
    summary, rows = analyze_cases(
        cases, pair_token_count=build_local_pair_token_counter()
    )
    review_queue = select_review_queue(rows)
    manual_path = output / "manual_review.jsonl"
    if manual_path.is_file():
        manual_review = load_and_validate_manual_review(manual_path, review_queue)
        summary["manual_review"] = {
            "count": len(manual_review),
            "primary_observation_layer_counts": dict(
                sorted(
                    Counter(
                        row["primary_observation_layer"] for row in manual_review
                    ).items()
                )
            ),
            "confidence_counts": dict(
                sorted(Counter(row["confidence"] for row in manual_review).items())
            ),
        }
    write_analysis(summary, rows, review_queue, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"review_queue_count={len(review_queue)}")


if __name__ == "__main__":
    main()
