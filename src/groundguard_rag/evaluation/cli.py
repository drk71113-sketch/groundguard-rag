"""Command-line entry point for local JSONL verification evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from groundguard_rag.evaluation.benchmark import (
    evaluate_verification,
    load_jsonl_dataset,
)
from groundguard_rag.integrations.factory import load_guard


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GroundGuard-RAG on a caller-supplied JSONL dataset"
    )
    parser.add_argument("--factory", required=True, help="module:attribute guard factory")
    parser.add_argument("--dataset", required=True, help="local labeled JSONL file")
    parser.add_argument("--output", help="optional JSON output path; defaults to stdout")
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args(argv)

    report = evaluate_verification(
        load_guard(args.factory),
        load_jsonl_dataset(args.dataset),
        calibration_bins=args.calibration_bins,
    )
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
