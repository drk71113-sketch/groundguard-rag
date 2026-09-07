from __future__ import annotations

import json

from groundguard_rag.evaluation.cli import main


def test_evaluation_cli_prints_json_report(capsys):
    main(
        [
            "--factory",
            "examples.demo_factory:build_groundguard",
            "--dataset",
            "examples/sample_benchmark.jsonl",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["case_count"] == 2
    assert payload["accuracy"] == 1.0
