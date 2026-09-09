from __future__ import annotations

import ast
import importlib.resources
import subprocess
import sys
import textwrap
from pathlib import Path

import groundguard_rag


def test_public_import_is_lightweight_and_exports_stable_api():
    assert groundguard_rag.__version__ == "0.1.0"
    assert callable(groundguard_rag.verify)
    assert callable(groundguard_rag.heal)
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, groundguard_rag; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'mcp' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr


def test_audit_schema_is_available_as_package_data():
    schema = (
        importlib.resources.files("groundguard_rag.schema")
        .joinpath("audit_report.v1.schema.json")
        .read_text(encoding="utf-8")
    )
    assert '"$schema"' in schema
    assert '"schema_version"' in schema


def test_application_code_has_no_optimization_stripped_assert_statements():
    application_root = (
        Path(__file__).parents[2] / "src" / "groundguard_rag" / "application"
    )
    occurrences = []
    for path in application_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        occurrences.extend(
            f"{path.relative_to(application_root)}:{node.lineno}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Assert)
        )

    assert occurrences == []


def test_heal_invariant_guard_survives_python_optimized_mode():
    script = textwrap.dedent(
        """
        from types import SimpleNamespace

        from groundguard_rag.application.exceptions import ApplicationInvariantError
        from groundguard_rag.application.heal_service import HealService
        from groundguard_rag.domain.enums import RepairAction, VerificationState
        from groundguard_rag.domain.models import (
            AtomicClaim,
            ClaimVerdict,
            RepairDecision,
            VerificationRequest,
        )

        claim = AtomicClaim("c1", "claim", 0, 5)
        verdict = ClaimVerdict(
            claim=claim,
            state=VerificationState.INSUFFICIENT_EVIDENCE,
            evidence_assessments=(),
        )
        request = VerificationRequest("r1", "claim", (), "query")
        decision = RepairDecision(
            action=RepairAction.RETRIEVE,
            estimated_cost=0.0,
        )
        try:
            HealService._build_candidate(
                SimpleNamespace(_retriever=None),
                decision=decision,
                target=verdict,
                current_request=request,
                stable_ids=("c1",),
            )
        except ApplicationInvariantError:
            pass
        else:
            raise SystemExit("optimized mode removed the invariant guard")
        """
    )
    check = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert check.returncode == 0, check.stderr or check.stdout


def test_ci_correctness_lint_covers_reference_demo_code():
    repository_root = Path(__file__).parents[2]
    workflow = (repository_root / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert "ruff check src tests examples demo_rag" in workflow
