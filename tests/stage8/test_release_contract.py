from __future__ import annotations

import importlib.resources
import subprocess
import sys

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
