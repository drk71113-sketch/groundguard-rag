"""Dataset adapters used by the reference application."""

from demo_rag.app.datasets.ragtruth import (
    RagTruthAnnotation,
    RagTruthExample,
    RagTruthFormatError,
    load_qa_test_examples,
    parse_qa_passages,
    select_smoke_subset,
)

__all__ = [
    "RagTruthAnnotation",
    "RagTruthExample",
    "RagTruthFormatError",
    "load_qa_test_examples",
    "parse_qa_passages",
    "select_smoke_subset",
]
