import pytest

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import Chunk, EvidenceCandidate, EvidenceReference


def test_whole_chunk_candidate_constructs():
    candidate = EvidenceCandidate(
        reference=EvidenceReference(chunk_id="c1"),
        text="Paris is the capital of France.",
    )
    assert candidate.text == "Paris is the capital of France."


def test_span_candidate_constructs_with_matching_length():
    ref = EvidenceReference(chunk_id="c1", start_char=0, end_char=5)
    candidate = EvidenceCandidate(reference=ref, text="Paris")
    assert candidate.text == "Paris"


def test_span_candidate_with_mismatched_text_length_rejected():
    ref = EvidenceReference(chunk_id="c1", start_char=0, end_char=5)
    with pytest.raises(DomainValidationError):
        EvidenceCandidate(reference=ref, text="Paris is the capital")


def test_empty_text_rejected():
    with pytest.raises(DomainValidationError):
        EvidenceCandidate(reference=EvidenceReference(chunk_id="c1"), text="")


def test_reference_must_be_evidence_reference_instance():
    with pytest.raises(DomainValidationError):
        EvidenceCandidate(reference="c1", text="hi")


def test_from_chunk_whole_chunk_reference_slices_full_text():
    chunk = Chunk(chunk_id="c1", text="Paris is the capital of France.")
    ref = EvidenceReference(chunk_id="c1")
    candidate = EvidenceCandidate.from_chunk(ref, chunk)
    assert candidate.text == chunk.text


def test_from_chunk_span_reference_slices_exact_substring():
    chunk = Chunk(chunk_id="c1", text="Paris is the capital of France.")
    ref = EvidenceReference(chunk_id="c1", start_char=0, end_char=5)
    candidate = EvidenceCandidate.from_chunk(ref, chunk)
    assert candidate.text == "Paris"


def test_from_chunk_rejects_mismatched_chunk_id():
    # This is the concrete mechanism that replaces "look up chunk_id in a
    # hidden global store": the caller must already hold the matching
    # Chunk, and a mismatch is caught explicitly rather than silently
    # resolved against something else.
    chunk = Chunk(chunk_id="c1", text="hello")
    ref = EvidenceReference(chunk_id="different-chunk")
    with pytest.raises(DomainValidationError):
        EvidenceCandidate.from_chunk(ref, chunk)
