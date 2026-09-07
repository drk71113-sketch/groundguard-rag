import pytest

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import Chunk, VerificationRequest


def test_valid_request_constructs():
    request = VerificationRequest(
        request_id="req-1",
        answer="Paris is the capital of France.",
        chunks=(Chunk(chunk_id="c1", text="Paris is the capital of France."),),
    )
    assert request.query is None
    assert len(request.chunks) == 1


def test_empty_chunks_is_a_valid_input():
    # No chunks is a legitimate input (later stages resolve this to
    # INSUFFICIENT_EVIDENCE for every claim) -- it must not be a
    # construction-time error.
    request = VerificationRequest(request_id="req-1", answer="hello", chunks=())
    assert request.chunks == ()


def test_chunks_list_is_coerced_to_tuple():
    request = VerificationRequest(
        request_id="req-1",
        answer="hello",
        chunks=[Chunk(chunk_id="c1", text="hello")],
    )
    assert isinstance(request.chunks, tuple)


def test_query_is_optional_and_defaults_to_none():
    request = VerificationRequest(request_id="req-1", answer="hello", chunks=())
    assert request.query is None


def test_query_can_be_supplied_for_future_heal_mode_use():
    request = VerificationRequest(
        request_id="req-1", answer="hello", chunks=(), query="who said hello?"
    )
    assert request.query == "who said hello?"


@pytest.mark.parametrize("request_id", ["", "   "])
def test_empty_request_id_rejected(request_id):
    with pytest.raises(DomainValidationError):
        VerificationRequest(request_id=request_id, answer="hello", chunks=())


@pytest.mark.parametrize("answer", ["", "   ", "\t\r\n"])
def test_empty_or_whitespace_only_answer_rejected(answer):
    with pytest.raises(DomainValidationError):
        VerificationRequest(request_id="req-1", answer=answer, chunks=())


@pytest.mark.parametrize("answer", [None, 123, ["not", "text"]])
def test_non_string_answer_rejected(answer):
    with pytest.raises(DomainValidationError):
        VerificationRequest(request_id="req-1", answer=answer, chunks=())


def test_non_chunk_element_in_chunks_rejected():
    with pytest.raises(DomainValidationError):
        VerificationRequest(request_id="req-1", answer="hello", chunks=[{"chunk_id": "c1", "text": "hi"}])


def test_duplicate_chunk_id_rejected():
    with pytest.raises(DomainValidationError):
        VerificationRequest(
            request_id="req-1",
            answer="hello",
            chunks=[
                Chunk(chunk_id="c1", text="first"),
                Chunk(chunk_id="c1", text="second"),
            ],
        )


def test_distinct_chunk_ids_accepted():
    request = VerificationRequest(
        request_id="req-1",
        answer="hello",
        chunks=[Chunk(chunk_id="c1", text="first"), Chunk(chunk_id="c2", text="second")],
    )
    assert len(request.chunks) == 2


@pytest.mark.parametrize("bad_query", [123, 4.5, ["not", "a", "string"], {}])
def test_non_string_query_rejected(bad_query):
    with pytest.raises(DomainValidationError):
        VerificationRequest(request_id="req-1", answer="hello", chunks=(), query=bad_query)
