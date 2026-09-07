import re

from groundguard_rag.application.input_hashing import compute_input_hash
from groundguard_rag.domain.models import Chunk, VerificationRequest

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _chunk(chunk_id="c1", text="hello world", source="doc.md", metadata=None):
    return Chunk(chunk_id=chunk_id, text=text, source=source, metadata=metadata or {})


def test_hash_has_expected_format():
    digest = compute_input_hash("hello", [_chunk()])
    assert _HASH_RE.match(digest)


def test_hash_is_deterministic_for_identical_input():
    chunks = [_chunk("c1", "text one"), _chunk("c2", "text two")]
    first = compute_input_hash("answer text", chunks)
    second = compute_input_hash("answer text", list(chunks))
    assert first == second


def test_hash_unaffected_by_metadata_key_insertion_order():
    chunk_a = _chunk(metadata={"page": 1, "author": "x"})
    chunk_b = _chunk(metadata={"author": "x", "page": 1})
    assert compute_input_hash("answer", [chunk_a]) == compute_input_hash("answer", [chunk_b])


def test_hash_unaffected_by_nested_metadata_key_insertion_order():
    chunk_a = _chunk(metadata={"nested": {"a": 1, "b": 2}})
    chunk_b = _chunk(metadata={"nested": {"b": 2, "a": 1}})
    assert compute_input_hash("answer", [chunk_a]) == compute_input_hash("answer", [chunk_b])


def test_hash_changes_when_answer_changes():
    chunks = [_chunk()]
    assert compute_input_hash("answer one", chunks) != compute_input_hash("answer two", chunks)


def test_hash_changes_when_chunk_text_changes():
    a = compute_input_hash("answer", [_chunk(text="original")])
    b = compute_input_hash("answer", [_chunk(text="changed")])
    assert a != b


def test_hash_changes_when_chunk_order_changes():
    # RAG context ordering can affect what a verifier concludes, so order
    # is part of what verify actually consumed.
    chunk_a = _chunk("c1", "first")
    chunk_b = _chunk("c2", "second")
    forward = compute_input_hash("answer", [chunk_a, chunk_b])
    backward = compute_input_hash("answer", [chunk_b, chunk_a])
    assert forward != backward


def test_hash_changes_when_metadata_value_changes():
    a = compute_input_hash("answer", [_chunk(metadata={"page": 1})])
    b = compute_input_hash("answer", [_chunk(metadata={"page": 2})])
    assert a != b


def test_hash_changes_when_source_changes():
    a = compute_input_hash("answer", [_chunk(source="doc-a.md")])
    b = compute_input_hash("answer", [_chunk(source="doc-b.md")])
    assert a != b


def test_hash_changes_when_chunk_id_changes():
    a = compute_input_hash("answer", [_chunk(chunk_id="c1")])
    b = compute_input_hash("answer", [_chunk(chunk_id="c2")])
    assert a != b


def test_hash_ignores_query_and_request_id():
    # compute_input_hash structurally cannot see query/request_id (they
    # are not parameters at all) -- demonstrated here via two
    # VerificationRequest objects that differ only in those two fields.
    chunks = (_chunk(),)
    request_a = VerificationRequest(
        request_id="req-a", answer="same answer", chunks=chunks, query="query one"
    )
    request_b = VerificationRequest(
        request_id="req-b", answer="same answer", chunks=chunks, query="query two"
    )
    hash_a = compute_input_hash(request_a.answer, request_a.chunks)
    hash_b = compute_input_hash(request_b.answer, request_b.chunks)
    assert hash_a == hash_b


def test_empty_chunks_hash_is_well_formed_and_deterministic():
    first = compute_input_hash("answer only", [])
    second = compute_input_hash("answer only", [])
    assert first == second
    assert _HASH_RE.match(first)


def test_unicode_answer_and_metadata_hash_deterministically():
    chunk = _chunk(text="世界你好", metadata={"标签": "测试"})
    first = compute_input_hash("你好，世界", [chunk])
    second = compute_input_hash("你好，世界", [chunk])
    assert first == second
    assert _HASH_RE.match(first)
