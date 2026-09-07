"""Deterministic SHA-256 hash of what ``VerifyService.verify`` actually consumes.

Per core_requirements #9.3, the hash covers exactly ``answer`` and
``chunks`` (in request order, each chunk's ``chunk_id``/``text``/``source``/
``metadata``) -- nothing else. It deliberately does NOT take a
``VerificationRequest`` or a ``query``/``request_id``: ``verify`` never
consumes ``query`` (core_requirements #4), and ``request_id`` is a tracing
identifier, not content being verified, so omitting them here is a
structural guarantee (there is no field to accidentally include), not a
convention someone has to remember to follow.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from groundguard_rag.domain.models import Chunk


def compute_input_hash(answer: str, chunks: Sequence[Chunk]) -> str:
    """Return ``sha256:<64 hex chars>`` over ``answer`` and ``chunks``.

    Deterministic: the same ``answer`` and the same ``chunks`` in the same
    order always produce the same hash, regardless of ``metadata`` key
    insertion order (``json.dumps(..., sort_keys=True)`` canonicalizes
    that). Chunk *order* is preserved and does affect the hash -- RAG
    context ordering can affect what a verifier concludes, so it is part
    of what was actually consumed.
    """
    payload = {
        "answer": answer,
        "chunks": [_chunk_payload(chunk) for chunk in chunks],
    }
    # allow_nan=False turns a NaN/Infinity value into a hard error instead
    # of silently emitting the non-standard "NaN"/"Infinity" JSON tokens.
    # Chunk.metadata already rejects non-finite floats at construction
    # time (core_requirements #8.4), so this is a second, defense-in-depth
    # guarantee rather than the only line of defense.
    canonical = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def compute_answer_hash(answer: str) -> str:
    """Return an unambiguous SHA-256 hash of an answer string alone.

    Repair records call these fields ``answer_hash_*``; using a separate
    helper prevents accidentally placing the answer+chunks state hash in a
    field whose name promises answer-only lineage.
    """
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    digest = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _chunk_payload(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "source": chunk.source,
        "metadata": _json_safe(chunk.metadata),
    }


def _json_safe(value: Any) -> Any:
    # Chunk.metadata is frozen to MappingProxyType/tuple by the domain
    # layer (core_requirements #8.4), neither of which json.dumps can
    # serialize directly -- convert to plain dict/list. Key sorting is
    # left to json.dumps(sort_keys=True) at the top level, which sorts
    # every nested dict's keys recursively once they are plain dicts.
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
