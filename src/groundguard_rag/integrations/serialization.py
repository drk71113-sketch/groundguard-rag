"""Strict JSON-boundary conversion used by protocol adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import Chunk, VerificationRequest

_CHUNK_FIELDS = frozenset({"chunk_id", "text", "source", "metadata"})


def request_from_json(
    *,
    request_id: str,
    answer: str,
    chunks: list[dict[str, Any]],
    query: str | None = None,
) -> VerificationRequest:
    """Construct a validated domain request from an untrusted JSON payload."""

    if not isinstance(chunks, list):
        raise DomainValidationError("chunks must be a JSON array")
    parsed: list[Chunk] = []
    for index, document in enumerate(chunks):
        if not isinstance(document, Mapping):
            raise DomainValidationError(f"chunks[{index}] must be an object")
        unknown = set(document) - _CHUNK_FIELDS
        missing = {"chunk_id", "text"} - set(document)
        if unknown:
            raise DomainValidationError(
                f"chunks[{index}] contains unknown fields: {sorted(unknown)!r}"
            )
        if missing:
            raise DomainValidationError(
                f"chunks[{index}] is missing fields: {sorted(missing)!r}"
            )
        parsed.append(
            Chunk(
                chunk_id=document["chunk_id"],
                text=document["text"],
                source=document.get("source"),
                metadata=document.get("metadata", {}),
            )
        )
    return VerificationRequest(
        request_id=request_id,
        answer=answer,
        chunks=tuple(parsed),
        query=query,
    )
