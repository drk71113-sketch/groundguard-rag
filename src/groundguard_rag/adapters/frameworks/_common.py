"""Shared validation helpers for optional host-framework conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from groundguard_rag.domain.exceptions import DomainValidationError
from groundguard_rag.domain.models import Chunk


def require_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise DomainValidationError(f"{name} must be a sequence")
    return value


def build_chunk(
    *,
    item: Any,
    index: int,
    text: Any,
    metadata: Any,
    preferred_ids: tuple[Any, ...],
    id_prefix: str,
    source_name: str,
) -> Chunk:
    if not isinstance(text, str) or not text:
        raise DomainValidationError(
            f"{source_name}[{index}] must expose non-empty text"
        )
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise DomainValidationError(
            f"{source_name}[{index}].metadata must be a mapping"
        )
    copied_metadata = dict(metadata)

    chunk_id = next(
        (
            candidate.strip()
            for candidate in preferred_ids
            if isinstance(candidate, str) and candidate.strip()
        ),
        f"{id_prefix}-{index}",
    )
    source = copied_metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        source = None
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        metadata=copied_metadata,
    )


def reject_duplicate_ids(chunks: list[Chunk], name: str) -> None:
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise DomainValidationError(f"{name} produced duplicate chunk IDs")
