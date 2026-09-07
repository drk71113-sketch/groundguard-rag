"""LlamaIndex node/retriever adapters without importing LlamaIndex itself."""

from __future__ import annotations

from typing import Any

from groundguard_rag.adapters.frameworks._common import (
    build_chunk,
    reject_duplicate_ids,
    require_sequence,
)
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError
from groundguard_rag.domain.models import Chunk
from groundguard_rag.domain.ports import Retriever


def _unwrap_node(item: Any) -> Any:
    return getattr(item, "node", item)


def _node_text(node: Any) -> Any:
    text = getattr(node, "text", None)
    if isinstance(text, str) and text:
        return text
    getter = getattr(node, "get_content", None)
    if callable(getter):
        return getter()
    return None


def chunks_from_llamaindex_nodes(
    nodes: Any, *, id_prefix: str = "llamaindex"
) -> list[Chunk]:
    """Convert Node or NodeWithScore-like objects into immutable chunks."""

    if not isinstance(id_prefix, str) or not id_prefix.strip():
        raise DomainValidationError("id_prefix must be a non-empty string")
    items = require_sequence(nodes, "nodes")
    chunks: list[Chunk] = []
    for index, item in enumerate(items):
        node = _unwrap_node(item)
        metadata = getattr(node, "metadata", {})
        metadata_chunk_id = (
            metadata.get("chunk_id") if hasattr(metadata, "get") else None
        )
        chunks.append(
            build_chunk(
                item=node,
                index=index,
                text=_node_text(node),
                metadata=metadata,
                preferred_ids=(
                    getattr(node, "node_id", None),
                    getattr(node, "id_", None),
                    metadata_chunk_id,
                ),
                id_prefix=id_prefix,
                source_name="nodes",
            )
        )
    reject_duplicate_ids(chunks, "LlamaIndex conversion")
    return chunks


class LlamaIndexRetrieverAdapter(Retriever):
    """Wrap a caller-owned LlamaIndex retriever exposing ``retrieve(query)``."""

    def __init__(
        self, retriever: Any, *, id_prefix: str = "llamaindex-retrieved"
    ) -> None:
        if not callable(getattr(retriever, "retrieve", None)):
            raise ConfigurationError(
                "LlamaIndexRetrieverAdapter.retriever must expose retrieve(query)"
            )
        if not isinstance(id_prefix, str) or not id_prefix.strip():
            raise ConfigurationError(
                "LlamaIndexRetrieverAdapter.id_prefix must be non-empty"
            )
        self._retriever = retriever
        self._id_prefix = id_prefix

    def retrieve(self, query: str) -> list[Chunk]:
        if not isinstance(query, str) or not query.strip():
            raise DomainValidationError(
                "LlamaIndexRetrieverAdapter.query must be non-empty"
            )
        nodes = self._retriever.retrieve(query)
        return chunks_from_llamaindex_nodes(nodes, id_prefix=self._id_prefix)
