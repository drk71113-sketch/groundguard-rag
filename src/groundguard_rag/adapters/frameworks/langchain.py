"""LangChain document/retriever adapters without importing LangChain itself."""

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


def chunks_from_langchain_documents(
    documents: Any, *, id_prefix: str = "langchain"
) -> list[Chunk]:
    """Convert document-like objects exposing ``page_content``/``metadata``."""

    if not isinstance(id_prefix, str) or not id_prefix.strip():
        raise DomainValidationError("id_prefix must be a non-empty string")
    items = require_sequence(documents, "documents")
    chunks: list[Chunk] = []
    for index, document in enumerate(items):
        metadata = getattr(document, "metadata", {})
        metadata_chunk_id = (
            metadata.get("chunk_id") if hasattr(metadata, "get") else None
        )
        chunks.append(
            build_chunk(
                item=document,
                index=index,
                text=getattr(document, "page_content", None),
                metadata=metadata,
                preferred_ids=(
                    getattr(document, "id", None),
                    metadata_chunk_id,
                ),
                id_prefix=id_prefix,
                source_name="documents",
            )
        )
    reject_duplicate_ids(chunks, "LangChain conversion")
    return chunks


class LangChainRetrieverAdapter(Retriever):
    """Wrap a caller-owned LangChain retriever exposing ``invoke(query)``."""

    def __init__(self, retriever: Any, *, id_prefix: str = "langchain-retrieved") -> None:
        if not callable(getattr(retriever, "invoke", None)):
            raise ConfigurationError(
                "LangChainRetrieverAdapter.retriever must expose invoke(query)"
            )
        if not isinstance(id_prefix, str) or not id_prefix.strip():
            raise ConfigurationError(
                "LangChainRetrieverAdapter.id_prefix must be non-empty"
            )
        self._retriever = retriever
        self._id_prefix = id_prefix

    def retrieve(self, query: str) -> list[Chunk]:
        if not isinstance(query, str) or not query.strip():
            raise DomainValidationError(
                "LangChainRetrieverAdapter.query must be non-empty"
            )
        documents = self._retriever.invoke(query)
        return chunks_from_langchain_documents(
            documents, id_prefix=self._id_prefix
        )
