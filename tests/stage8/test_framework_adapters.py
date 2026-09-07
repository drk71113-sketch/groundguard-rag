from __future__ import annotations

from types import SimpleNamespace

import pytest

from groundguard_rag.adapters.frameworks import (
    LangChainRetrieverAdapter,
    LlamaIndexRetrieverAdapter,
    chunks_from_langchain_documents,
    chunks_from_llamaindex_nodes,
)
from groundguard_rag.domain.exceptions import ConfigurationError, DomainValidationError


def test_langchain_documents_convert_without_importing_langchain():
    document = SimpleNamespace(
        id="doc-1",
        page_content="Evidence text",
        metadata={"source": "manual", "page": 2},
    )
    chunk = chunks_from_langchain_documents([document])[0]
    assert chunk.chunk_id == "doc-1"
    assert chunk.text == "Evidence text"
    assert chunk.source == "manual"
    assert chunk.metadata["page"] == 2


def test_langchain_retriever_wraps_host_invoke():
    host = SimpleNamespace(
        invoke=lambda query: [
            SimpleNamespace(page_content=f"result for {query}", metadata={})
        ]
    )
    chunk = LangChainRetrieverAdapter(host).retrieve("where")[0]
    assert chunk.chunk_id == "langchain-retrieved-0"
    assert chunk.text == "result for where"


def test_llamaindex_node_with_score_is_unwrapped():
    node = SimpleNamespace(node_id="node-1", text="Node text", metadata={})
    wrapped = SimpleNamespace(node=node, score=0.9)
    chunk = chunks_from_llamaindex_nodes([wrapped])[0]
    assert chunk.chunk_id == "node-1"
    assert chunk.text == "Node text"


def test_llamaindex_retriever_wraps_host_retrieve():
    host = SimpleNamespace(
        retrieve=lambda query: [
            SimpleNamespace(id_="node-2", text=f"node {query}", metadata={})
        ]
    )
    chunk = LlamaIndexRetrieverAdapter(host).retrieve("query")[0]
    assert chunk.chunk_id == "node-2"


def test_framework_adapters_reject_bad_hosts_and_duplicate_ids():
    with pytest.raises(ConfigurationError):
        LangChainRetrieverAdapter(object())
    with pytest.raises(ConfigurationError):
        LlamaIndexRetrieverAdapter(object())
    duplicated = [
        SimpleNamespace(id="same", page_content="a", metadata={}),
        SimpleNamespace(id="same", page_content="b", metadata={}),
    ]
    with pytest.raises(DomainValidationError):
        chunks_from_langchain_documents(duplicated)
