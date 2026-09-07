"""Duck-typed host-framework adapters with no mandatory framework imports."""

from groundguard_rag.adapters.frameworks.langchain import (
    LangChainRetrieverAdapter,
    chunks_from_langchain_documents,
)
from groundguard_rag.adapters.frameworks.llamaindex import (
    LlamaIndexRetrieverAdapter,
    chunks_from_llamaindex_nodes,
)

__all__ = [
    "LangChainRetrieverAdapter",
    "LlamaIndexRetrieverAdapter",
    "chunks_from_langchain_documents",
    "chunks_from_llamaindex_nodes",
]
