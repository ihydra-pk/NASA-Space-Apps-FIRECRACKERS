"""Persistent vector indexing and provenance-preserving semantic search."""

from .config import VectorStoreConfig
from .models import IndexResult, RetrievalResult, SearchResult
from .pipeline import index_embeddings
from .search import search
from .sqlite_store import SQLiteVectorStore

__all__ = [
    "IndexResult",
    "RetrievalResult",
    "SQLiteVectorStore",
    "SearchResult",
    "VectorStoreConfig",
    "index_embeddings",
    "search",
]
