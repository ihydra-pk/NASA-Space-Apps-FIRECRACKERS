"""Persistent vector indexing and provenance-preserving semantic search."""

from .config import VectorStoreConfig
from .memory import CachedCollection
from .models import IndexResult, RetrievalResult, SearchResult
from .pipeline import index_embeddings
from .search import search, search_many
from .sqlite_store import SQLiteVectorStore

__all__ = [
    "CachedCollection",
    "IndexResult",
    "RetrievalResult",
    "SQLiteVectorStore",
    "SearchResult",
    "VectorStoreConfig",
    "index_embeddings",
    "search",
    "search_many",
]
