"""A read-only, in-memory copy of one collection for repeated searches."""

from __future__ import annotations

from array import array
from typing import Any, Sequence

from .config import VectorStoreConfig
from .sqlite_store import SQLiteVectorStore
from .store import VectorStore


class CachedCollection:
    """Every record and vector of a collection, read once and kept in memory.

    Vectors are stored as JSON in SQLite, so reading the 4,103-chunk index
    takes about 9 seconds; without this cache every question paid that cost.
    Vectors are held as ``array('d')`` (about 84 MB for that index instead of
    roughly 340 MB as tuples of floats). It implements the read side of
    :class:`~src.vectorstore.store.VectorStore`, so ``search`` accepts it in
    place of a store.
    """

    def __init__(self, store: VectorStore) -> None:
        self._profile = store.get_collection_profile()
        # SQLiteVectorStore streams rows, so each vector becomes a compact
        # array before the next is read. Stored records also repeat the vector
        # under "embedding"; search never reads it (RetrievalResult drops it),
        # so the cache leaves it out. Together these keep a 512 MB server alive.
        rows = store.iter_records() if isinstance(store, SQLiteVectorStore) else store.records()
        self._entries: list[tuple[dict[str, Any], Sequence[float]]] = [
            ({key: value for key, value in record.items() if key != "embedding"}, array("d", vector))
            for record, vector in rows
        ]

    @classmethod
    def load(cls, config: VectorStoreConfig) -> "CachedCollection":
        with SQLiteVectorStore(config) as store:
            return cls(store)

    def get_collection_profile(self) -> tuple[str, str, int] | None:
        return self._profile

    def records(self) -> list[tuple[dict[str, Any], Sequence[float]]]:
        # Shared, not copied: callers must not modify the list or its records.
        return self._entries

    def upsert(self, record: dict[str, Any], vector: Sequence[float]) -> bool:
        raise TypeError("CachedCollection is read-only; index into SQLiteVectorStore instead")
