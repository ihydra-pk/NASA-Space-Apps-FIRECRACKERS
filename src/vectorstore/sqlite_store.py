"""SQLite-backed persistent vector store with cosine-search support."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from .config import VectorStoreConfig


class SQLiteVectorStore:
    """A dependency-free local store keyed by the source chunk_id.

    The full embedding record and exact vector are persisted as JSON. Repeated
    indexing performs an idempotent upsert by chunk_id rather than creating
    duplicate rows. A collection can contain only one provider/model/dimension.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        self.config = config
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.config.database_path), timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                similarity_metric TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS vectors (
                collection TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY (collection, chunk_id),
                FOREIGN KEY (collection) REFERENCES collections(name) ON DELETE CASCADE
            );
            """
        )
        self._connection.execute(
            """INSERT OR IGNORE INTO collections
               (name, provider, model_name, dimension, similarity_metric)
               VALUES (?, ?, ?, ?, ?)""",
            (
                config.collection,
                config.provider,
                config.model_name,
                config.dimension,
                config.similarity_metric,
            ),
        )
        profile = self._connection.execute(
            "SELECT provider, model_name, dimension, similarity_metric FROM collections WHERE name = ?",
            (config.collection,),
        ).fetchone()
        expected = (
            config.provider,
            config.model_name,
            config.dimension,
            config.similarity_metric,
        )
        actual = tuple(profile) if profile else None
        if actual != expected:
            self._connection.close()
            raise ValueError(
                f"Collection {config.collection!r} uses profile {actual}; requested {expected}. "
                "Choose a new collection or matching provider/model/dimension/metric."
            )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteVectorStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self._connection.commit()
        else:
            self._connection.rollback()
        self.close()

    def upsert(self, record: dict[str, Any], vector: Sequence[float]) -> bool:
        values = tuple(float(value) for value in vector)
        if len(values) != self.config.dimension:
            raise ValueError(
                f"Vector has dimension {len(values)}; expected {self.config.dimension}"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Vector contains non-finite values")
        if record.get("embedding_provider") != self.config.provider:
            raise ValueError("Embedding provider does not match collection configuration")
        if record.get("embedding_model") != self.config.model_name:
            raise ValueError("Embedding model does not match collection configuration")
        if record.get("embedding_dimension") != self.config.dimension:
            raise ValueError("Embedding record dimension does not match collection configuration")

        exists = self._connection.execute(
            "SELECT 1 FROM vectors WHERE collection = ? AND chunk_id = ?",
            (self.config.collection, record["chunk_id"]),
        ).fetchone()
        self._connection.execute(
            """INSERT INTO vectors (collection, chunk_id, vector_json, record_json)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(collection, chunk_id) DO UPDATE SET
                   vector_json = excluded.vector_json,
                   record_json = excluded.record_json""",
            (
                self.config.collection,
                record["chunk_id"],
                json.dumps(values, separators=(",", ":"), allow_nan=False),
                json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
            ),
        )
        self._connection.commit()
        return exists is None

    def get_collection_profile(self) -> tuple[str, str, int] | None:
        count = self._connection.execute(
            "SELECT COUNT(*) FROM vectors WHERE collection = ?", (self.config.collection,)
        ).fetchone()[0]
        if not count:
            return None
        return (self.config.provider, self.config.model_name, self.config.dimension)

    def records(self) -> list[tuple[dict[str, Any], tuple[float, ...]]]:
        rows = self._connection.execute(
            "SELECT record_json, vector_json FROM vectors "
            "WHERE collection = ? ORDER BY chunk_id",
            (self.config.collection,),
        ).fetchall()
        return [
            (json.loads(row["record_json"]), tuple(json.loads(row["vector_json"])))
            for row in rows
        ]
