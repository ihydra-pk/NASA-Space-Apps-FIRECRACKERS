"""Embedding output and failure records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """An embedding vector attached to the entire original chunk record."""

    chunk_record: dict[str, Any]
    embedding: tuple[float, ...]
    embedding_model: str
    embedding_provider: str
    embedding_dimension: int

    def to_dict(self) -> dict[str, Any]:
        record = dict(self.chunk_record)
        record.update(
            {
                "embedding": list(self.embedding),
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_provider,
                "embedding_dimension": self.embedding_dimension,
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class EmbeddingFailure:
    """A chunk that could not be embedded; retained in a companion JSONL."""

    input_file: str
    line_number: int
    error: str
    chunk_record: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_file": self.input_file,
            "line_number": self.line_number,
            "error": self.error,
            "chunk_record": self.chunk_record,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Summary of a batch embedding run."""

    input_chunks: int
    records: tuple[EmbeddingRecord, ...]
    failures: tuple[EmbeddingFailure, ...]
    processed_paths: tuple[str, ...]
    output_paths: tuple[str, ...]

    @property
    def successfully_embedded(self) -> int:
        return len(self.records)

    @property
    def failed_chunks(self) -> int:
        return len(self.failures)
