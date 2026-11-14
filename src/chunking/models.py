"""Chunk records that retain the complete normalized source record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A source-text span and its chunk metadata.

    ``source_record`` is copied into each JSON object before chunk fields are
    overlaid. This keeps existing and future source provenance fields intact;
    only the source ``text`` is replaced by this chunk's exact text span.
    """

    source_record: dict[str, Any]
    chunk_id: str
    chunk_index: int
    chunk_count: int
    text: str
    estimated_token_count: int

    def to_dict(self) -> dict[str, Any]:
        record = dict(self.source_record)
        record.update(
            {
                "chunk_id": self.chunk_id,
                "chunk_index": self.chunk_index,
                "chunk_count": self.chunk_count,
                "text": self.text,
                "estimated_token_count": self.estimated_token_count,
            }
        )
        return record
