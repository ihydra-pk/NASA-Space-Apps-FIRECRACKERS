"""Store interface shared by indexing and semantic search."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Provider-agnostic interface for persistent vector collections."""

    def upsert(self, record: dict[str, Any], vector: Sequence[float]) -> bool:
        """Store by deterministic chunk ID; return True when inserted new."""

    def get_collection_profile(self) -> tuple[str, str, int] | None:
        """Return (provider, model, dimension) for non-empty collection."""

    def records(self) -> list[tuple[dict[str, Any], tuple[float, ...]]]:
        """Return complete records and their original stored vectors."""
