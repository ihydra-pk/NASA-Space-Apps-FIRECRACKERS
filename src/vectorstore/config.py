"""Settings for a local persistent vector collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    """Collection identity and local SQLite configuration."""

    database_path: Path = Path("data/vectorstore/firelens.sqlite3")
    collection: str = "flex-qwen3-embedding-4b"
    provider: str = "openai-compatible"
    model_name: str = "gwdg.qwen3-embedding-4b"
    dimension: int = 2560
    similarity_metric: str = "cosine"
    top_k: int = 5

    def __post_init__(self) -> None:
        if not self.collection.strip():
            raise ValueError("collection must not be empty")
        if not self.provider.strip() or not self.model_name.strip():
            raise ValueError("provider and model_name must not be empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.similarity_metric != "cosine":
            raise ValueError("Only cosine similarity is currently supported")
