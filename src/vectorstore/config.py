"""Settings for a local persistent vector collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Qwen3 embedding models are trained to take a one-line task description in
# front of the *query*; documents are embedded as plain text. Without it, a
# plain question scores highest against very short chunks such as slide titles.
DEFAULT_QUERY_INSTRUCTION = (
    "Given a question about microgravity combustion experiments, "
    "retrieve passages from NASA documents that answer it"
)


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
    # Applied only when the model is a Qwen3 embedding model; None disables it.
    query_instruction: str | None = DEFAULT_QUERY_INSTRUCTION
    # Chunks with fewer words are skipped when ranking (they stay in the index).
    min_chunk_words: int = 0

    def __post_init__(self) -> None:
        if not self.collection.strip():
            raise ValueError("collection must not be empty")
        if not self.provider.strip() or not self.model_name.strip():
            raise ValueError("provider and model_name must not be empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.min_chunk_words < 0:
            raise ValueError("min_chunk_words must not be negative")
        if self.similarity_metric != "cosine":
            raise ValueError("Only cosine similarity is currently supported")
