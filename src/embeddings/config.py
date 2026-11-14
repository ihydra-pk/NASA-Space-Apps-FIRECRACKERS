"""Configuration for chunk embedding generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Provider-neutral embedding settings.

    The default provider is an explicitly non-semantic deterministic mock.
    Select ``openai-compatible`` for real semantic embeddings. Provider
    credentials and endpoint configuration are read from the environment.
    """

    input_dir: Path = Path("data/chunks")
    output_dir: Path = Path("data/embeddings")
    provider: str = "mock"
    model_name: str = "firelens-hash-v1"
    dimension: int = 384
    batch_size: int = 16

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
