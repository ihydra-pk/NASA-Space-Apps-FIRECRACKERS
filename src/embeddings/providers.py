"""Embedding provider interface and deterministic non-semantic test provider."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Callable, Protocol, Sequence, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import EmbeddingConfig

DEFAULT_GATEWAY_BASE_URL = "https://ai-gateway.uni-paderborn.de/v1"


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider contract: embed a batch of texts under a shared config."""

    provider_name: str

    def embed_batch(
        self, texts: Sequence[str], config: EmbeddingConfig
    ) -> Sequence[Sequence[float]]:
        """Return one finite vector of ``config.dimension`` values per text."""


class HashMockEmbeddingProvider:
    """Deterministic hash-derived vectors for tests and pipeline validation.

    These vectors are not semantic embeddings and must not be used for
    similarity search or presented as model-generated scientific vectors.
    """

    provider_name = "mock"

    def embed_batch(
        self, texts: Sequence[str], config: EmbeddingConfig
    ) -> list[list[float]]:
        return [self._embed(text, config) for text in texts]

    @staticmethod
    def _embed(text: str, config: EmbeddingConfig) -> list[float]:
        seed = hashlib.sha256(
            f"firelens-mock-embedding-v1\0{config.model_name}\0{text}".encode("utf-8")
        ).digest()
        values: list[float] = []
        counter = 0
        while len(values) < config.dimension:
            block = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
            for offset in range(0, len(block), 4):
                integer = int.from_bytes(block[offset : offset + 4], "big")
                # Deterministically map a 32-bit unsigned integer to [-1, 1].
                values.append((integer / 0xFFFFFFFF) * 2.0 - 1.0)
                if len(values) == config.dimension:
                    break
            counter += 1

        # Keep vector scale bounded consistently. No semantic meaning is implied.
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            values = [value / norm for value in values]
        return values


class OpenAICompatibleEmbeddingProvider:
    """Real semantic embedding provider for OpenAI-compatible embeddings APIs.

    Credentials are read from ``FIRELENS_EMBEDDING_API_KEY``,
    ``OPENAI_API_KEY``, or the existing OpenCode gateway credential
    ``UPB_AI_GATEWAY_KEY``. The endpoint may be overridden with
    ``FIRELENS_EMBEDDING_BASE_URL``. Requests are batched and use the standard
    ``POST /embeddings`` schema. This class never falls back to mock vectors.
    """

    provider_name = "openai-compatible"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        urlopen_func: Callable[..., object] | None = None,
        use_gateway_credential: bool | None = None,
        max_retries: int = 5,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        configured_base_url = base_url or os.environ.get("FIRELENS_EMBEDDING_BASE_URL")
        other_key = api_key or os.environ.get("FIRELENS_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if configured_base_url is None and not other_key and os.environ.get("UPB_AI_GATEWAY_KEY"):
            # Only the gateway credential is available, and the project's index was built through the gateway.
            configured_base_url = DEFAULT_GATEWAY_BASE_URL
        if use_gateway_credential is None:
            use_gateway_credential = configured_base_url is not None
        self.api_key = (
            api_key
            or os.environ.get("FIRELENS_EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or (
                os.environ.get("UPB_AI_GATEWAY_KEY")
                if use_gateway_credential
                else None
            )
        )
        self.base_url = (configured_base_url or "https://api.openai.com/v1").rstrip("/")
        self._urlopen = urlopen_func or urlopen
        self.max_retries = max_retries
        self._sleep = sleep_func

    def embed_batch(
        self, texts: Sequence[str], config: EmbeddingConfig
    ) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError(
                "Real embedding provider requires credentials. Set "
                "FIRELENS_EMBEDDING_API_KEY or OPENAI_API_KEY; no mock fallback was used."
            )
        if not texts:
            return []

        payload: dict[str, object] = {"model": config.model_name, "input": list(texts)}
        # The OpenAI text-embedding-3 family supports a requested output size.
        # Other compatible models choose their native dimension and are checked
        # against the configured dimension by the shared pipeline.
        if config.model_name.startswith("text-embedding-3"):
            payload["dimensions"] = config.dimension

        request = Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = self._urlopen(request, timeout=60)
                    with response as result:
                        response_body = result.read()
                    break
                except HTTPError as exc:
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                        raise
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 ** attempt * 2
                    self._sleep(min(delay, 60.0))
            parsed = json.loads(response_body.decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Embedding API returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach embedding API: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise RuntimeError(f"Embedding API request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Embedding API returned invalid JSON: {exc}") from exc

        data = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(data, list):
            raise RuntimeError("Embedding API response is missing a data array")

        indexed: list[tuple[int, list[float]]] = []
        try:
            for item in data:
                index = item["index"]
                vector = item["embedding"]
                if not isinstance(index, int) or not isinstance(vector, list):
                    raise TypeError("response item has invalid index or embedding")
                indexed.append((index, vector))
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Embedding API returned malformed data: {exc}") from exc

        indexed.sort(key=lambda item: item[0])
        if [index for index, _ in indexed] != list(range(len(texts))):
            raise RuntimeError("Embedding API response indices do not match the input batch")
        return [vector for _, vector in indexed]


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Create the explicitly selected provider without implicit fallbacks."""

    if config.provider == "mock":
        return HashMockEmbeddingProvider()
    if config.provider == OpenAICompatibleEmbeddingProvider.provider_name:
        return OpenAICompatibleEmbeddingProvider()
    raise ValueError(
        f"Unsupported embedding provider {config.provider!r}. "
        "Supported providers: 'mock', 'openai-compatible'."
    )
