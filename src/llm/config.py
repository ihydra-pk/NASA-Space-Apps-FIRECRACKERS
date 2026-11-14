"""LLM provider and generation configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_UPB_BASE_URL = "https://ai-gateway.uni-paderborn.de/v1"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Provider-neutral chat-completions settings.

    Values can be set explicitly or loaded from environment by
    :meth:`from_env`. No credentials are stored in project source/config files.
    """

    provider: str = "openai-compatible"
    model_name: str = "openai.gpt-6-luna"
    base_url: str = DEFAULT_UPB_BASE_URL
    timeout_seconds: float = 90.0
    temperature: float = 0.0
    # Room for findings, background and limitations as well as the answer.
    max_output_tokens: int = 4000

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must not be empty")
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

    @classmethod
    def from_env(cls, *, provider: str | None = None) -> "LLMConfig":
        selected_provider = provider or os.environ.get("FIRELENS_LLM_PROVIDER", "openai-compatible")
        return cls(
            provider=selected_provider,
            model_name=os.environ.get("FIRELENS_LLM_MODEL", "openai.gpt-6-luna"),
            base_url=os.environ.get("FIRELENS_LLM_BASE_URL", DEFAULT_UPB_BASE_URL),
            timeout_seconds=float(os.environ.get("FIRELENS_LLM_TIMEOUT", "90")),
            temperature=float(os.environ.get("FIRELENS_LLM_TEMPERATURE", "0")),
            max_output_tokens=int(os.environ.get("FIRELENS_LLM_MAX_OUTPUT_TOKENS", "4000")),
        )

    @property
    def credential_env_vars(self) -> tuple[str, ...]:
        return ("FIRELENS_LLM_API_KEY", "UPB_AI_GATEWAY_KEY", "OPENAI_API_KEY")
