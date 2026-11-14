"""LLM provider protocol, OpenAI-compatible adapter, and deterministic mock."""

from __future__ import annotations

import json
import os
from typing import Callable, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import LLMConfig


@runtime_checkable
class LLMProvider(Protocol):
    provider_name: str

    def generate(self, *, system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
        """Return raw assistant response text; the service validates its schema."""


class OpenAICompatibleLLMProvider:
    """Chat-completions adapter compatible with the existing UPB AI Gateway."""

    provider_name = "openai-compatible"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        urlopen_func: Callable[..., object] | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("FIRELENS_LLM_API_KEY")
            or os.environ.get("UPB_AI_GATEWAY_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = (base_url or os.environ.get("FIRELENS_LLM_BASE_URL") or "").rstrip("/")
        self._urlopen = urlopen_func or urlopen

    def generate(self, *, system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
        if not self.api_key:
            raise RuntimeError(
                "LLM credentials are missing. Set UPB_AI_GATEWAY_KEY, "
                "FIRELENS_LLM_API_KEY, or OPENAI_API_KEY."
            )
        base_url = self.base_url or config.base_url.rstrip("/")
        payload = {
            "model": config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            response = self._urlopen(request, timeout=config.timeout_seconds)
            with response as result:
                body = result.read()
            parsed = json.loads(body.decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"LLM API returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach LLM API: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM API returned invalid JSON: {exc}") from exc

        try:
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM API response is missing choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM API returned empty assistant content")
        if parsed["choices"][0].get("finish_reason") == "length":
            raise RuntimeError(
                "LLM answer was cut off at the output token limit "
                f"({config.max_output_tokens}); raise FIRELENS_LLM_MAX_OUTPUT_TOKENS."
            )
        return content


class MockLLMProvider:
    """Deterministic test provider. Its responses are fixtures, not LLM output."""

    provider_name = "mock"

    def __init__(self, response: str | dict | None = None) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self.response, str):
            return self.response
        if isinstance(self.response, dict):
            return json.dumps(self.response, ensure_ascii=False)

        # Deterministic protocol fixture only: no attempt at scientific reasoning.
        try:
            structured = user_prompt.split("STRUCTURED EVIDENCE ITEMS\n-------------------------\n", 1)[1]
            evidence = json.loads(structured)
        except (IndexError, json.JSONDecodeError):
            evidence = []
        if not evidence:
            return json.dumps(
                {
                    "answer": "The indexed evidence is insufficient to answer this question.",
                    "status": "insufficient_evidence",
                    "citations": [],
                }
            )
        first = evidence[0]
        citation = {
            key: first[key]
            for key in ("chunk_id", "source_id", "source_locator", "page_number", "evidence_status")
        }
        return json.dumps(
            {
                "answer": "Mock response fixture: retrieved evidence is available; inspect the cited source text.",
                "status": "grounded",
                "citations": [citation],
            },
            ensure_ascii=False,
        )


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    if config.provider == "mock":
        return MockLLMProvider()
    if config.provider == OpenAICompatibleLLMProvider.provider_name:
        return OpenAICompatibleLLMProvider()
    raise ValueError(
        f"Unsupported LLM provider {config.provider!r}; "
        "supported providers: 'openai-compatible', 'mock'."
    )
