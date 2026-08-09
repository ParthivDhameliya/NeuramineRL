"""Thin httpx client for the OpenAI Chat Completions API.

Structured output uses ``response_format: json_schema`` (strict mode).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from ..exceptions import LLMError
from .base import LLMResponse, Message

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIClient:
    """Works against any OpenAI-Chat-Completions-compatible server (Groq,
    Together, Fireworks, DeepSeek, OpenRouter, Azure, Ollama, vLLM, ...)
    via ``base_url``."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout
        # OPENAI_API_KEY authenticates to OpenAI and nowhere else. A custom
        # base_url is a different operator's server, so the provider key is
        # never forwarded to it: pass api_key=, or set NEURAMINERL_API_KEY.
        # A custom base_url may also be a local keyless server (Ollama, vLLM).
        self._api_key = api_key or os.environ.get("NEURAMINERL_API_KEY", "")
        if not self._api_key and self._base_url == DEFAULT_BASE_URL:
            self._api_key = os.environ.get("OPENAI_API_KEY", "")
            if not self._api_key:
                raise LLMError("No OpenAI API key (set OPENAI_API_KEY)")

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        chat_messages: list[Message] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": chat_messages,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": json_schema, "strict": True},
            }
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"OpenAI API error {exc.response.status_code}: {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"OpenAI API request failed: {exc}") from exc

        payload = response.json()
        # Gateways (OpenRouter, Azure, local servers) return 200 with an error
        # body or an empty choices list; index blindly and callers get a bare
        # KeyError instead of this adapter's own error type.
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError(f"OpenAI response contained no choices: {str(payload)[:500]}")
        message = choices[0].get("message") or {}
        refusal = message.get("refusal")
        if refusal:
            raise LLMError(f"OpenAI refused the request: {str(refusal)[:500]}")
        text = message.get("content") or ""
        data: dict[str, Any] | None = None
        if json_schema is not None and text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"OpenAI returned invalid JSON for structured output: {exc}"
                ) from exc
        usage = payload.get("usage", {})
        return LLMResponse(
            text=text,
            data=data,
            usage={
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
            },
        )
