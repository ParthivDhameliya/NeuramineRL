"""Thin httpx client for the Anthropic Messages API.

Structured output is implemented via forced tool use: the schema becomes a
single ``emit`` tool and ``tool_choice`` forces the model to call it.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..exceptions import LLMError
from .base import LLMResponse, Message

DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        if not self._api_key:
            raise LLMError("No Anthropic API key (set ANTHROPIC_API_KEY)")

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system
        if json_schema is not None:
            body["tools"] = [
                {
                    "name": "emit",
                    "description": "Emit the structured result.",
                    "input_schema": json_schema,
                }
            ]
            body["tool_choice"] = {"type": "tool", "name": "emit"}
        try:
            response = httpx.post(
                f"{self._base_url}/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"Anthropic API error {exc.response.status_code}: {exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Anthropic API request failed: {exc}") from exc

        payload = response.json()
        text_parts: list[str] = []
        data: dict[str, Any] | None = None
        for block in payload.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
            elif block.get("type") == "tool_use" and block.get("name") == "emit":
                data = block.get("input")
        usage = payload.get("usage", {})
        return LLMResponse(
            text="".join(text_parts),
            data=data,
            usage={
                "input_tokens": int(usage.get("input_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
            },
        )
