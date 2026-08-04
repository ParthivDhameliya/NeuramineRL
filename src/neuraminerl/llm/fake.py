"""In-memory fake LLM for tests and recorded-fixture runs."""

from __future__ import annotations

from typing import Any

from .base import LLMResponse, Message


class FakeLLM:
    """Returns queued responses in order; records every call for assertions.

    Queue items may be ``LLMResponse`` objects, dicts (returned as structured
    ``data``), or strings (returned as ``text``). When the queue is empty the
    ``default`` response is returned.
    """

    model = "fake"

    def __init__(self, responses: list[Any] | None = None, default: Any = None) -> None:
        self._queue: list[Any] = list(responses or [])
        self._default = default if default is not None else {"lessons": []}
        self.calls: list[dict[str, Any]] = []

    def queue(self, *responses: Any) -> None:
        self._queue.extend(responses)

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": messages,
                "system": system,
                "json_schema": json_schema,
                "max_tokens": max_tokens,
            }
        )
        item = self._queue.pop(0) if self._queue else self._default
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, dict):
            return LLMResponse(data=item)
        return LLMResponse(text=str(item))
