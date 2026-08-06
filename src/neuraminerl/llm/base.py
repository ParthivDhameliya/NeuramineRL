from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

Message = dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    data: dict[str, Any] | None = None  # parsed structured output when json_schema was given
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal completion interface. When ``json_schema`` is given, the
    provider is forced into structured output and ``LLMResponse.data`` holds
    the parsed object."""

    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse: ...


@dataclass
class UsageEvent:
    """One completed LLM call, reported to ``Learner(on_usage=...)`` so the
    library's own spend (reflection, dedup merges) can feed the host
    application's cost tracking."""

    client: str  # adapter class name, e.g. "AnthropicClient"
    model: str
    input_tokens: int
    output_tokens: int


class UsageTrackingLLM:
    """Wraps any LLMClient and reports a UsageEvent after every completion.
    A raising callback is reported as a warning — cost tracking must never
    break a run."""

    def __init__(self, inner: LLMClient, on_usage: Callable[[UsageEvent], None]) -> None:
        self._inner = inner
        self._on_usage = on_usage
        self.model = inner.model

    def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        response = self._inner.complete(
            messages, system=system, json_schema=json_schema, max_tokens=max_tokens
        )
        event = UsageEvent(
            client=type(self._inner).__name__,
            model=self._inner.model,
            input_tokens=int(response.usage.get("input_tokens", 0)),
            output_tokens=int(response.usage.get("output_tokens", 0)),
        )
        try:
            self._on_usage(event)
        except Exception as exc:  # observability must never break a run
            warnings.warn(f"on_usage callback raised: {exc!r}", stacklevel=2)
        return response
