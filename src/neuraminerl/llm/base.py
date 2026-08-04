from __future__ import annotations

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
