from __future__ import annotations

import os

from ..exceptions import ConfigError
from .anthropic import AnthropicClient
from .base import LLMClient
from .openai import OpenAIClient


def from_spec(spec: str) -> LLMClient:
    """Build a client from a ``provider:model`` spec, e.g.
    ``"anthropic:claude-haiku-4-5"`` or ``"openai:gpt-4o-mini"``."""
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    if provider == "anthropic":
        return AnthropicClient(model=model) if model else AnthropicClient()
    if provider == "openai":
        return OpenAIClient(model=model) if model else OpenAIClient()
    raise ConfigError(f"Unknown LLM provider '{provider}' (expected 'anthropic' or 'openai')")


def detect() -> LLMClient | None:
    """Pick a provider from environment keys; None when no key is set.
    ``NEURAMINERL_LLM`` overrides with an explicit spec."""
    spec = os.environ.get("NEURAMINERL_LLM")
    if spec:
        return from_spec(spec)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicClient()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIClient()
    return None
