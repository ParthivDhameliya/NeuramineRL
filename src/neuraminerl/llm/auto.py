from __future__ import annotations

import os

from ..exceptions import ConfigError
from .anthropic import DEFAULT_MODEL as _ANTHROPIC_DEFAULT
from .anthropic import AnthropicClient
from .base import LLMClient
from .gemini import DEFAULT_MODEL as _GEMINI_DEFAULT
from .gemini import GeminiClient
from .openai import DEFAULT_MODEL as _OPENAI_DEFAULT
from .openai import OpenAIClient


def from_spec(spec: str) -> LLMClient:
    """Build a client from a ``provider:model[@base_url]`` spec.

    Examples::

        "anthropic:claude-haiku-4-5"
        "gemini:gemini-2.5-flash"
        "openai:gpt-4o-mini"
        "openai:llama-3.1-70b@https://api.groq.com/openai/v1"
        "openai:qwen2.5@http://localhost:11434/v1"

    The ``@base_url`` form points the ``openai`` provider at any
    OpenAI-Chat-Completions-compatible server (Groq, Together, Fireworks,
    DeepSeek, OpenRouter, Azure, Ollama, vLLM, ...). The API key is read
    from ``NEURAMINERL_API_KEY`` first, then the provider's usual variable;
    servers on a custom base_url may be keyless (local Ollama/vLLM).
    Anything beyond these three wire protocols: implement the ``LLMClient``
    protocol and pass the instance via ``Learner(llm=...)``.
    """
    provider, _, rest = spec.partition(":")
    provider = provider.strip().lower()
    model, _, base_url = rest.partition("@")
    model = model.strip()
    base = base_url.strip() or None
    if provider == "anthropic":
        return AnthropicClient(model=model or _ANTHROPIC_DEFAULT, base_url=base)
    if provider == "gemini":
        return GeminiClient(model=model or _GEMINI_DEFAULT, base_url=base)
    if provider == "openai":
        return OpenAIClient(model=model or _OPENAI_DEFAULT, base_url=base)
    raise ConfigError(
        f"Unknown LLM provider '{provider}' (expected 'anthropic', 'gemini', or 'openai')"
    )


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
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return GeminiClient()
    return None
