from __future__ import annotations

import pytest

from neuraminerl.exceptions import ConfigError, LLMError
from neuraminerl.llm import AnthropicClient, GeminiClient, OpenAIClient, detect, from_spec
from neuraminerl.llm.gemini import _sanitize_schema
from neuraminerl.reflection.prompts import MERGE_SCHEMA, REFLECTION_SCHEMA

ALL_KEY_VARS = (
    "NEURAMINERL_LLM",
    "NEURAMINERL_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def test_spec_provider_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = from_spec("anthropic:claude-haiku-4-5")
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-haiku-4-5"


def test_spec_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    client = from_spec("openai")
    assert isinstance(client, OpenAIClient)
    assert client.model  # falls back to the adapter default


def test_spec_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client = from_spec("gemini:gemini-2.5-flash")
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-2.5-flash"


def test_spec_base_url_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEURAMINERL_API_KEY", "groq-key")
    client = from_spec("openai:llama-3.1-70b@https://api.groq.com/openai/v1")
    assert isinstance(client, OpenAIClient)
    assert client.model == "llama-3.1-70b"
    assert client._base_url == "https://api.groq.com/openai/v1"


def test_spec_keyless_local_server() -> None:
    # Ollama/vLLM style: custom base_url, no API key anywhere.
    client = from_spec("openai:qwen2.5@http://localhost:11434/v1")
    assert isinstance(client, OpenAIClient)


def test_official_endpoint_requires_key() -> None:
    with pytest.raises(LLMError):
        OpenAIClient()
    with pytest.raises(LLMError):
        AnthropicClient()
    with pytest.raises(LLMError):
        GeminiClient()


def test_neuraminerl_api_key_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEURAMINERL_API_KEY", "k")
    assert isinstance(OpenAIClient(), OpenAIClient)
    assert isinstance(AnthropicClient(), AnthropicClient)
    assert isinstance(GeminiClient(), GeminiClient)


def test_spec_unknown_provider() -> None:
    with pytest.raises(ConfigError):
        from_spec("cohere:command-r")


def test_detect_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    assert detect() is None
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert isinstance(detect(), GeminiClient)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert isinstance(detect(), OpenAIClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(detect(), AnthropicClient)
    monkeypatch.setenv("NEURAMINERL_LLM", "openai:qwen2.5@http://localhost:11434/v1")
    override = detect()
    assert isinstance(override, OpenAIClient)
    assert override.model == "qwen2.5"


def test_gemini_schema_sanitization() -> None:
    for schema in (REFLECTION_SCHEMA, MERGE_SCHEMA):
        cleaned = _sanitize_schema(schema)

        def has_unsupported(node: object) -> bool:
            if isinstance(node, dict):
                if "additionalProperties" in node or "$schema" in node:
                    return True
                return any(has_unsupported(v) for v in node.values())
            if isinstance(node, list):
                return any(has_unsupported(v) for v in node)
            return False

        assert not has_unsupported(cleaned)
        # Structure that Gemini needs must survive.
        assert cleaned["type"] == "object"
        assert "properties" in cleaned
