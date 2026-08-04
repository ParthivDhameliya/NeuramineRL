from __future__ import annotations


class NeuramineError(Exception):
    """Base class for all neuramine errors."""


class ConfigError(NeuramineError):
    """Invalid or inconsistent configuration."""


class LLMError(NeuramineError):
    """An LLM provider call failed."""
