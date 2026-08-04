from __future__ import annotations


class NeuramineRLError(Exception):
    """Base class for all neuraminerl errors."""


class ConfigError(NeuramineRLError):
    """Invalid or inconsistent configuration."""


class LLMError(NeuramineRLError):
    """An LLM provider call failed."""
