"""Deterministic provider routing rules."""
from __future__ import annotations

from typing import Any, Iterable

from .. import config


def _available(availability: Any, provider: str) -> bool:
    """Read provider availability from a mapping or a set of names."""
    if isinstance(availability, dict):
        return bool(availability.get(provider, False))
    if isinstance(availability, (set, frozenset, list, tuple)):
        return provider in availability
    return True


def _contains_keyword(message: str, keywords: Iterable[str]) -> bool:
    """Return whether a message contains one of the configured keywords."""
    lower = message.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def choose_provider(user_message: str, mode: str, availability: Any) -> str:
    """Choose a provider using only deterministic local rules."""
    if mode == "local_only":
        preferred = "ollama"
    elif mode == "gemini_first":
        preferred = "gemini"
    else:
        needs_gemini = _contains_keyword(user_message, config.ROUTING_GEMINI_KEYWORDS)
        is_casual = len(user_message.strip()) <= config.ROUTING_LOCAL_MAX_CHARS
        has_local_phrase = _contains_keyword(user_message, config.ROUTING_LOCAL_KEYWORDS)
        preferred = "ollama" if is_casual and (has_local_phrase or not needs_gemini) else "gemini"
    if _available(availability, preferred):
        return preferred
    other = "ollama" if preferred == "gemini" else "gemini"
    return other if _available(availability, other) else preferred
