"""Ollama provider adapter."""
from __future__ import annotations

import json
from typing import Any

from ollama import chat

from .base_types import NormalizedResponse, ToolCall


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from dict-style or object-style SDK data."""
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class OllamaProvider:
    """Adapter preserving the existing Ollama chat behavior."""

    name = "ollama"

    def __init__(self, model: str) -> None:
        """Create an Ollama provider."""
        self.model = model

    def chat(self, messages: list[dict[str, Any]], tools: list[Any]) -> NormalizedResponse:
        """Call Ollama and normalize text and tool calls."""
        response = chat(model=self.model, messages=messages, tools=tools)
        message = _field(response, "message", {})
        calls: list[ToolCall] = []
        for call in _field(message, "tool_calls", []) or []:
            function = _field(call, "function", {})
            arguments = _field(function, "arguments", {}) or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(ToolCall(_field(function, "name", ""), arguments))
        return NormalizedResponse(text=_field(message, "content", "") or "", tool_calls=calls)
