"""Provider response data models without selection logic."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

class MissingAPIKeyError(RuntimeError):
    """Raised when Gemini is selected without its required environment variable."""

@dataclass
class ToolCall:
    """A normalized model-requested tool call."""
    name: str
    arguments: dict[str, Any]

@dataclass
class NormalizedResponse:
    """A normalized final response or list of tool calls."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    notice: str = ""
