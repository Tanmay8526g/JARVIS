"""Google Gemini provider using the official google-genai SDK."""
from __future__ import annotations

import inspect
from typing import Any

from google import genai
from google.genai import types

from .. import config
from .base_types import MissingAPIKeyError, NormalizedResponse, ToolCall


def _parameter_schema(function: Any) -> dict[str, Any]:
    """Build a small JSON schema from an existing Jarvis tool signature."""
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []
    for name, parameter in inspect.signature(function).parameters.items():
        if name == "self":
            continue
        annotation = parameter.annotation
        kind = "integer" if annotation is int else "string"
        properties[name] = {"type": kind}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _contents(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Convert common messages into text and a system instruction."""
    system: list[str] = []
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = str(message.get("content", ""))
        if role == "system":
            system.append(content)
        elif role == "tool":
            lines.append(f"Tool result ({message.get('name', 'tool')}): {content}")
        else:
            lines.append(f"{role.title()}: {content}")
    return "\n".join(system), "\n".join(lines)


class GeminiProvider:
    """Adapter for Gemini function calling with automatic execution disabled."""

    name = "gemini"

    def __init__(self, model: str) -> None:
        """Create a Gemini provider using only GEMINI_API_KEY."""
        key = config.gemini_api_key()
        if not key:
            raise MissingAPIKeyError("GEMINI_API_KEY is missing")
        self.model = model
        self._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=config.TIMEOUT * 1000))

    def chat(self, messages: list[dict[str, Any]], tools: list[Any]) -> NormalizedResponse:
        """Call Gemini and normalize text and function calls."""
        system, contents = _contents(messages)
        declarations = [
            types.FunctionDeclaration(
                name=function.__name__,
                description=inspect.getdoc(function) or function.__name__,
                parameters_json_schema=_parameter_schema(function),
            )
            for function in tools
        ]
        tool_config = types.Tool(function_declarations=declarations) if declarations else None
        request_config = types.GenerateContentConfig(
            system_instruction=system or None,
            tools=[tool_config] if tool_config else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = self._client.models.generate_content(model=self.model, contents=contents, config=request_config)
        calls: list[ToolCall] = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                function_call = getattr(part, "function_call", None)
                if function_call:
                    calls.append(ToolCall(function_call.name, dict(function_call.args or {})))
        return NormalizedResponse(text=getattr(response, "text", "") or "", tool_calls=calls)
