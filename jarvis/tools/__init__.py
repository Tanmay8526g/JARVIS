"""Tool registry and safe dispatch."""
from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable
from ..config import DEBUG, short
from .time_tools import get_current_datetime
from .web_tools import web_search, read_webpage
from .weather_tools import get_weather, get_air_quality
from .news_tools import get_news
from .wiki_tools import wikipedia_summary
from .system_tools import open_website, open_application, take_screenshot

LOGGER = logging.getLogger("jarvis")

@dataclass(frozen=True)
class ToolDefinition:
    """Describe a registered tool."""
    function: Callable[..., str]
    direct_answer: bool = False

TOOL_DEFINITIONS = [ToolDefinition(get_current_datetime, True), ToolDefinition(web_search), ToolDefinition(get_weather, True), ToolDefinition(get_air_quality, True), ToolDefinition(get_news), ToolDefinition(wikipedia_summary), ToolDefinition(read_webpage), ToolDefinition(open_website, True), ToolDefinition(open_application, True), ToolDefinition(take_screenshot, True)]
TOOL_REGISTRY = {item.function.__name__: item for item in TOOL_DEFINITIONS}
TOOLS = [item.function for item in TOOL_DEFINITIONS]

def run_tool(name: str, arguments: dict[str, Any]) -> str:
    """Run a known tool and return a string, never raising to the caller."""
    started = time.perf_counter()
    definition = TOOL_REGISTRY.get(name)
    if not definition:
        return f"Unknown tool requested: {name}"
    try:
        result = short(definition.function(**arguments))
        LOGGER.info("tool=%s arguments=%s success=true duration=%.3f", name, short(arguments), time.perf_counter() - started)
        if DEBUG:
            print(f"[DEBUG] tool={name} arguments={arguments}")
        return result
    except Exception as error:
        LOGGER.error("tool=%s arguments=%s success=false duration=%.3f error=%s", name, short(arguments), time.perf_counter() - started, error)
        return f"Tool {name} failed: {error}"

def tool_key(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    """Make a stable key for per-turn deduplication."""
    return name, json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
