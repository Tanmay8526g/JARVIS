"""Central Jarvis settings."""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MODEL = "llama3.2:3b"
LLM_PROVIDER = "ollama"
LLM_FALLBACK_PROVIDER = "ollama"
GEMINI_MODEL = "gemini-3.5-flash-lite"
ROUTING_MODE = "gemini_first"
# Words that usually mean the answer needs live data, tools, or careful reasoning.
ROUTING_GEMINI_KEYWORDS = ("weather", "air quality", "aqi", "news", "latest", "current", "search", "look up", "summarize", "explain", "compare", "why", "how")
# Words that usually describe a short, casual conversation for the local model.
ROUTING_LOCAL_KEYWORDS = ("hello", "hi", "hey", "thanks", "thank you", "okay", "ok", "good morning", "good night")
# Short messages without a tool keyword are treated as casual in smart mode.
ROUTING_LOCAL_MAX_CHARS = 120
MAX_MESSAGES = 10
MAX_TOOL_ROUNDS = 4
TIMEOUT = 10
MAX_TOOL_CHARS = 3000
DEBUG = True
VOICE_ENABLED = True
VOICE_RATE = 180
VOICE_VOLUME = 1.0
VOICE_ID = ""
VOICE_MAX_CHARS = 400
USER_AGENT = "Jarvis/1.0 (personal assistant)"
ALLOWED_APPS = {"notepad": "notepad.exe", "calculator": "calc.exe", "browser": "https://www.google.com"}
ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = ROOT / "logs" / "jarvis.log"
CONFIG_FILE = ROOT / "config.json"
ENV_FILE = ROOT / ".env"
LOGGER = logging.getLogger("jarvis")


def _valid_app_target(value: Any) -> bool:
    """Return whether an allowlisted target is a safe name or web URL."""
    if not isinstance(value, str) or not value:
        return False
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return bool(parsed.netloc)
    return "/" not in value and "\\" not in value


def _valid_allowed_apps(value: Any) -> dict[str, str]:
    """Keep only safe application targets from configuration."""
    if not isinstance(value, dict):
        LOGGER.warning("Ignoring ALLOWED_APPS override because it is not an object")
        return ALLOWED_APPS
    valid: dict[str, str] = {}
    for name, target in value.items():
        if isinstance(name, str) and name and "/" not in name and "\\" not in name and _valid_app_target(target):
            valid[name] = target
        else:
            LOGGER.warning("Ignoring invalid ALLOWED_APPS entry: %r", name)
    return valid


def _load_dotenv() -> None:
    """Load simple KEY=VALUE entries from the project .env file."""
    if not ENV_FILE.exists():
        return
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except OSError:
        return


def gemini_api_key() -> str | None:
    """Return the Gemini key from the environment without logging it."""
    return os.environ.get("GEMINI_API_KEY")

def load_config() -> None:
    """Load known settings from config.json if it exists."""
    global MODEL, MAX_MESSAGES, MAX_TOOL_ROUNDS, TIMEOUT, MAX_TOOL_CHARS, DEBUG, USER_AGENT, ALLOWED_APPS
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}
        for name in ("MODEL", "LLM_PROVIDER", "LLM_FALLBACK_PROVIDER", "GEMINI_MODEL", "ROUTING_MODE", "ROUTING_GEMINI_KEYWORDS", "ROUTING_LOCAL_KEYWORDS", "ROUTING_LOCAL_MAX_CHARS", "MAX_MESSAGES", "MAX_TOOL_ROUNDS", "TIMEOUT", "MAX_TOOL_CHARS", "DEBUG", "VOICE_ENABLED", "VOICE_RATE", "VOICE_VOLUME", "VOICE_ID", "VOICE_MAX_CHARS", "USER_AGENT", "ALLOWED_APPS"):
            if name in data:
                globals()[name] = _valid_allowed_apps(data[name]) if name == "ALLOWED_APPS" else data[name]
    except (OSError, ValueError):
        return

def short(value: Any) -> str:
    """Convert a value to bounded text."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:MAX_TOOL_CHARS] + ("..." if len(text) > MAX_TOOL_CHARS else "")

_load_dotenv()
load_config()
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
