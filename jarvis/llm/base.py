"""Common provider response types and provider selection."""
from __future__ import annotations

import logging
import importlib.util
import time
from typing import Any, Protocol

from .. import config
from .base_types import MissingAPIKeyError, NormalizedResponse, ToolCall
from .gemini_provider import GeminiProvider
from .ollama_provider import OllamaProvider

LOGGER = logging.getLogger("jarvis")


class LLMProvider(Protocol):
    """Interface implemented by each language-model provider."""
    name: str
    model: str

    def chat(self, messages: list[dict[str, Any]], tools: list[Any]) -> NormalizedResponse:
        """Send conversation and tool declarations to the provider."""


def _retryable(error: Exception) -> bool:
    """Identify provider failures that should be retried once."""
    text = str(error).lower()
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    return status in {401, 408, 429, 500, 502, 503, 504} or any(word in text for word in ("429", "rate limit", "timeout", "timed out", "network", "connection", "connect", "dns", "unauthorized", "invalid api key", "api key"))


class ProviderManager:
    """Select a provider and fall back safely when it is unavailable."""

    def __init__(self) -> None:
        """Create a manager using current configuration values."""
        self.active_name = self._valid_name(config.LLM_PROVIDER)
        self._missing_reported = False
        self._last_reasons: dict[str, str] = {}

    @staticmethod
    def _valid_name(name: str) -> str:
        """Return a supported provider name, defaulting to Ollama."""
        return name if name in {"ollama", "gemini"} else "ollama"

    def _provider(self, name: str) -> LLMProvider:
        """Create the configured provider without exposing secrets."""
        if name == "gemini":
            return GeminiProvider(config.GEMINI_MODEL)
        return OllamaProvider(config.MODEL)

    def set_provider(self, name: str) -> bool:
        """Set the active provider for this session."""
        if name not in {"ollama", "gemini"}:
            return False
        self.active_name = name
        return True

    def model_name(self) -> str:
        """Return the active provider's configured model name."""
        return config.GEMINI_MODEL if self.active_name == "gemini" else config.MODEL

    def availability(self) -> dict[str, bool]:
        """Return local availability without making a network call."""
        return {"ollama": True, "gemini": bool(config.gemini_api_key())}

    def status(self) -> str:
        """Return a user-safe provider status line."""
        return f"Provider: {self.active_name} (model: {self.model_name()})"

    def status_lines(self) -> list[str]:
        """Return safe readiness lines for both providers."""
        ollama_reason = self._last_reasons.get("ollama", "local provider configured")
        if config.gemini_api_key():
            if importlib.util.find_spec("google.genai") is None:
                gemini_reason = "SDK google-genai is not installed"
            else:
                gemini_reason = self._last_reasons.get("gemini", "SDK installed and API key present")
        else:
            gemini_reason = "GEMINI_API_KEY is missing"
        return [
            f"ollama: available (model={config.MODEL}; reason={ollama_reason})",
            f"gemini: {'available' if config.gemini_api_key() and importlib.util.find_spec('google.genai') else 'unavailable'} (model={config.GEMINI_MODEL}; reason={gemini_reason})",
        ]

    def chat(self, messages: list[dict[str, Any]], tools: list[Any]) -> NormalizedResponse:
        """Call the active provider, retry once, then use the fallback provider."""
        provider_name = self.active_name
        missing_key = False
        try:
            return self._attempt(provider_name, messages, tools)
        except MissingAPIKeyError:
            self._last_reasons[provider_name] = "reason=missing GEMINI_API_KEY"
            self._report_missing_key()
            missing_key = True
        except Exception as error:
            if _retryable(error):
                time.sleep(0.2)
                try:
                    return self._attempt(provider_name, messages, tools)
                except Exception:
                    pass
            else:
                LOGGER.error("provider=%s success=false error_type=%s", provider_name, type(error).__name__)
        fallback_name = self._valid_name(config.LLM_FALLBACK_PROVIDER)
        if fallback_name == provider_name:
            raise RuntimeError("The language model is unavailable.")
        try:
            # A local fallback gets plain conversation only.
            fallback_tools = [] if provider_name == "gemini" and fallback_name == "ollama" else tools
            response = self._attempt(fallback_name, messages, fallback_tools)
            reason = self._last_reasons.get(provider_name, "reason=provider failure")
            response.notice = f"{provider_name.capitalize()} is unavailable ({reason}); using local model."
            return response
        except Exception as error:
            LOGGER.error("provider=%s success=false error_type=%s", fallback_name, type(error).__name__)
            raise RuntimeError("The language model is unavailable.") from error

    def _attempt(self, name: str, messages: list[dict[str, Any]], tools: list[Any]) -> NormalizedResponse:
        """Call one provider and record safe timing metadata."""
        started = time.perf_counter()
        provider_messages = [message for message in messages if not (name == "gemini" and message.get("private"))]
        provider = None
        try:
            provider = self._provider(name)
            response = provider.chat(provider_messages, tools)
            LOGGER.info("provider=%s model=%s duration=%.3f success=true", name, provider.model, time.perf_counter() - started)
            return response
        except MissingAPIKeyError:
            self._last_reasons[name] = "reason=missing GEMINI_API_KEY"
            LOGGER.warning("provider=%s model=%s duration=%.3f success=false error_type=missing_key", name, config.GEMINI_MODEL if name == "gemini" else config.MODEL, time.perf_counter() - started)
            raise
        except Exception as error:
            status = getattr(error, "status_code", None) or getattr(error, "code", None) or "unknown"
            self._last_reasons[name] = f"error_type={type(error).__name__}, status={status}"
            LOGGER.error("provider=%s model=%s duration=%.3f success=false error_type=%s status=%s", name, config.GEMINI_MODEL if name == "gemini" else config.MODEL, time.perf_counter() - started, type(error).__name__, status)
            raise

    def _report_missing_key(self) -> None:
        """Print the missing-key message once without revealing any secret."""
        if not self._missing_reported:
            print("Gemini is unavailable (reason=missing GEMINI_API_KEY); using local model.")
            self._missing_reported = True
