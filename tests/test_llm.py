"""Offline provider selection and fallback tests."""
from unittest.mock import patch

from jarvis import config
from jarvis import main
from jarvis.brain import answer
from jarvis.llm.base import ProviderManager
from jarvis.llm.base_types import MissingAPIKeyError, NormalizedResponse, ToolCall


def test_provider_selection_from_config(monkeypatch):
    """The configured provider is selected by the manager."""
    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    manager = ProviderManager()
    assert manager.active_name == "gemini"
    assert manager.model_name() == config.GEMINI_MODEL


def test_429_retries_then_falls_back(monkeypatch):
    """A rate limit retries once and then uses the fallback provider."""
    monkeypatch.setattr(config, "LLM_FALLBACK_PROVIDER", "ollama")
    manager = ProviderManager()
    manager.active_name = "gemini"
    with patch.object(manager, "_attempt", side_effect=[RuntimeError("429"), RuntimeError("429"), NormalizedResponse(text="local")]):
        response = manager.chat([], [])
    assert response.text == "local"
    assert "using local model" in response.notice


def test_missing_key_falls_back_without_exposing_key(monkeypatch, capsys, caplog):
    """Missing Gemini credentials produce a safe message and no secret text."""
    secret = "never-log-this-key"
    monkeypatch.setattr(config, "LLM_FALLBACK_PROVIDER", "ollama")
    manager = ProviderManager()
    manager.active_name = "gemini"
    with patch.object(manager, "_attempt", side_effect=[MissingAPIKeyError("GEMINI_API_KEY is missing"), NormalizedResponse(text="local")]):
        response = manager.chat([], [])
    assert response.text == "local"
    assert "GEMINI_API_KEY" in capsys.readouterr().out
    assert "reason=missing GEMINI_API_KEY" in response.notice
    assert secret not in caplog.text and secret not in str(response)


def test_failure_notice_contains_type_and_status(monkeypatch):
    """Fallback notices expose only the safe failure type and status."""
    class RateLimitError(Exception):
        """Represent a provider rate-limit response."""
        status_code = 429

    manager = ProviderManager()
    manager.active_name = "gemini"

    def fake_attempt(name, messages, tools):
        """Record the safe reason and fail Gemini before using Ollama."""
        if name == "gemini":
            manager._last_reasons[name] = "error_type=RateLimitError, status=429"
            raise RateLimitError("private provider detail")
        return NormalizedResponse(text="local")

    monkeypatch.setattr(config, "LLM_FALLBACK_PROVIDER", "ollama")
    with patch.object(manager, "_attempt", side_effect=fake_attempt):
        response = manager.chat([], [])
    assert "error_type=RateLimitError, status=429" in response.notice
    assert "private provider detail" not in response.notice


def test_provider_errors_do_not_log_key(monkeypatch, caplog):
    """Provider error logs contain only the exception type, never its text."""
    secret = "never-log-this-key"
    monkeypatch.setattr(config, "LLM_FALLBACK_PROVIDER", "ollama")
    manager = ProviderManager()
    manager.active_name = "gemini"
    with patch.object(manager, "_attempt", side_effect=[RuntimeError(secret), RuntimeError(secret), RuntimeError(secret)]):
        try:
            manager.chat([], [])
        except RuntimeError:
            pass
    assert secret not in caplog.text


def test_safety_logic_works_with_fake_provider():
    """The brain consumes a normalized response from any provider."""
    class FakeProvider:
        """Return one safe direct-answer tool call."""
        def chat(self, messages, tools):
            """Return a normalized time call."""
            return NormalizedResponse(tool_calls=[ToolCall("get_current_datetime", {})])

    with patch("jarvis.brain.PROVIDER_MANAGER.chat", side_effect=FakeProvider().chat), patch("jarvis.brain.run_tool", return_value="Wednesday, 12 November 2026, 06:25:00 PM (IST)"):
        result = answer("what time is it?", [{"role": "system", "content": "test"}])
    assert "It is 06:25:00 PM" in result


def test_provider_commands_do_not_call_model():
    """Provider commands are handled before answer/model execution."""
    with patch.object(main, "answer") as model:
        assert main._handle_provider_command("provider") is True
        assert main._handle_provider_command("provider ollama") is True
        assert main._handle_provider_command("provider gemini") is True
    model.assert_not_called()

def test_mode_and_private_commands_do_not_call_model(monkeypatch):
    """Routing and privacy commands are handled in Python."""
    monkeypatch.setattr(main.config, "ROUTING_MODE", "gemini_first")
    with patch.object(main, "answer") as model:
        assert main._handle_mode_command("mode smart") is True
        assert main._handle_mode_command("mode") is True
        assert main._handle_private_command("private on", False) is True
        assert main._handle_private_command("private off", True) is False
    model.assert_not_called()

def test_gemini_fallback_ollama_has_no_system_tools():
    """Ollama fallback after Gemini failure receives no tools."""
    manager = ProviderManager()
    manager.active_name = "gemini"
    offered = []
    def fake_attempt(name, messages, tools):
        """Record fallback tools and simulate provider failure."""
        offered.append((name, [tool.__name__ for tool in tools]))
        if name == "gemini":
            raise RuntimeError("429")
        return NormalizedResponse(text="local")
    with patch.object(manager, "_attempt", side_effect=fake_attempt):
        response = manager.chat([], [type("Tool", (), {"__name__": "open_website"})(), type("Tool", (), {"__name__": "web_search"})()])
    assert response.text == "local"
    assert offered[-1] == ("ollama", [])


def test_doctor_does_not_call_model(capsys):
    """Doctor diagnostics are handled without a model call."""
    with patch.object(main, "answer") as model:
        assert main._handle_doctor_command("doctor") is True
    model.assert_not_called()
    assert "Routing mode:" in capsys.readouterr().out


def test_casual_local_turn_has_no_tools(monkeypatch):
    """Casual conversation does not offer tools to the local model."""
    from jarvis import brain
    offered = []

    def fake_call(messages, tools):
        """Record tools and return a plain answer."""
        offered.append(tools)
        return NormalizedResponse(text="Here is a joke.")

    monkeypatch.setattr(brain.PROVIDER_MANAGER, "availability", lambda: {"gemini": False, "ollama": True})
    with patch("jarvis.brain._call", side_effect=fake_call):
        answer("tell me a joke", [{"role": "system", "content": "test"}])
    assert offered == [[]]
