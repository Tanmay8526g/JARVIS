"""Offline deterministic routing tests."""
from jarvis.llm.router import choose_provider


def test_gemini_first_prefers_gemini():
    """Gemini-first sends model work to Gemini when available."""
    assert choose_provider("hello", "gemini_first", {"gemini": True, "ollama": True}) == "gemini"


def test_smart_routes_casual_local_and_live_gemini():
    """Smart mode separates casual conversation from live/tool work."""
    available = {"gemini": True, "ollama": True}
    assert choose_provider("hello", "smart", available) == "ollama"
    assert choose_provider("what is the weather in Jaipur", "smart", available) == "gemini"


def test_local_only_never_selects_gemini():
    """Local-only mode always selects Ollama."""
    assert choose_provider("latest news", "local_only", {"gemini": True, "ollama": True}) == "ollama"


def test_unavailable_preferred_provider_uses_other():
    """The router uses the available provider when its preference is unavailable."""
    assert choose_provider("latest news", "gemini_first", {"gemini": False, "ollama": True}) == "ollama"
