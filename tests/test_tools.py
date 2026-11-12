"""Offline registry and safety tests."""
from unittest.mock import patch
from pathlib import Path
from jarvis.tools import run_tool
from jarvis.brain import fallback
from jarvis.tools import system_tools

def test_unknown_tool_is_rejected():
    """Unknown names return an error string."""
    assert "Unknown tool" in run_tool("missing", {})

def test_fallback_punctuation_and_integer():
    """Fallback strips punctuation and uses integer max_items."""
    name, args = fallback("latest news about India?")
    assert name == "get_news" and args["topic"] == "india" and isinstance(args["max_items"], int)

def test_system_actions_use_shared_confirmation():
    """Every system action refuses a denied confirmation."""
    with patch.object(system_tools, "confirm_action", return_value=False) as confirm:
        assert "declined" in system_tools.open_website("https://example.com")
        assert "declined" in system_tools.open_application("notepad")
        assert "declined" in system_tools.take_screenshot()
        assert confirm.call_count == 3

def test_bad_input_returns_string():
    """Bad tool input does not escape as an exception."""
    assert isinstance(run_tool("web_search", {"query":"x", "max_results":"bad"}), str)
    assert isinstance(run_tool("read_webpage", {"url":"file:///bad"}), str)

def test_screenshot_path_is_inside_workspace(tmp_path):
    """Screenshots use the project workspace screenshots folder."""
    with patch.object(system_tools, "ROOT", tmp_path), patch.object(system_tools, "confirm_action", return_value=True), patch("PIL.ImageGrab.grab") as grab:
        grab.return_value.save = lambda path: None
        result = system_tools.take_screenshot()
    saved_path = Path(result.removeprefix("Screenshot saved to ").removesuffix("."))
    assert saved_path.parent == tmp_path / "workspace" / "screenshots"
    assert saved_path.name.startswith("screenshot_") and saved_path.suffix == ".png"

def test_website_prompt_cleans_newline_and_shows_domain():
    """Website confirmation text is clean and domain-focused."""
    with patch.object(system_tools, "confirm_action", return_value=False) as confirm:
        system_tools.open_website("https://example.com/hello\nmalicious")
    prompt = confirm.call_args.args[0]
    assert "example.com" in prompt and "\n" not in prompt

def test_unsafe_website_urls_are_rejected():
    """Javascript and file URLs are rejected before confirmation."""
    with patch.object(system_tools, "confirm_action") as confirm:
        assert "allowed" in system_tools.open_website("javascript:alert(1)")
        assert "allowed" in system_tools.open_website("file:///tmp/test")
        confirm.assert_not_called()

def test_unallowlisted_app_is_rejected():
    """Applications outside the allowlist cannot open."""
    with patch.object(system_tools, "confirm_action") as confirm:
        assert "not allowlisted" in system_tools.open_application("powershell")
        confirm.assert_not_called()

def test_declined_actions_do_not_open_or_save():
    """Denied confirmations prevent all system side effects."""
    with patch.object(system_tools, "confirm_action", return_value=False), patch.object(system_tools.webbrowser, "open") as browser_open, patch.object(system_tools.os, "startfile") as startfile, patch("PIL.ImageGrab.grab") as grab:
        system_tools.open_website("https://example.com")
        system_tools.open_application("notepad")
        system_tools.take_screenshot()
    browser_open.assert_not_called()
    startfile.assert_not_called()
    grab.assert_not_called()
