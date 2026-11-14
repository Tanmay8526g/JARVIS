"""Offline voice-output tests."""
from unittest.mock import Mock, patch

from jarvis import config, main
from jarvis.voice import tts


def test_clean_for_speech_removes_markup_links_sources_and_truncates(monkeypatch, capsys):
    """Speech text excludes screen-only link details and stays bounded."""
    monkeypatch.setattr(config, "VOICE_MAX_CHARS", 45)
    text = "# Time *now* is 6:25 PM. This is a longer spoken answer for testing. Source: Clock.\nLinks:\n1. https://example.com\n😀"
    result = tts.clean_for_speech(text)
    assert "https://" not in result
    assert "Source:" not in result
    assert "#" not in result and "*" not in result
    assert "6:25 PM" in result
    assert len(result) <= 45
    assert "full answer shown" in capsys.readouterr().out


def test_speak_returns_false_without_raising_when_engine_fails(monkeypatch):
    """Engine errors are converted to a False result."""
    monkeypatch.setattr(config, "VOICE_ENABLED", True)
    fake_engine = Mock()
    fake_engine.runAndWait.side_effect = RuntimeError("engine failure")
    with patch.object(tts.pyttsx3, "init", return_value=fake_engine):
        assert tts.speak("hello") is False


def test_voice_disabled_never_calls_engine(monkeypatch):
    """Disabled voice output does not initialize pyttsx3."""
    monkeypatch.setattr(config, "VOICE_ENABLED", False)
    with patch.object(tts.pyttsx3, "init") as init:
        assert tts.speak("hello") is False
    init.assert_not_called()


def test_voice_off_command_does_not_call_model(monkeypatch):
    """The voice-off command is handled in Python."""
    monkeypatch.setattr(config, "VOICE_ENABLED", True)
    with patch.object(main, "answer") as model:
        assert main._handle_voice_command("voice off") is True
    model.assert_not_called()
    assert config.VOICE_ENABLED is False


def test_doctor_output_is_not_spoken(capsys):
    """Doctor diagnostics do not invoke speech."""
    with patch.object(main, "speak") as speech:
        assert main._handle_doctor_command("doctor") is True
    speech.assert_not_called()
    assert "Voice:" in capsys.readouterr().out
