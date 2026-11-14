"""Safe offline text-to-speech using pyttsx3."""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from .. import config

LOGGER = logging.getLogger("jarvis")

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


def _remove_emoji(text: str) -> str:
    """Remove common emoji characters while keeping readable text."""
    return "".join(
        character
        for character in text
        if not (
            0x1F000 <= ord(character) <= 0x1FAFF
            or 0x2600 <= ord(character) <= 0x27BF
            or (unicodedata.category(character) == "So" and ord(character) > 127)
        )
    )


def _truncate_sentences(text: str) -> str:
    """Keep the first complete sentences within the configured limit."""
    limit = max(1, int(config.VOICE_MAX_CHARS))
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected = ""
    for sentence in sentences:
        candidate = f"{selected} {sentence}".strip()
        if len(candidate) > limit:
            break
        selected = candidate
    if not selected:
        selected = text[:limit].rstrip()
    print("(full answer shown on screen)")
    return selected


def clean_for_speech(text: str) -> str:
    """Remove links, markup, source labels, emoji, and debug text."""
    if not isinstance(text, str):
        return ""
    lines: list[str] = []
    for line in text.splitlines():
        if line.strip().lower() == "links:":
            break
        if "[DEBUG]" in line:
            continue
        if re.match(r"^\s*\d+\.\s+https?://", line):
            continue
        line = re.sub(r"https?://\S+", "", line)
        line = re.sub(r"\s*(?:—|-)?\s*Source:\s*[^.\n]+\.?", "", line, flags=re.IGNORECASE)
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"[*#`\[\]()]", "", cleaned)
    cleaned = _remove_emoji(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _truncate_sentences(cleaned)


def speak(text: str) -> bool:
    """Speak text with a new pyttsx3 engine and never raise an exception."""
    if not config.VOICE_ENABLED:
        LOGGER.info("speech success=false reason=disabled")
        return False
    if not text:
        LOGGER.info("speech success=false reason=empty_text")
        return False
    if pyttsx3 is None:
        LOGGER.error("speech success=false reason=pyttsx3_import_error")
        return False
    engine: Any = None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", config.VOICE_RATE)
        engine.setProperty("volume", config.VOICE_VOLUME)
        if config.VOICE_ID:
            engine.setProperty("voice", config.VOICE_ID)
        engine.say(text)
        engine.runAndWait()
        LOGGER.info("speech success=true")
        return True
    except Exception as error:
        LOGGER.error("speech success=false reason=%s", type(error).__name__)
        return False
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


def list_voices() -> list[tuple[str, str, str]]:
    """Return installed voice id, name, and language tuples safely."""
    if pyttsx3 is None:
        return []
    engine: Any = None
    try:
        engine = pyttsx3.init()
        voices = []
        for voice in engine.getProperty("voices") or []:
            languages = getattr(voice, "languages", []) or []
            language = languages[0] if languages else ""
            if isinstance(language, bytes):
                language = language.decode(errors="replace")
            voices.append((str(getattr(voice, "id", "")), str(getattr(voice, "name", "")), str(language)))
        return voices
    except Exception:
        return []
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
