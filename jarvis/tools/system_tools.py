"""Allowlisted system actions with shared confirmation."""
from __future__ import annotations
import os
import webbrowser
from datetime import datetime
from urllib.parse import urlparse
from ..config import ALLOWED_APPS, ROOT

def confirm_action(action: str) -> bool:
    """Ask the user before a system-changing action."""
    # Keep confirmations in the terminal.
    try:
        answer = input(f"Jarvis wants to {action}. Do you want me to do this? (y/n) ")
        return answer.strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def _clean_model_text(value: str) -> str:
    """Remove control characters from text supplied by the model."""
    return "".join(character for character in str(value) if character.isprintable())

def open_website(url: str) -> str:
    """Confirm and open an HTTP(S) website."""
    try:
        cleaned_url = _clean_model_text(url)
        parsed = urlparse(cleaned_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Only http and https URLs with a domain are allowed."
        display_url = cleaned_url[:100]
        action = f"open website {parsed.netloc} ({display_url})"
        if not confirm_action(action):
            return "The user declined to open the website."
        webbrowser.open(cleaned_url)
        return f"Opened {cleaned_url} in the default browser."
    except Exception as error:
        return f"Could not open website: {error}"

def open_application(name: str) -> str:
    """Confirm and open an allowlisted application."""
    try:
        key = str(name).strip().lower()
        if key not in ALLOWED_APPS:
            return f"Application is not allowlisted. Choose: {', '.join(ALLOWED_APPS)}."
        if not confirm_action(f"open {key}"):
            return "The user declined to open the application."
        target = ALLOWED_APPS[key]
        if target.startswith(("http://", "https://")):
            webbrowser.open(target)
        else:
            os.startfile(target)
        return f"Opened {key}."
    except Exception as error:
        return f"Could not open application: {error}"

def take_screenshot() -> str:
    """Confirm and save a screenshot."""
    try:
        if not confirm_action("take and save a screenshot"):
            return "The user declined to take a screenshot."
        from PIL import ImageGrab

        screenshot_folder = ROOT / "workspace" / "screenshots"
        screenshot_folder.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = screenshot_folder / f"screenshot_{timestamp}.png"
        counter = 1
        while path.exists():
            path = screenshot_folder / f"screenshot_{timestamp}_{counter}.png"
            counter += 1
        ImageGrab.grab(all_screens=True).save(path)
        return f"Screenshot saved to {path}."
    except Exception as error:
        return f"Could not take screenshot: {error}"
