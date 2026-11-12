"""Terminal chat loop."""
from __future__ import annotations
from typing import Any
import logging
from pathlib import Path
from .brain import answer
from .brain import PROVIDER_MANAGER
from . import config
from .config import LOG_FILE

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def main() -> None:
    """Run Jarvis's terminal chat loop."""
    history: list[dict[str, Any]] = [{"role":"system","content":"You are Jarvis. Use tools for current facts, never guess, mention source names or URLs for web answers, and ask consent through tools before system actions."}]
    private_mode = False
    print(f"Jarvis is ready (mode: {config.ROUTING_MODE}). Type 'exit' to quit.")
    for status_line in PROVIDER_MANAGER.status_lines():
        print(status_line)
    print("Commands: doctor, mode, mode smart, mode gemini_first, mode local_only, provider, private on, private off")
    while True:
        prompt = "[private] You: " if private_mode else "You: "
        try: user_text = input(prompt).strip()
        except (KeyboardInterrupt, EOFError): print("\nJarvis: Goodbye!"); break
        if user_text.lower() == "exit": print("Jarvis: Goodbye!"); break
        if not user_text: continue
        if _handle_provider_command(user_text): continue
        if _handle_doctor_command(user_text): continue
        if _handle_mode_command(user_text): continue
        private_command = _handle_private_command(user_text, private_mode)
        if private_command is not None:
            private_mode = private_command
            continue
        try: print("Jarvis:", answer(user_text, history, private_mode))
        except Exception as error:
            print("Jarvis: I can't reach the model right now. Is the Ollama app running?")
            status = getattr(error, "status_code", None) or getattr(error, "code", None) or "unknown"
            print(f"Details: error_type={type(error).__name__}, status={status}")


def _handle_provider_command(user_text: str) -> bool:
    """Handle provider commands without calling a language model."""
    command = user_text.strip().lower()
    if command == "provider":
        print(PROVIDER_MANAGER.status())
        return True
    if command.startswith("provider "):
        name = command.removeprefix("provider ").strip()
        if PROVIDER_MANAGER.set_provider(name):
            print(f"Provider switched to {name}.")
        else:
            print("Use: provider, provider ollama, or provider gemini")
        return True
    return False


def _env_is_gitignored() -> bool:
    """Check whether the project environment file is ignored."""
    try:
        lines = (config.ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    return any(line.strip() in {".env", "**/.env"} for line in lines)


def _handle_doctor_command(user_text: str) -> bool:
    """Show safe provider and project diagnostics without calling a model."""
    if user_text.strip().lower() != "doctor":
        return False
    for status_line in PROVIDER_MANAGER.status_lines():
        print(status_line)
    print(f"Routing mode: {config.ROUTING_MODE}")
    print(f".env git-ignored: {_env_is_gitignored()}")
    return True


def _handle_mode_command(user_text: str) -> bool:
    """Handle routing-mode commands without calling a language model."""
    command = user_text.strip().lower()
    if command == "mode":
        print(f"Mode: {config.ROUTING_MODE}; {PROVIDER_MANAGER.status()}")
        return True
    if command.startswith("mode "):
        mode = command.removeprefix("mode ").strip()
        if mode in {"smart", "gemini_first", "local_only"}:
            config.ROUTING_MODE = mode
            print(f"Routing mode switched to {mode}.")
        else:
            print("Use: mode, mode smart, mode gemini_first, or mode local_only")
        return True
    return False


def _handle_private_command(user_text: str, current: bool) -> bool | None:
    """Handle private-mode commands without calling a language model."""
    command = user_text.strip().lower()
    if command == "private on":
        print("Private mode on.")
        return True
    if command == "private off":
        print("Private mode off.")
        return False
    return None
