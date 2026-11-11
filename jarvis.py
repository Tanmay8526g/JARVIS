"""Jarvis: a local Ollama assistant with safe, free information tools."""
from __future__ import annotations

import json
import os
import re
import webbrowser
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus, urlparse

import feedparser
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from ollama import chat

MODEL = "llama3.2:3b"
MAX_MESSAGES = 10
MAX_TOOL_ROUNDS = 4
TIMEOUT = 10
MAX_TOOL_CHARS = 3000
DEBUG = True
USER_AGENT = "Jarvis/1.0 (personal assistant)"


def _short(value: Any) -> str:
    """Convert a value to a bounded string for the model context."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:MAX_TOOL_CHARS] + ("..." if len(text) > MAX_TOOL_CHARS else "")


def _json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fetch JSON with a timeout and a descriptive User-Agent."""
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def _ask(action: str) -> bool:
    """Ask before an action that opens something or writes a file."""
    try:
        return input(f"Jarvis wants to {action}. Do you want me to do this? (y/n) ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def get_current_datetime() -> str:
    """Return the computer's current local weekday, date, time, and timezone."""
    return datetime.now().astimezone().strftime("%A, %d %B %Y, %I:%M:%S %p (%Z)")


def web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return each result's title, snippet, and URL."""
    try:
        results = DDGS().text(query, max_results=max(1, min(max_results, 10)))
        rows = [{"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")} for r in results]
        return _short(rows) if rows else "No search results were found."
    except Exception as error:
        return f"Web search failed: {error}"


def _place(city: str) -> dict[str, Any] | None:
    """Find the first Open-Meteo geocoding result for a city."""
    data = _json("https://geocoding-api.open-meteo.com/v1/search", {"name": city, "count": 1, "language": "en", "format": "json"})
    return (data.get("results") or [None])[0]


def get_weather(city: str) -> str:
    """Return current Open-Meteo temperature, feels-like, humidity, wind, rain, and conditions."""
    try:
        place = _place(city)
        if not place:
            return f"No location was found for {city}."
        current = _json("https://api.open-meteo.com/v1/forecast", {
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,rain,weather_code",
            "timezone": "auto",
        }).get("current", {})
        codes = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle", 55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "heavy rain showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}
        return _short({"location": f"{place.get('name')}, {place.get('country')}", "time": current.get("time"), "temperature_c": current.get("temperature_2m"), "feels_like_c": current.get("apparent_temperature"), "humidity_percent": current.get("relative_humidity_2m"), "wind_kmh": current.get("wind_speed_10m"), "rain_mm": current.get("rain"), "conditions": codes.get(current.get("weather_code"), "unknown"), "source": "Open-Meteo"})
    except Exception as error:
        return f"Weather lookup failed: {error}"


def get_air_quality(city: str) -> str:
    """Return PM2.5, PM10, US AQI, and European AQI for a city."""
    try:
        place = _place(city)
        if not place:
            return f"No location was found for {city}."
        current = _json("https://air-quality-api.open-meteo.com/v1/air-quality", {
            "latitude": place["latitude"], "longitude": place["longitude"],
            "current": "pm2_5,pm10,us_aqi,european_aqi", "timezone": "auto",
        }).get("current", {})
        return _short({"location": f"{place.get('name')}, {place.get('country')}", "time": current.get("time"), "pm2_5_ug_m3": current.get("pm2_5"), "pm10_ug_m3": current.get("pm10"), "us_aqi": current.get("us_aqi"), "european_aqi": current.get("european_aqi"), "important": "AQI labels are Open-Meteo's European AQI scale; US AQI is separate. Neither is India's national AQI scale.", "source": "Open-Meteo"})
    except Exception as error:
        return f"Air-quality lookup failed: {error}"


def get_news(topic: str = "", max_items: int = 5) -> str:
    """Read public RSS feeds and return headlines, source, published time, and link."""
    try:
        urls = [f"https://news.google.com/rss/search?q={quote_plus(topic)}&hl=en-IN&gl=IN&ceid=IN:en"] if topic else ["https://feeds.bbci.co.uk/news/world/rss.xml", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"]
        rows: list[dict[str, str]] = []
        for url in urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max(1, min(max_items, 10))]:
                rows.append({"headline": entry.get("title", ""), "source": feed.feed.get("title", "RSS"), "published": entry.get("published", entry.get("updated", "")), "link": entry.get("link", "")})
        return _short(rows[:max(1, min(max_items, 10))]) if rows else "No news items were found."
    except Exception as error:
        return f"News lookup failed: {error}"


def wikipedia_summary(topic: str) -> str:
    """Return a summary and source URL from Wikipedia's public REST API."""
    try:
        data = _json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(topic.strip().replace(' ', '_'))}", {})
        return _short({"title": data.get("title"), "summary": data.get("extract"), "url": data.get("content_urls", {}).get("desktop", {}).get("page"), "source": "Wikipedia"})
    except Exception as error:
        return f"Wikipedia lookup failed: {error}"


def read_webpage(url: str) -> str:
    """Download an HTTP(S) page and extract a bounded amount of readable text."""
    try:
        if urlparse(url).scheme not in {"http", "https"}:
            return "Only http and https URLs are allowed."
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()
        return _short(" ".join(soup.get_text(" ", strip=True).split())) or "The page contained no readable text."
    except Exception as error:
        return f"Webpage read failed: {error}"


def open_website(url: str) -> str:
    """Ask for consent, then open an HTTP(S) URL in the default browser."""
    try:
        if urlparse(url).scheme not in {"http", "https"}:
            return "Only http and https URLs are allowed."
        if not _ask(f"open {url} in your default browser"):
            return "The user declined to open the website."
        webbrowser.open(url)
        return f"Opened {url} in the default browser."
    except Exception as error:
        return f"Could not open website: {error}"


def open_application(name: str) -> str:
    """Ask for consent, then open only notepad, calculator, or browser."""
    try:
        allowed = {"notepad": "notepad.exe", "calculator": "calc.exe", "browser": "https://www.google.com"}
        key = name.strip().lower()
        if key not in allowed:
            return f"Application is not allowlisted. Choose: {', '.join(allowed)}."
        if not _ask(f"open {key}"):
            return "The user declined to open the application."
        target = allowed[key]
        webbrowser.open(target) if target.startswith("http") else os.startfile(target)
        return f"Opened {key}."
    except Exception as error:
        return f"Could not open application: {error}"


def take_screenshot() -> str:
    """Ask for consent, then save a screenshot as jarvis_screenshot.png."""
    try:
        if not _ask("take and save a screenshot as jarvis_screenshot.png"):
            return "The user declined to take a screenshot."
        from PIL import ImageGrab
        path = os.path.abspath("jarvis_screenshot.png")
        ImageGrab.grab(all_screens=True).save(path)
        return f"Screenshot saved to {path}."
    except Exception as error:
        return f"Could not take screenshot: {error}"


TOOLS = [get_current_datetime, web_search, get_weather, get_air_quality, get_news, wikipedia_summary, read_webpage, open_website, open_application, take_screenshot]
TOOL_MAP = {tool.__name__: tool for tool in TOOLS}


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from Ollama's dict-style or object-style response."""
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _fallback(text: str) -> tuple[str, dict[str, Any]] | None:
    """Select an obvious live-data tool if the small model skipped tool calling."""
    lower = text.lower()
    if any(x in lower for x in ("what time", "current time", "what date", "today's date")):
        return "get_current_datetime", {}
    match = re.search(r"(?:weather|air quality|aqi)\s+(?:in|for)\s+([a-z .'-]+)", lower)
    if match:
        city = match.group(1).strip(" ?.!\"")
        return ("get_air_quality", {"city": city}) if "air quality" in lower or "aqi" in lower else ("get_weather", {"city": city})
    if "news" in lower or "latest" in lower or "current events" in lower:
        topic = re.sub(r"\b(latest|news|about|the)\b", " ", lower, flags=re.I).strip() or "world"
        return "get_news", {"topic": topic, "max_items": 5}
    if any(x in lower for x in ("who is the current", "current prime minister", "current president", "today", "score", "price")):
        return "web_search", {"query": text, "max_results": 5}
    return None


def _run(name: str, arguments: dict[str, Any]) -> str:
    """Run a registered tool and always return a safe bounded string."""
    if DEBUG:
        print(f"[DEBUG] tool={name} arguments={arguments}")
    tool = TOOL_MAP.get(name)
    if not tool:
        return f"Unknown tool requested: {name}"
    try:
        return _short(tool(**arguments))
    except Exception as error:
        return f"Tool {name} failed: {error}"


def _is_refusal(text: str) -> bool:
    """Recognize common small-model refusals to use a supplied live result."""
    lower = text.lower()
    return any(phrase in lower for phrase in ("i don't know", "i do not know", "not possible", "don't have the ability", "not allowed", "don't have real-time", "can't access real-time", "cannot access real-time", "unable to access real-time"))


def _failed_result(text: str) -> bool:
    """Identify a tool failure or empty result that must not be embellished."""
    lower = text.lower()
    return lower.startswith(("web search failed:", "weather lookup failed:", "air-quality lookup failed:", "news lookup failed:", "wikipedia lookup failed:", "webpage read failed:", "tool ")) or "no search results were found" in lower or "no news items were found" in lower or "no location was found" in lower


def _call(messages: list[dict[str, Any]]) -> Any:
    """Call Ollama with Python functions as tool schemas."""
    return chat(model=MODEL, messages=messages, tools=TOOLS)


def answer(user_text: str, history: list[dict[str, Any]]) -> str:
    """Answer one turn with multiple bounded rounds of model-requested tools."""
    history.append({"role": "user", "content": user_text})
    messages: list[dict[str, Any]] = [history[0]] + history[1:][-MAX_MESSAGES:]
    try:
        failed_result: str | None = None
        for _ in range(MAX_TOOL_ROUNDS):
            response = _call(messages)
            message = _field(response, "message", {})
            calls = _field(message, "tool_calls", []) or []
            if not calls:
                reply = _field(message, "content", "") or "I could not produce an answer."
                if failed_result:
                    reply = failed_result
                fallback = _fallback(user_text)
                if fallback:
                    name, args = fallback
                    result = _run(name, args)
                    if _failed_result(result):
                        history.append({"role": "assistant", "content": result})
                        return result
                    try:
                        fallback_response = _call([messages[0], {"role": "user", "content": user_text}, {"role": "tool", "content": result}])
                        reply = _field(_field(fallback_response, "message", {}), "content", "") or result
                        if _is_refusal(reply):
                            reply = result
                    except Exception:
                        reply = result
                history.append({"role": "assistant", "content": reply})
                return reply
            messages.append({"role": "assistant", "content": _field(message, "content", ""), "tool_calls": calls})
            for call in calls:
                function = _field(call, "function", {})
                name, args = _field(function, "name", ""), _field(function, "arguments", {}) or {}
                if isinstance(args, str):
                    args = json.loads(args)
                result = _run(name, args)
                if _failed_result(result):
                    failed_result = result
                messages.append({"role": "tool", "content": result})
            messages = [messages[0]] + messages[1:][-MAX_MESSAGES:]
        return "I stopped after the maximum number of tool rounds. Please try a simpler question."
    except Exception as error:
        fallback = _fallback(user_text)
        if fallback:
            name, args = fallback
            result = _run(name, args)
            try:
                response = _call([messages[0], {"role": "user", "content": user_text}, {"role": "tool", "content": result}])
                reply = _field(_field(response, "message", {}), "content", "") or result
            except Exception:
                reply = result
            history.append({"role": "assistant", "content": reply})
            return reply
        history.pop()
        raise error


def main() -> None:
    """Run Jarvis's terminal chat loop."""
    history: list[dict[str, Any]] = [{"role": "system", "content": "You are Jarvis, a helpful local assistant. Use tools for anything time-sensitive or factual about the current world. Answer only what the user asked and keep answers short. Never guess real-time facts. Mention source names or URLs for web answers. If a tool fails or returns no data, say so honestly. Ask for consent through tools before opening apps, opening websites, or taking screenshots."}]
    print(f"Jarvis is ready (model: {MODEL}). Type 'exit' to quit.")
    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\\nJarvis: Goodbye!")
            break
        if user_text.lower() == "exit":
            print("Jarvis: Goodbye!")
            break
        if not user_text:
            continue
        try:
            print("Jarvis:", answer(user_text, history))
        except Exception as error:
            print("Jarvis: I can't reach the model right now. Is the Ollama app running?")
            print("Details:", error)


if __name__ == "__main__":
    main()
