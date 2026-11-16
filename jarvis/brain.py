"""Ollama loop, fallback logic, deduplication, and answer selection."""
from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from . import config
from .config import DEBUG, MAX_MESSAGES, MAX_TOOL_ROUNDS
from .llm import NormalizedResponse, ProviderManager
from .llm.router import choose_provider
from .tools import TOOLS, TOOL_REGISTRY, run_tool, tool_key

LOGGER = logging.getLogger("jarvis")
PROVIDER_MANAGER = ProviderManager()
INSTRUCTION = "Use the tool result above to answer. Do not say you lack access to real-time information."
SYSTEM_TOOLS = {"open_website", "open_application", "take_screenshot"}
WEB_TOOLS = {"web_search", "read_webpage", "get_weather", "get_air_quality", "get_news", "wikipedia_summary"}
DIRECT_ANSWER_TOOLS = {"get_current_datetime", "get_weather", "get_air_quality"}
DECLINED = "The user declined"
PENDING_QUESTION: dict[str, Any] | None = None
PENDING_PLACE_CONFIRMATION: dict[str, Any] | None = None
PRIVATE_INTERNET_REPLY = "Private mode is on, so I can't use the internet. Type 'private off' to turn it back on."


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from dict or object responses."""
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _normalized(response: Any) -> NormalizedResponse:
    """Accept a provider response or the old Ollama-shaped test response."""
    if isinstance(response, NormalizedResponse):
        return response
    message = _field(response, "message", {})
    calls = []
    for call in _field(message, "tool_calls", []) or []:
        function = _field(call, "function", {})
        arguments = _field(function, "arguments", {}) or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        from .llm import ToolCall
        calls.append(ToolCall(_field(function, "name", ""), arguments))
    return NormalizedResponse(text=_field(message, "content", "") or "", tool_calls=calls)


def fallback(text: str) -> tuple[str, dict[str, Any]] | None:
    """Choose a live tool when the model skips tool calling."""
    lower = text.lower().strip().rstrip("?.")
    if any(x in lower for x in ("what time", "current time", "what date", "today's date")):
        return "get_current_datetime", {}
    match = re.search(r"(?:weather|air quality|aqi)\s+(?:in|for)\s+([a-z .'-]+)", lower)
    if match:
        city = match.group(1).strip(" ?.!\"'\t\r\n")
        return ("get_air_quality", {"city": city}) if "air quality" in lower or "aqi" in lower else ("get_weather", {"city": city})
    if "news" in lower or "latest" in lower or "current events" in lower:
        topic = re.sub(r"\b(latest|news|about|the)\b", " ", lower, flags=re.I).strip(" ?.!\"'") or "world"
        return "get_news", {"topic": topic, "max_items": 5}
    if any(x in lower for x in ("who is the current", "current prime minister", "current president", "today", "score", "price")):
        return "web_search", {"query": text, "max_results": 5}
    return None


def _failed(result: str) -> bool:
    """Identify tool errors or empty results."""
    lower = result.lower()
    return lower.startswith(("web search failed:", "weather lookup failed:", "air-quality lookup failed:", "news lookup failed:", "wikipedia lookup failed:", "webpage read failed:", "tool ")) or "no search results were found" in lower or "no news items were found" in lower or "no location was found" in lower or "i couldn't find that place" in lower or lower.startswith("nothing was found")


def _refusal(text: str) -> bool:
    """Detect a refusal after a successful tool result."""
    lower = text.lower()
    return any(x in lower for x in ("i don't have access", "i'm not aware", "i am not aware", "i cannot", "static ai", "don't have real-time", "can't access real-time", "cannot access real-time"))


def _consent_request(text: str) -> bool:
    """Detect model-written consent requests that the program must replace."""
    lower = text.lower()
    return "consent" in lower or "permission" in lower or "ask you to confirm" in lower


def _clean_user_text(text: str) -> str:
    """Trim repeated spaces and harmless trailing punctuation."""
    return re.sub(r"\s+", " ", str(text)).strip().rstrip("?.").strip()


def clear_pending() -> None:
    """Clear the pending clarification state."""
    global PENDING_QUESTION
    PENDING_QUESTION = None


def clear_place_confirmation() -> None:
    """Clear a pending place confirmation."""
    global PENDING_PLACE_CONFIRMATION
    PENDING_PLACE_CONFIRMATION = None


def _set_place_confirmation(name: str, city: str, result: str) -> None:
    """Remember a place choice and its original geocoding result."""
    global PENDING_PLACE_CONFIRMATION
    data = json.loads(result)
    PENDING_PLACE_CONFIRMATION = {"tool_name": name, "city": city, "result": result, "suggested_city": data.get("suggested_city"), "remaining": 2}


def _set_pending(name: str, argument: str) -> None:
    """Remember one missing tool argument for up to two turns."""
    global PENDING_QUESTION
    PENDING_QUESTION = {"tool_name": name, "missing_argument": argument, "remaining": 2}


def _pending_followup(text: str) -> tuple[str, dict[str, Any]] | None:
    """Consume a short pending clarification answer, if one exists."""
    global PENDING_QUESTION
    if PENDING_QUESTION is None:
        return None
    words = text.split()
    casual_words = {str(word).lower() for word in config.ROUTING_LOCAL_KEYWORDS}
    if 0 < len(words) <= 3 and text.lower() not in casual_words | {"yes", "no", "y", "n"}:
        name = str(PENDING_QUESTION["tool_name"])
        argument = str(PENDING_QUESTION["missing_argument"])
        clear_pending()
        return name, {argument: text}
    PENDING_QUESTION["remaining"] -= 1
    if PENDING_QUESTION["remaining"] <= 0:
        clear_pending()
    return None


def _action_request(text: str) -> bool:
    """Return whether the user clearly asked for a system action."""
    return bool(re.search(r"\b(open|launch|start|go to|visit|play|screenshot|take a picture of the screen)\b", text.lower()))


def _internet_request(text: str) -> bool:
    """Detect a request that needs live internet data."""
    return fallback(text) is not None and fallback(text)[0] in WEB_TOOLS


def _target_allowed(name: str, arguments: dict[str, Any], user_text: str) -> bool:
    """Require a system target to appear in the user's request."""
    lower_text = user_text.lower()
    if name == "take_screenshot":
        return True
    if name == "open_application":
        target = str(arguments.get("name", "")).strip().lower()
        return bool(target) and target in lower_text
    if name == "open_website":
        parsed = urlparse(str(arguments.get("url", "")))
        domain = parsed.netloc.lower().removeprefix("www.")
        return bool(domain) and (domain in lower_text or domain.split(".")[0] in lower_text)
    return False


def _tools_for_turn(user_text: str, private_mode: bool = False) -> list[Any]:
    """Offer only tools that match a clear request."""
    if _action_request(user_text):
        tools = TOOLS
    elif not any(keyword in user_text.lower() for keyword in config.ROUTING_GEMINI_KEYWORDS):
        tools = []
    else:
        tools = [tool for tool in TOOLS if tool.__name__ not in SYSTEM_TOOLS]
    if private_mode:
        tools = [tool for tool in tools if tool.__name__ not in WEB_TOOLS]
    return tools


def _record(history: list[dict[str, Any]], role: str, content: str, private_mode: bool) -> None:
    """Add a history message and mark private turns for Gemini filtering."""
    message: dict[str, Any] = {"role": role, "content": content}
    if private_mode:
        message["private"] = True
    history.append(message)


def _normalize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Convert known numeric tool arguments to their practical type."""
    normalized = dict(arguments)
    for name in ("max_items", "max_results"):
        if name in normalized:
            try:
                normalized[name] = int(normalized[name])
            except (TypeError, ValueError):
                pass
    return normalized


def _simplify_query(query: str) -> str:
    """Remove punctuation and common filler from a factual query."""
    cleaned = re.sub(r"[^\w\s]", " ", query)
    cleaned = re.sub(r"\b(who is the|what is the|tell me about|current|latest|please|search for)\b", " ", cleaned, flags=re.I)
    return " ".join(cleaned.split())


def _search_with_fallback(query: str, arguments: dict[str, Any], cache: dict[tuple[str, str], str]) -> str:
    """Retry web search once, then try Wikipedia, without using model memory."""
    attempts = [query]
    simplified = _simplify_query(query)
    if simplified and simplified.lower() != query.strip().lower():
        attempts.append(simplified)
    for attempt_number, attempt in enumerate(attempts, 1):
        LOGGER.info("search_attempt=%s query=%s", attempt_number, attempt[:100])
        search_args = dict(arguments)
        search_args["query"] = attempt
        key = tool_key("web_search", search_args)
        result = cache.get(key) or run_tool("web_search", search_args)
        cache[key] = result
        if not _failed(result):
            return result
    LOGGER.info("search_attempt=wikipedia query=%s", query[:100])
    wiki_args = {"topic": _simplify_query(query) or query}
    wiki_key = tool_key("wikipedia_summary", wiki_args)
    wiki_result = cache.get(wiki_key) or run_tool("wikipedia_summary", wiki_args)
    cache[wiki_key] = wiki_result
    if not _failed(wiki_result):
        return wiki_result
    return "Nothing was found for that question."


def _tool_result(name: str, arguments: dict[str, Any], user_text: str, cache: dict[tuple[str, str], str]) -> str:
    """Run a requested tool with factual-search recovery when needed."""
    key = tool_key(name, arguments)
    if name == "web_search":
        return _search_with_fallback(str(arguments.get("query", user_text)), arguments, cache)
    result = cache.get(key)
    if result is None:
        result = run_tool(name, arguments)
        cache[key] = result
    return result


def _template(name: str, result: str, question: str = "") -> str:
    """Build a concise Python-owned answer from a tool result."""
    if name in {"get_news", "web_search"}:
        return _format_linked_results(name, result)
    if name == "get_current_datetime":
        parts = result.split(", ", 2)
        if "time" in question.lower() and len(parts) == 3:
            return f"It is {parts[2]} on {parts[0]}, {parts[1]}."
        if "date" in question.lower() and len(parts) >= 2:
            return f"Today is {parts[0]}, {parts[1]}."
    if name in {"get_weather", "get_air_quality"}:
        try:
            data = json.loads(result)
            location = data.get("location", "Unknown location")
            if data.get("needs_confirmation"):
                return f"Did you mean {data.get('suggested_location', data.get('suggested_city'))}? (y/n)"
            if name == "get_weather":
                reply = (
                    f"{location}: {data.get('conditions', 'unknown')}, "
                    f"{data.get('temperature_c', 'unknown')} C "
                    f"(feels like {data.get('feels_like_c', 'unknown')} C), "
                    f"humidity {data.get('humidity_percent', 'unknown')}%, "
                    f"wind {data.get('wind_kmh', 'unknown')} km/h. Source: Open-Meteo."
                )
                return _assumption_prefix(data, reply)
            reply = (
                f"{location}: PM2.5 {data.get('pm2_5_ug_m3', 'unknown')} ug/m3, "
                f"PM10 {data.get('pm10_ug_m3', 'unknown')} ug/m3, "
                f"US AQI {data.get('us_aqi', 'unknown')}, European AQI "
                f"{data.get('european_aqi', 'unknown')}. "
                "AQI scales are Open-Meteo's US and European scales, not India's national AQI. "
                "Source: Open-Meteo."
            )
            return _assumption_prefix(data, reply)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    if name == "open_website":
        domain = urlparse(result.removeprefix("Opened ").split(" in the default browser", 1)[0]).netloc
        return f"Done. I opened {domain or 'the website'}."
    if name == "open_application":
        return f"Done. {result}"
    if name == "take_screenshot":
        return result
    labels = {"get_news": "Here are the latest news results", "web_search": "Here are the current web-search results", "wikipedia_summary": "Here is the Wikipedia summary", "read_webpage": "Here is the webpage text"}
    return f"{labels.get(name, 'Tool result')}: {result}"


def _needs_place_confirmation(result: str) -> bool:
    """Return whether a weather result asks for a place choice."""
    try:
        return bool(json.loads(result).get("needs_confirmation"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _format_linked_results(name: str, result: str) -> str:
    """Format headlines or search results without inline URLs."""
    try:
        rows = json.loads(result)
    except (TypeError, ValueError, json.JSONDecodeError):
        return result
    if not isinstance(rows, list):
        return result
    lines: list[str] = []
    links: list[str] = []
    for index, row in enumerate(rows[:5], 1):
        if not isinstance(row, dict):
            continue
        title = str(row.get("headline", row.get("title", "")))
        title = re.sub(r"https?://\S+", "", title).replace("\ufffd", "").strip(" .-\t\r\n")
        source = str(row.get("source", ""))
        if not source or source.lower() in {"google news", "rss"}:
            parts = [part.strip() for part in title.rsplit(" - ", 1)]
            source = parts[-1] if len(parts) == 2 else (source or "Web search")
        source = re.sub(r"https?://\S+", "", source).replace("\ufffd", "").strip(" .-\t\r\n") or "Unknown source"
        title = re.sub(rf"\s*[-–—�]\s*{re.escape(source)}\s*[�]?\s*$", "", title).strip(" .-\t\r\n")
        if source == "Web search":
            source = urlparse(str(row.get("url", ""))).netloc or source
        if title:
            lines.append(f"{index}. {title} — Source: {source}.")
        if title:
            lines[-1] = f"{index}. {title} - Source: {source}."
        link = str(row.get("link", row.get("url", ""))).strip()
        if link:
            links.append(link)
    if not lines:
        return result
    heading = "Here are the latest news headlines:" if name == "get_news" else "Here are the current web-search results:"
    if not links:
        return heading + "\n" + "\n".join(lines)
    return heading + "\n" + "\n".join(lines) + "\n\nLinks:\n" + "\n".join(f"{index}. {link}" for index, link in enumerate(links, 1))


def _assumption_prefix(data: dict[str, Any], reply: str) -> str:
    """Add a clear note when a city typo was corrected."""
    assumption = data.get("assumed_city")
    return f"I assumed you meant {assumption}. {reply}" if assumption else reply


def _report(source: str, provider: str | None = None) -> None:
    """Log and show the answer source when debugging."""
    provider_name = provider or PROVIDER_MANAGER.active_name
    model = "none" if provider_name == "none" else (config.GEMINI_MODEL if provider_name == "gemini" else config.MODEL)
    LOGGER.info("answer_source=%s provider=%s model=%s", source, provider_name, model)
    if DEBUG:
        print(f"[DEBUG] answer_source={source} provider={provider_name} model={model}")


def _call(messages: list[dict[str, Any]], tools: list[Any] | None = None) -> NormalizedResponse:
    """Call the active provider with the tools allowed for this turn."""
    return PROVIDER_MANAGER.chat(messages, TOOLS if tools is None else tools)


def _place_confirmation_answer(text: str, history: list[dict[str, Any]], private_mode: bool) -> str | None:
    """Handle y/n or a corrected city without a model call."""
    global PENDING_PLACE_CONFIRMATION
    if PENDING_PLACE_CONFIRMATION is None:
        return None
    state = PENDING_PLACE_CONFIRMATION
    lower = text.casefold()
    if lower in {"y", "yes"}:
        clear_place_confirmation()
        result = _tool_result(state["tool_name"], {"city": state["suggested_city"]}, text, {})
    elif lower in {"n", "no"}:
        clear_place_confirmation()
        result = state["result"]
    elif lower.startswith("no,"):
        clear_place_confirmation()
        result = _tool_result(state["tool_name"], {"city": text.split(",", 1)[1].strip()}, text, {})
    elif len(text.split()) <= 3:
        clear_place_confirmation()
        result = _tool_result(state["tool_name"], {"city": text}, text, {})
    else:
        state["remaining"] -= 1
        if state["remaining"] <= 0:
            clear_place_confirmation()
        return None
    if lower in {"n", "no"}:
        try:
            data = json.loads(result)
            data.pop("needs_confirmation", None)
            data.pop("suggested_city", None)
            data.pop("suggested_location", None)
            result = json.dumps(data)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    reply = _template(state["tool_name"], result, text)
    _record(history, "assistant", reply, private_mode)
    _report("direct answer", "none")
    return reply


def answer(user_text: str, history: list[dict[str, Any]], private_mode: bool = False) -> str:
    """Answer one turn while enforcing tool and confirmation safety."""
    user_text = _clean_user_text(user_text)
    _record(history, "user", user_text, private_mode)
    messages = [history[0]] + history[1:][-MAX_MESSAGES:]
    cache: dict[tuple[str, str], str] = {}
    place_reply = _place_confirmation_answer(user_text, history, private_mode)
    if place_reply is not None:
        return place_reply
    pending_call = _pending_followup(user_text)
    if user_text.lower() in {"yes", "no", "y", "n"}:
        reply = "I'm not sure what you're confirming. What would you like me to do?"
        _record(history, "assistant", reply, private_mode)
        _report("direct answer", "none")
        return reply
    if pending_call:
        name, arguments = pending_call
        result = _tool_result(name, arguments, user_text, cache)
        if _failed(result):
            _set_pending(name, "city")
            reply = "I couldn't find that place. Which city did you mean?"
        elif name in {"get_weather", "get_air_quality"} and _needs_place_confirmation(result):
            _set_place_confirmation(name, str(arguments.get("city", "")), result)
            reply = _template(name, result, user_text)
        else:
            reply = _template(name, result, user_text)
        _record(history, "assistant", reply, private_mode)
        _report("direct answer", "none")
        return reply
    requested = fallback(user_text)
    availability = PROVIDER_MANAGER.availability()
    chosen_provider = "ollama" if private_mode else choose_provider(user_text, config.ROUTING_MODE, availability)
    reason = "private mode" if private_mode else f"routing mode {config.ROUTING_MODE}"
    LOGGER.info("provider=%s mode=%s reason=%s", chosen_provider, config.ROUTING_MODE, reason)
    PROVIDER_MANAGER.set_provider(chosen_provider)
    allowed_tools = _tools_for_turn(user_text, private_mode)
    system_action_used = False
    successful: list[tuple[str, str]] = []
    used_tool = False
    notice_shown = False
    try:
        if private_mode and requested and requested[0] in WEB_TOOLS:
            reply = PRIVATE_INTERNET_REPLY
            _record(history, "assistant", reply, private_mode)
            _report("direct answer")
            return reply
        if private_mode and _internet_request(user_text):
            reply = PRIVATE_INTERNET_REPLY
            _record(history, "assistant", reply, private_mode)
            _report("direct answer", "none")
            return reply
        if requested and requested[0] in DIRECT_ANSWER_TOOLS:
            name, arguments = requested
            result = _tool_result(name, arguments, user_text, cache)
            if _failed(result) and "which city" in result.lower():
                _set_pending(name, "city")
                reply = "I couldn't find that place. Which city did you mean?"
            elif name in {"get_weather", "get_air_quality"} and _needs_place_confirmation(result):
                _set_place_confirmation(name, str(arguments.get("city", "")), result)
                reply = _template(name, result, user_text)
            else:
                reply = _template(name, result, user_text) if not _failed(result) else result
            _record(history, "assistant", reply, private_mode)
            _report("direct answer", "none")
            return reply
        for _ in range(MAX_TOOL_ROUNDS):
            response = _normalized(_call(messages, allowed_tools))
            if response.notice and not notice_shown:
                print(response.notice)
                notice_shown = True
            if response.notice and chosen_provider == "gemini" and _action_request(user_text):
                reply = "I can't do that right now without the cloud model."
                _record(history, "assistant", reply, private_mode)
                _report("fallback")
                return reply
            calls = response.tool_calls
            if not calls:
                reply = response.text or "I could not produce an answer."
                if _consent_request(reply):
                    reply = "I handle confirmations in the program. What would you like me to do?"
                if successful and _refusal(reply):
                    reply = _template(*successful[-1], user_text)
                elif successful and successful[-1][0] in {"get_news", "web_search"}:
                    reply = _template(*successful[-1], user_text)
                if not used_tool and requested:
                    name, arguments = requested
                    result = _tool_result(name, arguments, user_text, cache)
                    if _failed(result):
                        reply = result
                    elif TOOL_REGISTRY[name].direct_answer:
                        reply = _template(name, result, user_text)
                    else:
                        try:
                            follow = _normalized(_call(
                                [messages[0], {"role": "user", "content": user_text},
                                 {"role": "tool", "content": result},
                                 {"role": "user", "content": INSTRUCTION}],
                                allowed_tools,
                            ))
                            if follow.notice and not notice_shown:
                                print(follow.notice)
                                notice_shown = True
                            if _consent_request(follow.text):
                                reply = "I handle confirmations in the program. What would you like me to do?"
                            else:
                                reply = _template(name, result, user_text) if name in {"get_news", "web_search"} else follow.text or _template(name, result, user_text)
                        except Exception:
                            reply = _template(name, result, user_text)
                        if _refusal(reply) and not _failed(result):
                            reply = _template(name, result, user_text)
                    _report("fallback" if name not in {"get_current_datetime", "get_weather", "get_air_quality"} else "direct answer")
                    _record(history, "assistant", reply, private_mode)
                    return reply
                _report("model with tool result" if used_tool else "model only")
                _record(history, "assistant", reply, private_mode)
                return reply

            action_names = [call.name for call in calls]
            if sum(name in SYSTEM_TOOLS for name in action_names) > 1:
                return _cancel_turn(history, "OK, cancelled.", private_mode)

            used_tool = True
            messages.append({"role": "assistant", "content": response.text, "tool_calls": [{"name": call.name, "arguments": call.arguments} for call in calls], "private": private_mode})
            for call in calls:
                name = call.name
                arguments = _normalize_arguments(call.arguments)
                if private_mode and name in WEB_TOOLS:
                    reply = PRIVATE_INTERNET_REPLY
                    _record(history, "assistant", reply, private_mode)
                    _report("direct answer", "none")
                    return reply
                if name in {"get_weather", "get_air_quality"} and not arguments.get("city"):
                    _set_pending(name, "city")
                    reply = "Which city did you mean?"
                    _record(history, "assistant", reply, private_mode)
                    _report("direct answer", "none")
                    return reply
                if name in SYSTEM_TOOLS:
                    if not _action_request(user_text) or not _target_allowed(name, arguments, user_text):
                        return _cancel_turn(history, "OK, cancelled.", private_mode)
                    if system_action_used:
                        return _cancel_turn(history, "OK, cancelled.", private_mode)
                    system_action_used = True
                result = _tool_result(name, arguments, user_text, cache)
                if result.startswith(DECLINED):
                    return _cancel_turn(history, "OK, cancelled.", private_mode)
                if name in {"get_weather", "get_air_quality"} and _failed(result) and "which city" in result.lower():
                    _set_pending(name, "city")
                    reply = "I couldn't find that place. Which city did you mean?"
                    _record(history, "assistant", reply, private_mode)
                    _report("direct answer", "none")
                    return reply
                if name in {"get_weather", "get_air_quality"} and not _failed(result):
                    try:
                        if json.loads(result).get("needs_confirmation"):
                            _set_place_confirmation(name, str(arguments.get("city", "")), result)
                            reply = _template(name, result, user_text)
                            _record(history, "assistant", reply, private_mode)
                            _report("direct answer", "none")
                            return reply
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
                if name == "web_search" and _failed(result):
                    history.append({"role": "assistant", "content": result})
                    _report("fallback")
                    return result
                if not _failed(result) and (name, result) not in successful:
                    successful.append((name, result))
                messages.append({"role": "tool", "content": result, "name": name, "private": private_mode})
                definition = TOOL_REGISTRY.get(name)
                if definition and definition.direct_answer and not _failed(result):
                    reply = _template(name, result, user_text)
                    _record(history, "assistant", reply, private_mode)
                    _report("direct answer")
                    return reply
            messages.append({"role": "user", "content": INSTRUCTION})
            messages = [messages[0]] + messages[1:][-MAX_MESSAGES:]

        if requested and not any(name == requested[0] for name, _ in successful):
            name, arguments = requested
            result = _tool_result(name, arguments, user_text, cache)
            if not _failed(result):
                reply = _template(name, result, user_text)
                _record(history, "assistant", reply, private_mode)
                _report("fallback")
                return reply
        if successful:
            name, result = successful[-1]
            reply = _template(name, result, user_text)
            history.append({"role": "assistant", "content": reply})
            _report("model with tool result")
            return reply
        reply = "Nothing was found for that question." if requested and requested[0] == "web_search" else "I stopped after the maximum number of tool rounds. Please try a simpler question."
        _record(history, "assistant", reply, private_mode)
        return reply
    except Exception as error:
        if requested:
            name, arguments = requested
            result = _tool_result(name, arguments, user_text, cache)
            reply = _template(name, result, user_text) if not _failed(result) else result
            _record(history, "assistant", reply, private_mode)
            _report("fallback")
            return reply
        history.pop()
        raise error


def _cancel_turn(history: list[dict[str, Any]], reply: str, private_mode: bool = False) -> str:
    """End a cancelled turn with a Python-owned response."""
    _record(history, "assistant", reply, private_mode)
    _report("direct answer")
    return reply
