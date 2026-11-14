"""Offline brain tests."""
from unittest.mock import patch
from jarvis import brain
from jarvis.brain import answer

def history():
    """Return a minimal test history."""
    return [{"role":"system","content":"test"}]

def test_direct_time_answer_does_not_depend_on_model():
    """A successful time tool produces the answer directly."""
    with patch("jarvis.brain._call") as model, patch("jarvis.brain.run_tool", return_value="Wednesday, 11 November 2026, 06:25:00 PM (IST)"):
        assert "It is 06:25:00 PM" in answer("what time is it?", history())
    model.assert_not_called()

def test_duplicate_tool_call_runs_once():
    """Identical calls in one turn reuse the first result."""
    call = {"function":{"name":"web_search","arguments":{"query":"x","max_results":5}}}
    with patch("jarvis.brain._call", side_effect=[{"message":{"tool_calls":[call],"content":""}}, {"message":{"tool_calls":[call],"content":""}}, {"message":{"tool_calls":[],"content":"Here are results."}}]), patch("jarvis.brain.run_tool", return_value='[{"url":"https://example.com"}]') as run:
        answer("current price", history())
        assert run.call_count == 1

def test_refusal_after_tool_result_uses_result():
    """A refusal is replaced by a tool-result answer."""
    call = {"function":{"name":"web_search","arguments":{"query":"x","max_results":5}}}
    with patch("jarvis.brain._call", side_effect=[{"message":{"tool_calls":[call],"content":""}}, {"message":{"tool_calls":[],"content":"I'm not aware of real-time information."}}]), patch("jarvis.brain.run_tool", return_value='[{"url":"https://example.com"}]'):
        assert "https://example.com" in answer("current price", history())

def test_declined_confirmation_ends_turn_without_more_calls():
    """A declined action stops the model loop immediately."""
    call = {"function": {"name": "take_screenshot", "arguments": {}}}
    with patch("jarvis.brain._call", return_value={"message": {"tool_calls": [call], "content": ""}}) as model, patch("jarvis.brain.run_tool", return_value="The user declined to take a screenshot.") as run:
        assert answer("take a screenshot", history()) == "OK, cancelled."
    assert model.call_count == 1 and run.call_count == 1

def test_two_system_actions_are_refused():
    """Only one system action can run in a turn."""
    calls = [
        {"function": {"name": "open_website", "arguments": {"url": "https://example.com"}}},
        {"function": {"name": "take_screenshot", "arguments": {}}},
    ]
    with patch("jarvis.brain._call", return_value={"message": {"tool_calls": calls, "content": ""}}), patch("jarvis.brain.run_tool", return_value="Opened https://example.com in the default browser.") as run:
        assert answer("visit example.com and take a screenshot", history()) == "OK, cancelled."
    assert run.call_count == 0

def test_screenshot_request_cannot_trigger_website():
    """A screenshot request rejects an unrequested website target."""
    call = {"function": {"name": "open_website", "arguments": {"url": "https://www.google.com"}}}
    with patch("jarvis.brain._call", return_value={"message": {"tool_calls": [call], "content": ""}}), patch("jarvis.brain.run_tool") as run:
        assert answer("take a screenshot", history()) == "OK, cancelled."
    run.assert_not_called()

def test_non_action_turn_does_not_offer_system_tools():
    """Informational turns omit system tools from the model request."""
    seen = []
    def fake_call(messages, tools):
        """Record tools offered to the model."""
        seen.extend(tool.__name__ for tool in tools)
        return {"message": {"tool_calls": [], "content": "Hello."}}
    with patch("jarvis.brain._call", side_effect=fake_call):
        assert answer("tell me a fact", history()) == "Hello."
    assert "open_website" not in seen and "open_application" not in seen and "take_screenshot" not in seen

def test_untyped_url_target_is_refused():
    """A website not named by the user cannot be opened."""
    call = {"function": {"name": "open_website", "arguments": {"url": "https://example.com"}}}
    with patch("jarvis.brain._call", return_value={"message": {"tool_calls": [call], "content": ""}}), patch("jarvis.brain.run_tool") as run:
        assert answer("open a website", history()) == "OK, cancelled."
    run.assert_not_called()

def test_weather_answer_is_not_raw_json():
    """Weather output is a short Python-built sentence."""
    with patch("jarvis.brain._call", side_effect=RuntimeError("offline")) as model, patch("jarvis.brain.run_tool", return_value='{"location":"Jaipur, India","conditions":"clear sky","temperature_c":21.8,"feels_like_c":22.0,"humidity_percent":54,"wind_kmh":2.6,"source":"Open-Meteo"}'):
        result = answer("what is the weather in Jaipur", history())
    model.assert_not_called()
    assert "{" not in result and "Jaipur" in result and "Source: Open-Meteo" in result

def test_search_retries_then_uses_wikipedia():
    """Failed searches retry simply, then use Wikipedia when available."""
    def fake_tool(name, arguments):
        """Return failures for search and a result for Wikipedia."""
        if name == "wikipedia_summary":
            return '{"title":"Narendra Modi","summary":"Prime Minister","url":"https://en.wikipedia.org/wiki/Narendra_Modi"}'
        return "Web search failed: No results found"
    with patch("jarvis.brain._call", side_effect=RuntimeError("offline")), patch("jarvis.brain.run_tool", side_effect=fake_tool) as run:
        result = answer("who is the current prime minister of India?", history())
    assert "Narendra Modi" in result
    assert [call.args[0] for call in run.call_args_list] == ["web_search", "web_search", "wikipedia_summary"]


def test_news_links_are_only_at_end_and_preserved():
    """News prose has source names while complete URLs stay in Links."""
    long_link = "https://news.google.com/rss/articles/" + "a" * 240
    result = '[{"headline":"India update","source":"Example Outlet","link":"' + long_link + '"}]'
    call = {"function": {"name": "get_news", "arguments": {"topic": "India", "max_items": 5}}}
    with patch("jarvis.brain._call", side_effect=[
        {"message": {"tool_calls": [call], "content": ""}},
        {"message": {"tool_calls": [], "content": "A model rewrite with " + long_link}},
    ]), patch("jarvis.brain.run_tool", return_value=result):
        reply = answer("latest news about India", history())
    sentence_part, links_part = reply.split("\n\nLinks:\n", 1)
    assert long_link not in sentence_part
    assert "Source: Example Outlet" in sentence_part
    assert links_part == "1. " + long_link

def test_private_mode_never_calls_gemini_or_web_tools():
    """Private mode uses local model tools only and removes web tools."""
    seen = []
    def fake_call(messages, tools):
        """Record private-mode tools."""
        seen.extend(tool.__name__ for tool in tools)
        return {"message": {"tool_calls": [], "content": "local"}}
    with patch("jarvis.brain._call", side_effect=fake_call) as model:
        assert answer("hello", history(), True) == "local"
    model.assert_called_once()
    assert brain.PROVIDER_MANAGER.active_name == "ollama"
    assert "web_search" not in seen and "get_weather" not in seen and "get_news" not in seen

def test_private_history_is_excluded_from_gemini_messages(monkeypatch):
    """Private history entries are not sent when Gemini is selected."""
    captured = []
    class FakeProvider:
        """Capture messages passed to the provider."""
        model = "fake"
        def chat(self, messages, tools):
            """Capture provider messages without making a network call."""
            captured.extend(messages)
            return brain.NormalizedResponse(text="answer")
    monkeypatch.setattr(brain.config, "ROUTING_MODE", "gemini_first")
    with patch.object(brain.PROVIDER_MANAGER, "availability", return_value={"gemini": True, "ollama": True}), patch.object(brain.PROVIDER_MANAGER, "_provider", return_value=FakeProvider()):
        answer("hello", [{"role": "system", "content": "test"}, {"role": "user", "content": "secret", "private": True}], False)
    assert all(message.get("content") != "secret" for message in captured)
