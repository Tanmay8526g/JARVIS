# Jarvis

Stage 1 is a terminal personal assistant using local Ollama and free live-data services.

## Run Jarvis

1. Start Ollama and make sure llama3.2:3b is installed.
2. Activate the virtual environment: D:\\JARVIS\\venv\\Scripts\\Activate.ps1
3. Install dependencies: python -m pip install -r requirements.txt
4. Start the assistant: python jarvis.py

Jarvis uses Ollama or Google Gemini plus free services: DuckDuckGo search,
Open-Meteo, public RSS, and Wikipedia. Type exit to quit. Opening a website
or application and taking a screenshot always asks for terminal confirmation.
Settings are in `jarvis/config.py` and can be overridden by an optional `config.json`.
Debug output is controlled by `DEBUG` there. The package is split into `jarvis/brain.py`,
`jarvis/tools/`, and `jarvis/main.py`; `jarvis.py` is the compatibility launcher.

## Test

```powershell
python -m pip install -r requirements.txt
pytest
```

Logs are written to `logs/jarvis.log`. System actions use an allowlist and always ask for confirmation.

## Providers

Ollama is the default local provider. To use Gemini, create a free Google AI Studio
API key, put it in `.env` as `GEMINI_API_KEY=...`, and set `LLM_PROVIDER` to `gemini`
in `config.json`, or type `provider gemini` for the current session. Use `provider`
to show the active provider and `provider ollama` to switch back.

The Gemini model is configurable as `GEMINI_MODEL`; the default is
`gemini-3.5-flash-lite`, selected because Google's current pricing lists it with
free-tier input and output pricing and the current models page recommends the
3.5 Flash-Lite family for new projects.
Free-tier requests may be used by Google to improve its products. Jarvis sends only
conversation text and tool results to the selected provider; it never sends `.env`,
local file contents, or screenshots.

## Routing and private mode

`ROUTING_MODE` can be `gemini_first`, `smart`, or `local_only` in `config.json`.
Smart mode sends casual short conversation to Ollama and tool, search, real-time, or
reasoning requests to Gemini. Local-only mode never selects Gemini.

Terminal commands are `mode`, `mode smart`, `mode gemini_first`, `mode local_only`,
`private on`, and `private off`. Private mode forces Ollama, offers no web tools, and
marks private history so it is excluded from any later Gemini request.
