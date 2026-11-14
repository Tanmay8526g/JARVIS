# Jarvis project rules

## Goal
A real-time personal assistant for Windows 11 that can talk (voice in/out), get live
information from the internet, and control my computer safely.

## Environment
- Windows 11, PowerShell, VS Code. Project folder: D:\JARVIS. Python venv: D:\JARVIS\venv.
- Python 3.14 (very new). If a package has no working wheel for it, STOP and tell me.
  Do not silently drop a feature. Suggest a free alternative, or tell me to create a
  Python 3.12 venv.
- Brains: Google Gemini (free tier, model set in config) is the main provider; local
  Ollama (llama3.2:3b) is the offline fallback. A rule-based router chooses between them.
  Private mode uses Ollama only.

## Cost and keys
- EVERYTHING MUST BE FREE. Never enable billing, never use a paid plan, never use a
  service that needs a credit card. Free-tier API keys are allowed only if I approve
  them first (GEMINI_API_KEY is approved).
- Prefer free options with no key. If a feature needs a new key, account, database
  server, hardware, large download (over 500 MB), personal information, software
  outside the venv, or a permission change, STOP and ask me first.
- Ask using a "NEEDED FROM YOU" list: what, why, free or paid, how to get it, where it
  will be stored, what still works without it.

## Secrets
- NEVER open, read, print, copy or search the contents of .env. To check it, use scripts
  that print variable NAMES and True/False only.
- Never ask me to paste a secret into chat. Never write a secret into source, tests,
  logs, README, error messages or terminal commands. I edit .env myself.
- Keep .env in .gitignore. Keep .env.example up to date (empty values and comments).
- Never send .env contents, file contents or screenshots to any model.

## Safety rules for Jarvis (must never be broken)
- Never use eval, exec, os.system, or subprocess with shell=True. Never pass text from
  the user or a model to any command, shell or launcher except through the fixed
  allowlist.
- System actions (open apps, open websites, screenshots, file writes, volume, keyboard or
  mouse control, close apps, shutdown) are ONLY allowed through a fixed allowlist of named
  functions. No generic "run this command" tool.
- Any action that changes the system asks for confirmation (y/n) first. Destructive
  actions (delete, shutdown, restart, send message) always need confirmation, and there
  is no setting that turns this off. A declined confirmation ends the turn.
- The running Jarvis may only write inside D:\JARVIS\workspace\ and D:\JARVIS\logs\,
  unless I name another path in my request and confirm it.
- Real-time facts (news, weather, prices, people in office, sports, dates) must come from
  a tool, never from the model's memory. If a tool fails, say so with the reason. Never
  invent. Include source names for web-based answers.
- Never log message text, spoken text, or secrets. Log only metadata (tool name, provider,
  duration, success or failure, error type).

## Code rules
- Small modules, one job each (llm, tools, voice, memory, config).
- Type hints and a docstring on every function, comments in simple language (I am
  learning).
- Every external call has a timeout and try/except. One failing tool must never crash
  Jarvis.
- Constants and settings live in config.py (overridable from config.json), not scattered.
- Keep the existing safety logic (confirmation gate, intent guard, one system action per
  turn, dedupe, direct-answer tools, refusal handling) working with every change.
- Do not change files I did not ask about. Do not revert other changes. List every file
  you changed at the end. Ignore my scratch file test.py.

## How to finish every task
1. Run `python -m py_compile` on every changed file.
2. Run pytest and show me the full output.
3. Test the feature yourself and report honestly what worked, what failed, and what you
   could not test. Show RAW terminal output, not summaries. Do not say "verified" for
   anything you did not run. You cannot listen to audio or judge answer quality, so mark
   those as "needs a human to check".
4. Search the project (excluding venv and .env) for eval(, exec(, os.system, shell=True,
   subprocess, Popen and startfile, and explain every match.
5. Do not commit or push. I do that myself.
6. End with: files changed, what worked, what failed, what you could not test, and a
   "NEEDED FROM YOU" section ("Nothing needed" if nothing is missing).