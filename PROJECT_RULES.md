# Jarvis project rules

## Goal
A real-time personal assistant for Windows 11 that can talk (voice in/out), get live
information from the internet, and control my computer safely.

## Environment
- Windows 11, PowerShell, VS Code. Project folder: D:\JARVIS. Python venv: D:\JARVIS\venv.
- Python 3.14 (very new). If a package has no working wheel for it, STOP and tell me.
  Do not silently drop a feature. Suggest a free alternative, or tell me to create a
  Python 3.12 venv.
- Local LLM through Ollama (model llama3.2:3b). Python package: ollama.
- EVERYTHING MUST BE FREE. No paid APIs, no API keys, no accounts. If a feature truly
  needs a key, tell me before using it.

## Safety rules (must never be broken)
- Never use eval, exec, os.system, or subprocess with shell=True on text from the user
  or the model.
- System actions (open apps, close apps, volume, screenshots, file writes, keyboard or
  mouse control, shutdown) are ONLY allowed through a fixed allowlist of named
  functions. No generic "run this command" tool.
- Any action that changes the system asks for confirmation (y/n) first. Destructive
  actions (delete, shutdown, restart, send message) always need confirmation, and
  there must be no setting that turns this off.
- Never touch files outside D:\JARVIS\workspace\ unless I name the path in my request
  and confirm it.
- No secrets in code. Settings go in config.json or .env. .env is in .gitignore.
- Real-time facts (news, weather, prices, people in office, sports, dates) must come
  from a tool, never from the model's memory. If a tool fails, say so. Never invent.

## Code rules
- Small modules, one job each (brain, tools, voice, memory, config).
- Type hints and a docstring on every function, comments in simple language (I am learning).
- Every external call has a timeout and try/except. One failing tool must never crash Jarvis.
- Constants and settings live in one config place, not scattered in code.
- Do not change files I did not ask about. List every file you changed at the end.

## How to finish every task
1. Run `python -m py_compile` on every changed file.
2. Run any tests you wrote and show me the output.
3. Test the feature yourself and report honestly what worked, what failed, what you
   could not test. Do not say "verified" for anything you did not run.
4. Do not commit or push. I do that myself.

############################################
## User-provided stage requirements
############################################

This stage also requires the requested modular structure, offline tests, safe system
actions, logging, and honest reporting described in the Stage 1 task specification.

## Secrets and missing information
- Create `.env.example` with empty variable values and keep `.env` in `.gitignore`.
- Never ask me to paste a secret into chat, and never write secrets into source,
  tests, logs, README files, or terminal commands.
- Ask before using API keys, paid services, accounts, databases, servers, personal
  information, hardware, large downloads, software outside the project environment,
  or permission changes.
- End every task with a `NEEDED FROM YOU` section. Say `Nothing needed` when no input
  is missing.

## Stage 1 boundaries
- 
- Run compilation on changed Python files, run tests, run the requested feature checks,
  and report what worked, failed, or could not be tested honestly.
