My Jarvis voice assistant project

## Run Jarvis

1. Start Ollama and make sure llama3.2:3b is installed.
2. Activate the virtual environment: D:\\JARVIS\\venv\\Scripts\\Activate.ps1
3. Install dependencies: python -m pip install -r requirements.txt
4. Start the assistant: python jarvis.py

Jarvis uses only local Ollama plus free, keyless services: DuckDuckGo search,
Open-Meteo, public RSS, and Wikipedia. Type exit to quit. Opening a website
or application and taking a screenshot always asks for terminal confirmation.
Set DEBUG = True in jarvis.py to print tool calls and arguments.
