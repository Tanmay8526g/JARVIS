from ollama import chat

MODEL = 'llama3.2:3b'
MAX_MESSAGES = 10  # how many recent messages to send besides the system message

history = [
    {'role': 'system', 'content': 'You are Jarvis, a helpful assistant. Keep answers short.'}
]

while True:
    user_text = input("You: ")
    if user_text.lower() == "exit":
        break
    
    history.append({'role': 'user', 'content': user_text})

    # Fix 1: send the system message + only the most recent messages
    messages_to_send = [history[0]] + history[1:][-MAX_MESSAGES:]

    # Fix 2: don't crash if Ollama fails
    try:
        response = chat(model=MODEL, messages=messages_to_send)
        reply = response['message']['content']
    except Exception as error:
        print("Jarvis: I can't reach the model right now. Is the Ollama app running?")
        print("Details:", error)
        history.pop()  # remove the user message that got no reply
        continue

    history.append({'role': 'assistant', 'content': reply})
    print("Jarvis:", reply)