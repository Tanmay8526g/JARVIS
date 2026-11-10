from ollama import chat

history = [
    {'role': 'system', 'content': 'You are Jarvis, a helpful assistant. Keep answers short.'}
]

while True:
    user_text = input("You: ")
    if user_text.lower() == "exit":
        break

    history.append({'role': 'user', 'content': user_text})

    response = chat(model='llama3.2:3b', messages=history)
    reply = response['message']['content']

    history.append({'role': 'assistant', 'content': reply})
    print("Jarvis:", reply)