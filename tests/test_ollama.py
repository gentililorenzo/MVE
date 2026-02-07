import ollama

response = ollama.chat(
    model='gemma3:4b',
    messages=[
        {'role': 'user', 'content': 'Who are you? Give greetings'}
    ]
)

print(response['message']['content'])