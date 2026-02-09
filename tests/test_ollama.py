import ollama

response = ollama.chat(
    model='Qwen2.5:7b',
    messages=[
        {'role': 'user', 'content': 'Who are you? Give greetings'}
    ]
)

print(response['message']['content'])