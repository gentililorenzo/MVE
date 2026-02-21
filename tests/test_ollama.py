import ollama

response = ollama.chat(
    model='Qwen2.5:7b',
    messages=[
        {'role': 'user', 'content': 'Who are you? Give greetings. Do you know what is the VSME standard?'}
    ]
)

print(response['message']['content'])