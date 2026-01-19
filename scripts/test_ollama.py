import ollama # type: ignore

# Test base
response = ollama.chat(
    model='gemma3:4b',
    messages=[
        {'role': 'user', 'content': 'Rispondi in italiano: chi sei?'}
    ]
)

print(response['message']['content'])