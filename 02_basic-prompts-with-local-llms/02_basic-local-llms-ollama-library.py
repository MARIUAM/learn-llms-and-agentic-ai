from IPython.display import Markdown, display
import ollama

# OLLAMA_API = "http://localhost:11434/api/chat"
# HEADERS = {'Content-Type': 'application/json'}
MODEL = "llama3.2:1b"

messages = [
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]

# payload = {
#     "model": MODEL,
#     "messages": messages,
#     "stream": False
# }

response = ollama.chat(model=MODEL, messages=messages)
#print(response.json()['message']['content'])
print(response['message']['content'])