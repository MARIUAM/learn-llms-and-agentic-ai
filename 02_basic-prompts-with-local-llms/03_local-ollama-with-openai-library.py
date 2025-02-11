from IPython.display import Markdown, display
from openai import OpenAI

#OLLAMA_API = "http://localhost:11434/api/chat"
OLLAMA_API = "http://localhost:11434/v1"

# use any modal
#MODEL = "llama3.2:1b"
MODEL = "qwen2.5:0.5b"

ollama_via_openai = OpenAI(base_url=OLLAMA_API, api_key="ollama")

messages = [
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]

response = ollama_via_openai.chat.completions.create(model=MODEL, messages=messages)
print(response.choices[0].message.content)   
