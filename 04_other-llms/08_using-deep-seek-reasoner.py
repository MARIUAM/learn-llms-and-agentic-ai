import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
from IPython.display import Markdown, display, update_display

load_dotenv(override=True)

deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')

if deepseek_api_key:
    print(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
else:
    print("DeepSeek API Key not set")


system_message = "You are a helpful assistan"
user_prompt = "How many words are there in your answer to this prompt"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

deepseek_via_openai_client = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
)

response = deepseek_via_openai_client.chat.completions.create(
    model="deepseek-reasoner",
    messages=prompts,
)
# Not tested yet, as deekseek-reasoner is not available yet
reasoning_content = response.choices[0].message.reasoning_content
content = response.choices[0].message.content

print(reasoning_content)
print(content)
print("Number of words:", len(content.split(" ")))

