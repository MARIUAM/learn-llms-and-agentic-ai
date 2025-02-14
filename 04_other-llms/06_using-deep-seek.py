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


system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell 10 light-hearted jokes for an audience of Data Scientists"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

deepseek_via_openai_client = OpenAI(
    api_key=deepseek_api_key, 
    base_url="https://api.deepseek.com"
)

# Not tested yet, as deekseek is not available yet
response = deepseek_via_openai_client.chat.completions.create(
    model="deepseek-chat",
    messages=prompts,
)


print(response.choices[0].message.content)

