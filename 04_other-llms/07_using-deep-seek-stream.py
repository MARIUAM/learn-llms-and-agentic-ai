import os
from dotenv import load_dotenv
from openai import OpenAI
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

stream = deepseek_via_openai_client.chat.completions.create(
    model="deepseek-chat",
    messages=prompts,
    stream=True
)

# Not tested yet, as deekseek is not available yet
reply = ""
# display handle works with Jupytor notebook
#display_handle = display(Markdown(reply), display_id=True)
for chunk in stream:
    reply += chunk.choices[0].delta.content or ''
    #reply = reply.replace("```","").replace("markdown","")
    #update_display(Markdown(reply), display_id=display_handle.display_id)
    print("\r", reply, end="")

print("Number of words:", len(reply.split(" ")))

