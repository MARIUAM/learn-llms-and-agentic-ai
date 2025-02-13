import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
from IPython.display import Markdown, display, update_display

# Load the .env file
load_dotenv(override=True)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

if anthropic_api_key:
    print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
else:
    print("Anthropic API Key not set")


claude = anthropic.Anthropic()

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

# call anthropic create function by sending all the required parameters
message = claude.messages.create(
    model="gpt-3.5-turbo",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        { "role": "user", "content": user_prompt }
    ]
)

print(message.content[0].text)

