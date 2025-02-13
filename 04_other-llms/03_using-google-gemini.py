import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
from IPython.display import Markdown, display, update_display

load_dotenv(override=True)

google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")

google.generativeai.configure()

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell 10 light-hearted jokes for an audience of Data Scientists"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

gemini = google.generativeai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=system_message
)

response = gemini.generate_content(user_prompt)

print(response.text)

