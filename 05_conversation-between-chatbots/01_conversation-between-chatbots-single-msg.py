import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set")


openai = OpenAI()
google.generativeai.configure()

gpt_model = "gpt-4o-mini"
gemini_model = "gemini-2.0-flash"

gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way."

gemini_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting."

gpt_messages = ["Hi there"]
gemini_messages = ["Hi"]

def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt_msg, gemini_msg in zip(gpt_messages, gemini_messages):
        messages.append({"role": "assistant", "content": gpt_msg})
        messages.append({"role": "user", "content": gemini_msg})
    completion = openai.chat.completions.create(
        model=gpt_model,
        messages=messages
    )
    return completion.choices[0].message.content

gpt_response_msg = call_gpt()
gpt_messages.append(gpt_response_msg)
print(gpt_response_msg)

def call_gemini():
    #messages = [{"role": "system", "content": gemini_system}]
    messages = []
    for gpt_msg, gemini_msg  in zip(gpt_messages, gemini_messages):
        messages.append({"role": "user", "parts": gpt_msg})
        messages.append({"role": "model", "parts": gemini_msg})
    messages.append({"role": "user", "parts": gpt_messages[-1]})

    gemini = google.generativeai.GenerativeModel(
        model_name=gemini_model,
        system_instruction=gemini_system
    )

    response = gemini.generate_content(messages)
    return response.text

gemini_response_msg = call_gemini()
gemini_messages.append(gemini_response_msg)
print(gemini_response_msg)
