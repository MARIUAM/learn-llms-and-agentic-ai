import os
import json
from dotenv import load_dotenv
from openai import OpenAI, pydantic_function_tool
import gradio as gr
import requests
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

def openAIGPT4oMiniModal():
    api_key = os.getenv('OPENAI_API_KEY')

    if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:
        print("API key looks good so far")
    else:
        print("There might be a problem with your API key? Please visit the troubleshooting notebook!")
    
    MODEL = 'gpt-4o-mini'
    openai = OpenAI()
    return MODEL,openai

def ollamaAPIModal():
    OLLAMA_API = "http://localhost:11434/v1"

    # use any modal
    MODEL = "llama3.2:1b"
    #MODEL = "qwen2.5:0.5b"

    openai = OpenAI(base_url=OLLAMA_API, api_key="ollama")
    return MODEL,openai


#MODEL, openai = ollamaAPIModal()
MODEL, openai = openAIGPT4oMiniModal()

class SendEmail(BaseModel):
    to: str = Field(..., description="Email address of the recipient")
    subject: str = Field(..., description="Subject of the email")
    body: str = Field(..., description="Body of the email")


def send_email(to, subject, body):
    print(f"Tool send_email Sending email to {to} with subject {subject}")
    print(f"Body: {body}")
    print(f"Tool call completed")
    return "Email Sent"


tools = [pydantic_function_tool(SendEmail)]
print("tools", tools)

def call_openai(prompt):
    messages=[
            {"role": "system", "content": "You are a helpful assistant and you can send an email. Email should be courteous and professional."},
            {"role": "user", "content": prompt}
    ]
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    if completion.choices[0].finish_reason == "tool_calls":
        #print(completion.choices[0].message)
        messages.append(completion.choices[0].message)
        handle_tool_call(messages, completion.choices[0].message)
        completion2 = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        print(completion2.choices[0].message.content)


def handle_tool_call(messages, message):
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        email_sent = send_email(**args)
        response = {
            "role": "tool",
            "content": str(email_sent),
            "tool_call_id": tool_call.id
        }
        messages.append(response)

#call_openai("send an email to shan@gmail.com about the weather in Karachi")
call_openai("send an email to shan@gmail.com and hello@gmail.com about the weather in Lahore")
# call_openai("Karachi")
# call_openai("Berlin, Germany")
#call_openai("Turkey")
#call_openai("NYC")