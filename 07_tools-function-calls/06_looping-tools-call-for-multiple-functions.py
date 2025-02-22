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


MODEL, openai = ollamaAPIModal()
MODEL, openai = openAIGPT4oMiniModal()

class SendEmail(BaseModel):
    to: str = Field(..., description="Email address of the recipient")
    subject: str = Field(..., description="Subject of the email")
    body: str = Field(..., description="Body of the email")

class GetWeather(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")


def send_email(to, subject, body):
    print(f"Tool send_email Sending email to {to} with subject {subject}")
    print(f"Body: {body}")
    print(f"Email Tool call completed")
    return "Email Sent"

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    print(f"Tool get_weather Getting weather for latitude = {latitude}, longitude = {longitude}")
    print(f"Result is  = {data['current']['temperature_2m']}")
    return data['current']['temperature_2m']


tools = [pydantic_function_tool(SendEmail), pydantic_function_tool(GetWeather)]
print("tools", tools)


def call_openai(messages):
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    return completion


def handle_tool_call(messages, message):
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        #weather = send_email(**args)
        result = None
        if name == "SendEmail":
            result = send_email(**args)
        elif name == "GetWeather":
            result = get_weather(**args)
        response = {
            "role": "tool",
            "content": str(result),
            "tool_call_id": tool_call.id
        }
        messages.append(response)

def ask_model(prompt):
    messages=[
            #{"role": "system", "content": "You are a helpful assistant, you can provide update on weather in a city. Also you can send email the weather update. Email should be courteous and professional. If you are asked to send only email then just send email without considring weather."},
            #{"role": "system", "content": "You are a helpful assistant, you can provide update on weather in a city and send email the weather update. Email should be courteous and professional."},
            {"role": "system", "content": "You are a helpful assistant, you can send email about weather in a city. Email should be courteous and professional."},
            {"role": "user", "content": prompt}
    ]
    completion = call_openai(messages)
    counter = 0
    while True:
        print("Finish Reason = ",completion.choices[0].finish_reason)
        print("Tool Calls Length = ",len(completion.choices[0].message.tool_calls or []))
        print("Tool Calls = ",completion.choices[0].message.tool_calls)
        print("Tool Calls Ended")
        if(completion.choices[0].finish_reason == "stop"):
            print("Tool calls from model are = ",counter)
            print(completion.choices[0].message.content)
            break
        elif completion.choices[0].finish_reason == "tool_calls":
            counter += 1
            print("Recevied tool calls from model counter = ",counter)
            messages.append(completion.choices[0].message)
            handle_tool_call(messages, completion.choices[0].message)
            completion = call_openai(messages)
    
# Now we can call the model in a loop and it is calling multiple get weather function in single 
# tool call but for email it is calling one by one

#ask_model("send an email to shan@gmail.com and hello@gmail.com about weather in Karachi")

# This is sending one weather info to one email and then second weather to second email
#ask_model("send an email to shan@gmail.com and hello@gmail.com about weather in Karachi and Lahore") 

ask_model("send an email about weather in Karachi and Lahore to both shan@gmail.com and hello@gmail.com")
#call_openai("send an email to shan@gmail.com about the weather in Karachi")
#call_openai("send an email to shan@gmail.com and hello@gmail.com about the weather in Lahore")