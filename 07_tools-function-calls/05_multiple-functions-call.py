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

class GetWeather(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")


def send_email(to, subject, body):
    print(f"Tool send_email Sending email to {to} with subject {subject}")
    print(f"Body: {body}")
    print(f"Tool call completed")
    return "Email Sent"

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    print(f"Tool get_weather Getting weather for latitude = {latitude}, longitude = {longitude}")
    print(f"Result is  = {data['current']['temperature_2m']}")
    return data['current']['temperature_2m']


tools = [pydantic_function_tool(SendEmail), pydantic_function_tool(GetWeather)]
print("tools", tools)

def call_openai(prompt):
    messages=[
            #{"role": "system", "content": "You are a helpful assistant, you can provide update on weather in a city. Also you can send email the weather update. Email should be courteous and professional. If you are asked to send only email then just send email without considring weather."},
            #{"role": "system", "content": "You are a helpful assistant, you can provide update on weather in a city and send email the weather update. Email should be courteous and professional."},
            {"role": "system", "content": "You are a helpful assistant, you can send email about weather in a city. Email should be courteous and professional."},
            {"role": "user", "content": prompt}
    ]
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    print(completion.choices[0].message)
    if completion.choices[0].finish_reason == "tool_calls":
        #print(completion.choices[0].message)
        messages.append(completion.choices[0].message)
        handle_tool_call(messages, completion.choices[0].message)
        print("messages", messages)
        completion2 = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        print("Completion 2 message")
        print(completion2.choices[0].finish_reason)
        print(completion2.choices[0].message)
        print(completion2.choices[0].message.content)


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
    
# Because model needs to call multiple functions, we need to call the model in a loop
# Initially model is calling weather function and once we send that reponse to the model, 
# it will call email function


#call_openai("send an email to shan@gmail.com about the weather in Karachi")
#call_openai("send an email to shan@gmail.com and hello@gmail.com about the weather in Lahore")
#call_openai("send an email to shan@gmail.com and hello@gmail.com about internet history")
call_openai("send an email to shan@gmail.com and hello@gmail.com about weather in Karachi")
# call_openai("Berlin, Germany")
#call_openai("Turkey")
#call_openai("NYC")