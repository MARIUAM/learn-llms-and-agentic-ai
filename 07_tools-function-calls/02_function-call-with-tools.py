import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import requests

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

def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    print(f"Tool get_weather Getting weather for latitude = {latitude}, longitude = {longitude}")
    print(f"Result is  = {data['current']['temperature_2m']}")
    return data['current']['temperature_2m']

def get_weather_function():
    return {
        "name": "get_weather",
        "description": "Get the weather for a location. Call this whenever you need to know the weather, for example when a customer asks 'What's the weather like in this city'",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Latitude of the location"
                },
                "longitude": {
                    "type": "number",
                    "description": "Longitude of the location"
                }
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }

tools = [{"type": "function", "function": get_weather_function()}]

completion = openai.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant and provide update on weather in a city."},
        {"role": "user", "content": "What is the weather like in Karachi, Pakistan?"}
    ],
    tools=tools
)

def call_openai(prompt):
    messages=[
            {"role": "system", "content": "You are a helpful assistant and provide update on weather in a city."},
            {"role": "user", "content": prompt}
    ]
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    if completion.choices[0].finish_reason == "tool_calls":
        tool_call = completion.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        latitude = arguments.get("latitude")
        longitude = arguments.get("longitude")
        weather = get_weather(latitude, longitude)
        response = {
            "role": "tool",
            "content": json.dumps({"latitude": latitude, "longitude": longitude, "weather": weather}),
            "tool_call_id": tool_call.id
        }
        messages.append(completion.choices[0].message)
        messages.append(response)
        completion2 = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        print(completion2.choices[0].message.content)


#call_openai("What is the weather like in Karachi, Pakistan?")
call_openai("Karachi")
#call_openai("Berlin, Germany")
#call_openai("Turkey")
#call_openai("NYC")