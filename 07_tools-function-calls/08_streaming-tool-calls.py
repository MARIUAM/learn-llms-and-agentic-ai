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

class GetWeather(BaseModel):
    latitude: float = Field(..., description="Latitude of the location")
    longitude: float = Field(..., description="Longitude of the location")


def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    print(f"Tool get_weather Getting weather for latitude = {latitude}, longitude = {longitude}")
    print(f"Result is  = {data['current']['temperature_2m']}")
    return data['current']['temperature_2m']


tools = [pydantic_function_tool(GetWeather)]
print("tools", tools)

def accumulte_tool_calls(stream, messages):
    final_tool_calls = {}
    content = ""
    is_tool_calls = False
    for chunk in stream:
        content += chunk.choices[0].delta.content or ''
        for tool_call in chunk.choices[0].delta.tool_calls or []:
            index = tool_call.index

            if index not in final_tool_calls:
                final_tool_calls[index] = tool_call

            final_tool_calls[index].function.arguments += tool_call.function.arguments
        if chunk.choices[0].finish_reason == "tool_calls":
            is_tool_calls = True
            print("Tool calls detected", [val for val in final_tool_calls.values()])
            messages.append({"role": "assistant", "content": None, "tool_calls": [val for val in final_tool_calls.values()]})
    return final_tool_calls, content, is_tool_calls


def call_model(messages, stream=False):
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        stream=stream
    )
    return completion


def call_openai(prompt):
    messages=[
            {"role": "system", "content": "You are a helpful assistant and provide update on weather in a city."},
            {"role": "user", "content": prompt}
    ]
    stream = call_model(messages, stream=True)

    tool_calls, content, is_tool_calls = accumulte_tool_calls(stream, messages)
    print("tool_calls", tool_calls)
    print("content", content)
    print("isToolCall", is_tool_calls)

    if is_tool_calls:
        tool_call = tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        latitude = arguments.get("latitude")
        longitude = arguments.get("longitude")
        weather = get_weather(latitude, longitude)
        response = {
            "role": "tool",
            "content": json.dumps({"latitude": latitude, "longitude": longitude, "weather": weather}),
            "tool_call_id": tool_call.id
        }
        messages.append(response)
        # print(messages)
        stream2 = call_model(messages, stream=True)
        tool_calls2, content2, is_tool_calls2 = accumulte_tool_calls(stream2, messages)
        print("tool_calls2", tool_calls2)
        print("content2", content2)
        print("isToolCall2", is_tool_calls2)

    # for chunk in stream:
    #     print(chunk)
    #     print()
    #     print(chunk.choices[0])
    #     print()
    #     print(chunk.choices[0].delta)
    #     print()
    #     delta = chunk.choices[0].delta
    #     print(delta.tool_calls)



call_openai("hello")
#call_openai("What is the weather like in Karachi")
#call_openai("What is the weather like in Karachi, Pakistan?")
#call_openai("What is the weather like in Karachi and Lahore") # This will fail as we are not handling multiple locations
# call_openai("Karachi")
# call_openai("Berlin, Germany")
#call_openai("Turkey")
#call_openai("NYC")