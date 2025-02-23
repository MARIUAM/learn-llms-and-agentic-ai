import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

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

system_message = "You are a helpful assistant for an Airline called FlightAI. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499", "karachi": "$599"}

def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price Getting ticket price for {destination_city}")
    return ticket_prices.get(destination_city.lower(), "Unkonwn city")
    

"""
create json object that can can discript the get_ticket_price function and its parameters to the OpenAI API
add description to the function and other important parameters
"""
def get_ticket_price_function():
    return {
        "name": "get_ticket_price",
        "description": "Get the price of a return ticket to a destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
        "parameters": {
            "type": "object",
            "properties": {
                "destination_city": {
                    "type": "string",
                    "description": "The city that the customer wants to travel to"
                }
            },
            "required": ["destination_city"],
            "additionalProperties": False
        }
    }

tools = [{"type": "function", "function": get_ticket_price_function()}]

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get("destination_city")
    ticket_price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city, "price": ticket_price}),
        "tool_call_id": tool_call.id
    }
    return response, city

def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [
        {"role": "user", "content": message}
    ]
    response = openai.chat.completions.create(
        model=MODEL, 
        messages=messages,
        tools=tools
    )

    print(response)
    print()
    print(response.choices[0])
    print()
    print(response.choices[0].finish_reason)
    print()
    print(response.choices[0].message.tool_calls)

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    
    return response.choices[0].message.content

gr.ChatInterface(fn=chat, type="messages",title="FlightAI Chat").launch()