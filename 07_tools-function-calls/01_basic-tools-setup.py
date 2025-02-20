import os
from dotenv import load_dotenv
from openai import OpenAI

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

def get_weather_function():
    return {
        "name": "get_weather",
        "description": "Get the weather for a location. Call this whenever you need to know the weather, for example when a customer asks 'What's the weather like in this city'",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and Countery e.g Karachi, Pakistan"
                }
            },
            "required": ["location"],
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
    completion = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant and provide update on weather in a city."},
            {"role": "user", "content": prompt}
        ],
        tools=tools
    )
    print(completion.choices[0].message.content)
    print(completion.choices[0].message.tool_calls)
    print(completion.choices[0].message.tool_calls[0].function.name)
    print(completion.choices[0].message.tool_calls[0].function.arguments)


#call_openai("What is the weather like in Karachi, Pakistan?")
#call_openai("Karachi")
#call_openai("Berlin, Germany")
#call_openai("Turkey")
call_openai("NYC")