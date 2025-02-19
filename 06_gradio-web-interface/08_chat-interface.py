import os
from dotenv import load_dotenv
import gradio as gr
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


MODEL, openai = ollamaAPIModal()
#MODEL, openai = openAIGPT4oMiniModal()

system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."

system_message += "\nIf the customer asks for shoes, you should respond that shoes are not on sale today, \
but remind the customer to look at hats!"

def chat(message, history):
    messages = [
        {"role": "system", "content": system_message}] + history + [
        {"role": "user", "content": message}
    ]

    stream = openai.chat.completions.create(
        model= MODEL,
        messages=messages,
        stream=True
    )
    result =""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

gr.ChatInterface(fn=chat, type="messages").launch()

