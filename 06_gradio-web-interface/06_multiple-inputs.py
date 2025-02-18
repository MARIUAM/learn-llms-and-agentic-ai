import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import google.generativeai

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google.generativeai.configure(api_key=os.getenv("Google_API_KEY"))

gpt_model = "gpt-4o-mini"
gemini_model = "gemini-2.0-flash"
#gemini_model = "gemini-2.0-flash-lite-preview-02-05"


system_message = "You are a helpful assistant that responds in markdown"

def message_gpt_stream(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    stream = openai.chat.completions.create(
        model= gpt_model,
        messages=messages,
        stream=True
    )
    result =""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

def message_gemini_stream(prompt):
    messages = [
        {"role": "user", "parts": prompt}
    ]
    gemini = google.generativeai.GenerativeModel(
        model_name= gemini_model,
        system_instruction= system_message,
        generation_config={
            "temperature": 1
        }
    )
    response = gemini.generate_content(messages, stream=True)
    result = ""
    for chunk in response:
        result += chunk.text
        yield result

def message_with_model(prompt, model):
    if model == "GPT":
        result = message_gpt_stream(prompt)
    elif model == "Gemini":
        result = message_gemini_stream(prompt)
    else:
        raise ValueError("Invalid model")
    yield from result



view = gr.Interface(
    fn=message_with_model,
    inputs=[gr.Textbox(label="Your message"), gr.Dropdown(["GPT", "Gemini"], label="Model")],
    outputs=[gr.Markdown(label="Response")],
    flagging_mode="never"
)
view.launch()