import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def helloWorld(name):
    print(f"helloWorld called with {name}   ")
    return "Hello " + name + "!"

def message_gpt(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt}
    ]
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return completion.choices[0].message.content

view = gr.Interface(
    fn=message_gpt,
    inputs=[gr.Textbox(label="Your message", lines=6)],
    outputs=[gr.Textbox(label="Response", lines=8)]
)
view.launch()