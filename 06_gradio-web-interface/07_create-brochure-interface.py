import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
#from IPython.display import Markdown, display, update_display
from openai import OpenAI
import gradio as gr
import google.generativeai

# Load environment variables
load_dotenv()

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google.generativeai.configure(api_key=os.getenv("Google_API_KEY"))

gpt_model = "gpt-4o-mini"
gemini_model = "gemini-2.0-flash"

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:
    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"
    

def get_brochure_system_prompt():
    brochure_system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."
    return brochure_system_prompt

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += Website(url).get_contents()
    return user_prompt


def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": get_brochure_system_prompt()},
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

def stream_gemini(prompt):
    messages = [
        {"role": "user", "parts": prompt}
    ]
    gemini = google.generativeai.GenerativeModel(
        model_name= gemini_model,
        system_instruction= get_brochure_system_prompt(),
    )
    response = gemini.generate_content(messages, stream=True)
    result = ""
    for chunk in response:
        result += chunk.text
        yield result


def stream_brochure(company_name, url, model):
    prompt = get_brochure_user_prompt(company_name, url)

    if model == "GPT":
        result = stream_gpt(prompt)
    elif model == "Gemini":
        result = stream_gemini(prompt)
    else:
        raise ValueError("Invalid model")
    yield from result

#create_brochure("HuggingFace", "https://huggingface.co")
#stream_brochure("HuggingFace", "https://huggingface.co")

view = gr.Interface(
    fn=stream_brochure,
    inputs=[gr.Textbox(label="Company Name"), gr.Textbox(label="Company URL"), gr.Dropdown(["GPT", "Gemini"], label="Model")],
    outputs=[gr.Markdown(label="Brochure")],
    title="Create a Brochure",
    description="Create a short brochure about a company for prospective customers, investors and recruits.",
    article="https://ollama.com",
    flagging_mode="never",
    examples=[[ "HuggingFace", "https://huggingface.co", "GPT"], ["Gradio", "https://www.gradio.app/", "Gemini"], ["Gradio", "https://www.gradio.app/", "GPT"]]
).launch(share=True)

