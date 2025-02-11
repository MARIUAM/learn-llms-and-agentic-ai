import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

load_dotenv(override=True)

OLLAMA_API = "http://localhost:11434/v1"

# use any modal
MODEL = "llama3.2:1b"
#MODEL = "qwen2.5:0.5b"

ollama_via_openai = OpenAI(base_url=OLLAMA_API, api_key="ollama")

#user_prompt = "spinach, pea, carrot"

def messages_for(user_prompt):
    system_prompt = "You are an expert chef with mastery in all world cuisines \
and a deep understanding of nutritional science. Given a list of ingredients, \
you will create two unique recipes that showcase different culinary styles. \
For each recipe, provide a detailed breakdown of ingredients, step-by-step instructions, \
and estimated nutritional values, including macronutrients (calories, protein, fats, and carbohydrates) \
and key micronutrients. Additionally, identify any health considerations, such as allergen warnings, \
dietary suitability (e.g., vegan, keto, gluten-free), and potential health benefits or risks. \
Your goal is to craft delicious, balanced, and health-conscious meals using the provided ingredients."

    return [
        { "role": "system", "content": system_prompt},
        { "role": "user", "content": user_prompt}
    ]

def generate_recipe(ingredients):
    response = ollama_via_openai.chat.completions.create(
        model=MODEL,
        messages= messages_for(ingredients)
    )
    return response.choices[0].message.content

def display_recipe(ingredients):
    recipe = generate_recipe(ingredients)
    #display(Markdown(recipe))
    print(recipe)

#display_recipe("") # add comma separated ingredients here
#display_recipe("spinach, pea, carrot")

#display_recipe("chicken, rice, Cauliflower")
#display_recipe("create korean recipes for chicken, rice, Cauliflower")
display_recipe("what are the health benefits of chicken, rice, Cauliflower")