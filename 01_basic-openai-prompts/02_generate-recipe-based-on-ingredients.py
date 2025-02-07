import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith('sk-proj-'):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")

openai = OpenAI()


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
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
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
display_recipe("create korean recipes for chicken, rice, Cauliflower")