import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")
    
MODEL = "gpt-4o-mini"
IMAGE_MODEL = "dall-e-2"
#IMAGE_MODEL = "dall-e-3"

openai = OpenAI()

def generate_image(city: str) -> Image:
    image_response = openai.images.generate(
        model=IMAGE_MODEL,
        prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
        n=1,
        response_format="b64_json",
        #size="256x256"
        size="1024x1024"
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

#gr.image(generate_image("Paris"))

gr.Interface(fn=generate_image, 
             inputs="text", 
             outputs=gr.Image(height=1024, width=1024),
             title="City Image Generator", 
             description="Generate an image of a city using DALL-E 2").launch()
