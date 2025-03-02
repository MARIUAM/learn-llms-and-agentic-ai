import os
from huggingface_hub import login
from dotenv import load_dotenv
from diffusers import DiffusionPipeline
from IPython.display import display, Audio 
from PIL import Image
import torch

load_dotenv()

hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)


image_gen = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
    ).to("cuda")

text = "A class of Data Scientists learning about AI, in the surreal style of Salvador Dali"
image = image_gen(prompt=text).images[0]

image.save("surreal.png")
display(image)