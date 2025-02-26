import os
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)


# Named Entity Recognition

ner = pipeline("ner", grouped_entities=True, device="cuda")
result = ner("Barack Obama was the 44th president of the United States.")
print(result)