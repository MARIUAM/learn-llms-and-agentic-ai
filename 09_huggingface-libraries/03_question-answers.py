import os
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)


# Question Answering with Context

question_answerer = pipeline("question-answering", device="cuda")
result = question_answerer(question="Who was the 44th president of the United States?", context="Barack Obama was the 44th president of the United States.")
print(result)