import os
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import pipeline
import torch

print(torch.__version__)  
print(torch.cuda.is_available())  
print(torch.version.cuda)


load_dotenv()

hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# Sentiment Analysis

# To run on GPU we need Torch with CUDA enabled
# use below command to install torch with CUDA support
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

#classifier = pipeline("sentiment-analysis", device="cuda")
classifier = pipeline("sentiment-analysis")
result = classifier("I'm super excited to be on the way to LLM mastery!")
print(result)
