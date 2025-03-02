import os
from huggingface_hub import login
from dotenv import load_dotenv
from transformers import pipeline
from IPython.display import display, Audio 
from datasets import load_dataset
import soundfile as sf
import torch

load_dotenv()

hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)


synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesiser("Hi to an artificial intelligence engineer, on the way to mastery!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

Audio("speech.wav")