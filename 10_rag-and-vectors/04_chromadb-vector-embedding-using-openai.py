import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

load_dotenv()

MODEL = "gpt-4o-mini"
db_name = "vector_db"

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


folders = glob.glob("./knowledge-base/*")
print(folders)

text_loader_kwargs = {"encoding": "utf-8"}
documents = []

for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    # print(folder_docs)
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(len(documents))

text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(len(chunks))

doc_types = set(chunk.metadata["doc_type"] for chunk in chunks)
print(f"Document types found: {doc_types}")

embeddings = OpenAIEmbeddings()

# use this if you want to create new vectors
# if os.path.exists(db_name):
#     Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
# vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# use this if you want to load existing vectors from the database
vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)

print(f"VectorStore created with {vectorstore._collection.count()} documents")


# Get one vector and find how many dimensions it has
# So lenth of embaddings means number of chunks for all documents and each
# chunk/embadding will have 1536 dimensions

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")
