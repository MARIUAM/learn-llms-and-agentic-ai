import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr

# You can switch between OpenAIEmbeddings or HuggingFaceEmbeddings
# FAISS is in memory database for vector storage

load_dotenv()

MODEL = "gpt-4o-mini"

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


# db_name = "vector_db"
# embeddings = OpenAIEmbeddings()

# If you would rather use the free Vector Embeddings from HuggingFace sentence-transformers
# Then replace embeddings = OpenAIEmbeddings()
# with:
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# If using ChromaDB
#vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# If using FAISS
vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
total_vectors = vectorstore.index.ntotal
dimensions = vectorstore.index.d

print(f"There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store")


llm = ChatOpenAI(model=MODEL, temperature=0.7)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# query = "Can you describe Insurellm in a few sentences"
# result = conversation_chain.invoke({"question":query})
# print(result["answer"])

# print()
# query = "who is Blak" 
# result = conversation_chain.invoke({"question":query})
# print(result["answer"]) # still unable to answer because incorrect spelling

# print()
# query = "Who is Alex?" 
# result = conversation_chain.invoke({"question":query})
# print(result["answer"]) 

def chat(message, history):
    result = conversation_chain.invoke({"question":message})
    return result["answer"]

view = gr.ChatInterface(chat, type="messages").launch()