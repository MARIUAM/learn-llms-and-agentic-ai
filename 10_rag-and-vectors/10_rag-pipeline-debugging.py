import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import numpy as np
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler

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

#embeddings = OpenAIEmbeddings()

# If you would rather use the free Vector Embeddings from HuggingFace sentence-transformers
# Then replace embeddings = OpenAIEmbeddings()
# with:
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# use this if you want to create new vectors
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# use this if you want to load existing vectors from the database
# vectorstore = Chroma(persist_directory=db_name, embedding_function=embeddings)

print(f"VectorStore created with {vectorstore._collection.count()} documents")


# Get one vector and find how many dimensions it has
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")


llm = ChatOpenAI(model=MODEL, temperature=0.7)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# With this call you will not get response for below question
# retriever = vectorstore.as_retriever()

retriever = vectorstore.as_retriever(search_kwargs={"k":25})

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

query = "Who received the prestigious IIOTY award in 2023?" 
result = conversation_chain.invoke({"question":query})
print(result["answer"]) 
print()