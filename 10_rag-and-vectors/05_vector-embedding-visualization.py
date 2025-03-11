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
collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")


#Prework

# So lenth of embaddings means number of chunks for all documents and each
# chunk/embadding will have 1536 dimensions

result = collection.get(include=["embeddings","documents","metadatas"])
vectors = np.array(result["embeddings"])
# vectors has dimensions for each chunk so 123 vectors means 123 elements each have 1536 dimensions
#print(len(vectors))
#print(vectors[0])
#print(vectors[115])

documents = result["documents"]
# documents will have raw document text but in chunks so 123 means 123 chunks of documents
# print(len(documents))
# print(documents[0])

doc_types = [metadata["doc_type"] for metadata in result["metadatas"]]
# list of all metadata from all of the chunks
#print(len(result["metadatas"]))
#print(result["metadatas"][0])
colors = [["blue","green","red","orange"][["products","employees","contracts","company"].index(t)] for t in doc_types]


# We humans find it easier to visalize things in 2D!
# Reduce the dimensionality of the vectors to 2D using t-SNE
# (t-distributed stochastic neighbor embedding)

tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

# Create the 2D scatter plot
fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
)])

fig.update_layout(
    title='2D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x',yaxis_title='y'),
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40)
)

fig.show()
