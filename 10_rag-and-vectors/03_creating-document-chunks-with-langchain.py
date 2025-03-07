import os
import glob
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

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

for chunk in chunks:
    if "Car" in chunk.page_content:
        print(chunk)
        print("<><><><>")