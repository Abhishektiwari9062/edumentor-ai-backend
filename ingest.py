import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(DATA_DIR, filename)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs

import time

def main():
    print("Loading PDFs...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

    batch_size = 20
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Embedding batch {i // batch_size + 1} of {(len(chunks) - 1) // batch_size + 1}...")
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        time.sleep(15)  # pace requests to stay under the free-tier rate limit

    vectorstore.save_local(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}/")

if __name__ == "__main__":
    main()