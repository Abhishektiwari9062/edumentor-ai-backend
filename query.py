import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
INDEX_DIR = "faiss_index"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

def ask(question, k=4):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(
        [f"[Source: {d.metadata.get('source','?')} page {d.metadata.get('page','?')}]\n{d.page_content}" for d in docs]
    )
    prompt = f"""You are a helpful tutor. Answer the student's question using ONLY the context below.
If the answer isn't in the context, say you don't have enough information.

Context:
{context}

Question: {question}

Answer (mention which source you used):"""
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
    return content, docs

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'quit'): ")
        if q.lower() == "quit":
            break
        answer, sources = ask(q)
        print("\nAnswer:", answer)