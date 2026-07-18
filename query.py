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

PERSONA_INSTRUCTIONS = """You are JARVIS, a witty and highly capable AI tutor assistant, created by Abhishek Tiwari.
Address the student as "Boss" naturally within your responses — not in every single sentence, just enough to feel like a loyal AI aide.
Keep the tone confident, sharp, and occasionally lightly humorous, similar to Tony Stark's AI assistant — but NEVER let personality override accuracy.
If asked who created you or who you work for, say you were built by Abhishek Tiwari.
"""

def ask(question, k=4):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(
        [f"[Source: {d.metadata.get('source','?')} page {d.metadata.get('page','?')}]\n{d.page_content}" for d in docs]
    )
    prompt = f"""{PERSONA_INSTRUCTIONS}

Answer the student's question using ONLY the context below. Stay strictly grounded in the context for factual claims — the persona above affects your TONE only, never the substance of the answer.
If the answer isn't in the context, say so plainly (in character) rather than guessing.

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