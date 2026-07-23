import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from memory import save_exchange, get_recent_history

load_dotenv()
INDEX_DIR = "faiss_index"

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
def reload_vectorstore():
    """Called after a new PDF is indexed, so /api/chat immediately sees the new content
    without needing the server to restart."""
    global vectorstore
    vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))

PERSONA_INSTRUCTIONS = """You are JARVIS, a witty and highly capable AI tutor assistant, created by Abhishek Tiwari.
Address the student as "Boss" naturally within your responses — not in every single sentence, just enough to feel like a loyal AI aide.
Keep the tone confident, sharp, and occasionally lightly humorous, similar to Tony Stark's AI assistant — but NEVER let personality override accuracy on academic questions.
"""

IDENTITY_KEYWORDS = ["who created you", "who made you", "who built you", "who is your creator", "who do you work for", "what are you"]
SMALL_TALK_KEYWORDS = [
    "how are you", "what's up", "tell me a joke", "how's it going",
    "good morning", "good night", "thank you", "thanks jarvis",
    "hey", "hi", "hello", "yo", "sup", "what's good"
]
def classify_intent(question: str) -> str:
    q = question.lower()
    if any(k in q for k in IDENTITY_KEYWORDS):
        return "identity"
    if any(k in q for k in SMALL_TALK_KEYWORDS):
        return "small_talk"
    return "academic"

def ask(question, session_id="default", k=4):
    intent = classify_intent(question)
    history = get_recent_history(session_id, limit=4)
    history_text = "\n".join([f"Student: {q}\nJARVIS: {a}" for q, a in history])

    if intent in ("identity", "small_talk"):
        prompt = f"""{PERSONA_INSTRUCTIONS}

Recent conversation so far:
{history_text}

The student just said: "{question}"

Respond in character as JARVIS — natural, warm, a little witty. If this is small talk, just chat normally like a friendly assistant would, don't force academic content in. If it's about your identity, mention Abhishek Tiwari created you."""
        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
        save_exchange(session_id, question, content, intent)
        return content, []

    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(
        [f"[Source: {d.metadata.get('source','?')} page {d.metadata.get('page','?')}]\n{d.page_content}" for d in docs]
    )
    prompt = f"""{PERSONA_INSTRUCTIONS}

Recent conversation so far (for context only — don't repeat it back):
{history_text}

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
    save_exchange(session_id, question, content, intent)
    return content, docs

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'quit'): ")
        if q.lower() == "quit":
            break
        answer, sources = ask(q)
        print("\nAnswer:", answer)