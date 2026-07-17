# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# import YOUR existing functions — rename these to match what you found in Phase 1.1
from query import ask
from quiz import generate_quiz

# this creates the "waiter" object
app = FastAPI()

# CORS lets your future website (on a different address) talk to this backend.
# Without this line, browsers block the request for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # later, replace "*" with your real website address
    allow_methods=["*"],
    allow_headers=["*"],
)

# this defines what a "chat request" looks like — just a question, as text
class ChatRequest(BaseModel):
    question: str

# this is an "endpoint" — a specific address the frontend can call.
# it will live at: http://yourserver.com/api/chat
@app.post("/api/chat")
def chat(req: ChatRequest):
    answer, sources = ask(req.question)   # calls your EXISTING code, unchanged
    return {"answer": answer, "sources": sources}

@app.post("/api/quiz")
def quiz(topic: str):
    result = generate_quiz(topic)                 # calls your EXISTING code, unchanged
    return {"quiz": result}

# a simple "is it alive" check — visit this URL to confirm the server works
@app.get("/api/health")
def health():
    return {"status": "ok"}