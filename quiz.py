import json
import os
from datetime import datetime
from query import vectorstore, llm

RESULTS_FILE = "quiz_results.json"

def generate_quiz(topic, num_questions=5):
    docs = vectorstore.similarity_search(topic, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""Based on the following material, write {num_questions} multiple choice questions about "{topic}".
For each question, give 4 options (A-D) and clearly mark the correct answer.
Respond with ONLY a JSON list, nothing else, in this exact format:
[{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A"}}]

Material:
{context}
"""
    response = llm.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        content = "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
    text = content.strip().strip("```json").strip("```").strip()
    return json.loads(text)

def save_result(topic, score_percent):
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    results.append({"topic": topic, "score": score_percent, "date": str(datetime.now())})
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def get_weak_topics(threshold=60):
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    weak = [r["topic"] for r in results if r["score"] < threshold]
    return list(set(weak))