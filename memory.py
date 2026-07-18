import sqlite3
from datetime import datetime

DB_PATH = "jarvis_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question TEXT,
            answer TEXT,
            intent TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_exchange(session_id, question, answer, intent):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO conversations (session_id, question, answer, intent, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, question, answer, intent, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def get_recent_history(session_id, limit=6):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT question, answer FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return list(reversed(rows))

def get_last_topic(session_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT question FROM conversations WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        (session_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

init_db()