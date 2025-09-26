import sqlite3
from datetime import datetime

DB_FILE = "student_progress.db"

def init_db():
    """Initialize database with tables if not exists."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Q&A logs
    c.execute('''
        CREATE TABLE IF NOT EXISTS qa_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )
    ''')

    # Quiz logs
    c.execute('''
        CREATE TABLE IF NOT EXISTS quiz_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            question TEXT,
            correct INTEGER,
            timestamp TEXT
        )
    ''')

    conn.commit()
    conn.close()


def log_qa(student_id: str, question: str, answer: str):
    """Log a Q&A interaction."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO qa_log (student_id, question, answer, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (student_id, question, answer, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def log_quiz(student_id: str, question: str, correct: bool):
    """Log a quiz attempt."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO quiz_log (student_id, question, correct, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (student_id, question, int(correct), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_progress(student_id: str):
    """Retrieve student progress summary."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Total Q&A count
    c.execute("SELECT COUNT(*) FROM qa_log WHERE student_id=?", (student_id,))
    total_qa = c.fetchone()[0]

    # Quiz stats
    c.execute("SELECT COUNT(*), SUM(correct) FROM quiz_log WHERE student_id=?", (student_id,))
    total_quiz, total_correct = c.fetchone()
    conn.close()

    if total_quiz == 0:
        accuracy = 0
    else:
        accuracy = round((total_correct / total_quiz) * 100, 2)

    return {
        "total_qa": total_qa,
        "total_quiz": total_quiz,
        "accuracy": accuracy
    }
