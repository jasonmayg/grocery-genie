import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "grocery_ai.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        budget INTEGER NOT NULL,
        cuisine TEXT NOT NULL,
        diet TEXT NOT NULL,
        pantry TEXT,
        meal_plan TEXT NOT NULL,
        shopping_list TEXT NOT NULL,
        shopping_list_filtered TEXT,
        tokens_total INTEGER,
        estimated_cost REAL
    )
    """)

    conn.commit()
    conn.close()

def save_run(
    budget,
    cuisine,
    diet,
    pantry,
    meal_plan,
    shopping_list,
    shopping_list_filtered,
    tokens_total,
    estimated_cost
):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO runs (
            created_at, budget, cuisine, diet, pantry,
            meal_plan, shopping_list, shopping_list_filtered,
            tokens_total, estimated_cost
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(timespec="seconds"),
        budget, cuisine, diet, pantry,
        meal_plan, shopping_list, shopping_list_filtered,
        tokens_total, estimated_cost
    ))

    conn.commit()
    conn.close()

def load_recent_runs(limit=10):
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM runs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
