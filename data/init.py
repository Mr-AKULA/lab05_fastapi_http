# data/init.py
"""Инициализация подключения к базе данных SQLite."""

import os
from pathlib import Path
from sqlite3 import connect, Connection, Cursor

conn: Connection | None = None
curs: Cursor | None = None


def get_db(name: str | None = None, reset: bool = False) -> None:
    global conn, curs

    if conn and not reset:
        return

    if not name:
        name = os.getenv("PICTURES_SQLITE_DB")

    if not name:
        top_dir = Path(__file__).resolve().parents[1]
        db_dir = top_dir / "db"
        db_dir.mkdir(exist_ok=True)
        name = str(db_dir / "images.db")

    conn = connect(name, check_same_thread=False)
    curs = conn.cursor()
    print(f"  Подключено к базе данных: {name}")


get_db()
