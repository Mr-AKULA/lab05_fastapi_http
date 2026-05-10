# data/pictures.py
"""Уровень данных: CRUD-операции с изображениями в SQLite (DB-API / PEP 249)."""

import cv2
import numpy as np
from datetime import datetime

from .init import conn, curs
from model.pictures import Picture
from error import Missing, Duplicate

curs.execute('''
    CREATE TABLE IF NOT EXISTS pictures (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL UNIQUE,
        description TEXT,
        image_data  BLOB    NOT NULL,
        created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()


def row_to_model(row: tuple) -> Picture | None:
    if row is None or len(row) < 5:
        return None

    try:
        _, name, description, image_bytes, created_at = row

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        dt = (
            datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            if isinstance(created_at, str)
            else created_at
        )

        return Picture(name=name, img=img, description=description or "", dt=dt)
    except Exception:
        return None


def add_one(picture: Picture) -> int:
    curs.execute("SELECT id FROM pictures WHERE name = ?", (picture.name,))
    if curs.fetchone():
        raise Duplicate(f"Изображение '{picture.name}' уже существует")

    success, encoded_img = cv2.imencode(".png", picture.img)
    if not success:
        raise ValueError("Не удалось закодировать изображение в PNG")

    image_blob = encoded_img.tobytes()
    created_at_str = picture.dt.strftime("%Y-%m-%d %H:%M:%S")

    curs.execute(
        "INSERT INTO pictures (name, description, image_data, created_at) VALUES (?, ?, ?, ?)",
        (picture.name, picture.description, image_blob, created_at_str),
    )
    conn.commit()
    return curs.lastrowid


def get_one(name: str) -> Picture | None:
    curs.execute("SELECT * FROM pictures WHERE name = ?", (name,))
    row = curs.fetchone()
    return row_to_model(row)


def get_all() -> list[Picture]:
    curs.execute("SELECT * FROM pictures")
    rows = curs.fetchall()
    result = []
    for row in rows:
        pic = row_to_model(row)
        if pic:
            result.append(pic)
    return result


def modify_one(name: str, description: str) -> Picture | None:
    curs.execute(
        "UPDATE pictures SET description = ? WHERE name = ?",
        (description, name),
    )
    conn.commit()

    if curs.rowcount == 0:
        return None

    return get_one(name)


def delete_one(name: str) -> bool:
    curs.execute("DELETE FROM pictures WHERE name = ?", (name,))
    conn.commit()
    return curs.rowcount > 0
