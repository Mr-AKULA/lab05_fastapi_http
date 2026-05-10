# src/main.py
"""
Точка входа FastAPI-приложения.

Запуск сервера:
    uvicorn src.main:app --reload

Приложение объединяет два роутера:
  1. /pictures  — CRUD-операции с изображениями в SQLite + обработка OpenCV
  2. /classify  — классификация изображений через CNN CIFAR-10

Swagger UI (интерактивная документация): http://127.0.0.1:8000/docs
ReDoc (альтернативная документация):     http://127.0.0.1:8000/redoc
"""

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импорты вида
# «from model.pictures import ...» работали при запуске через uvicorn
# из любой директории (не только из корня проекта).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from web.pictures import router as pictures_router
from web.classifier import router as classifier_router

app = FastAPI(
    title="FastAPI Image Processing & Classification",
    description=(
        "Лабораторная работа №5. Трёхуровневая архитектура FastAPI:\n\n"
        "- **data/** — SQLite через DB-API (PEP 249)\n"
        "- **service/** — бизнес-логика: OpenCV + CNN-инференс\n"
        "- **web/** — HTTP-роутеры FastAPI\n\n"
        "Модель CIFAR10CNN обучена на датасете CIFAR-10 (10 классов, точность ~81%)."
    ),
    version="2.0.0",
)

# Регистрируем роутеры — каждый добавляет свой набор эндпоинтов
app.include_router(pictures_router)   # /pictures/*
app.include_router(classifier_router) # /classify/*


@app.get("/", tags=["root"])
def root() -> dict:
    """Корневой маршрут — проверка работоспособности сервиса."""
    return {"status": "ok", "service": "FastAPI Image Processing & Classification"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
