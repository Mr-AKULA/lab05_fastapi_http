# web/classifier.py
"""
HTTP-роутеры FastAPI для классификации изображений через CNN.

Эндпоинты:
  POST /classify/          — загрузить изображение, получить предсказание
  GET  /classify/info      — диагностика: устройство, параметры, время инференса

Этот модуль реализует только транспортный уровень:
  - валидация входных данных (тип файла, диапазон top_k)
  - декодирование байтов → numpy через OpenCV
  - делегирование бизнес-логики в service.classifier.Classifier
  - формирование JSON-ответа
"""

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from service.classifier import Classifier

# Все маршруты этого роутера будут доступны по префиксу /classify
# tag="classifier" группирует их в отдельный раздел Swagger UI
router = APIRouter(prefix="/classify", tags=["classifier"])


@router.post("/")
async def classify_image(
    file: UploadFile = File(..., description="Изображение для классификации (JPEG, PNG и др.)"),
    top_k: int = 3,
) -> dict:
    """
    Классифицировать изображение с помощью обученной CNN CIFAR-10.

    Принимает файл изображения любого формата, который умеет декодировать OpenCV
    (JPEG, PNG, BMP, TIFF, WebP и др.), и возвращает **top-k** наиболее вероятных
    классов из набора CIFAR-10.

    **Классы CIFAR-10:** airplane, automobile, bird, cat, deer,
    dog, frog, horse, ship, truck.

    **Параметры:**
    - `file` — файл изображения (multipart/form-data)
    - `top_k` — сколько лучших классов вернуть (1..10, по умолчанию 3)

    **Пример ответа:**
    ```json
    {
      "filename": "cat.jpg",
      "top_k": 3,
      "predictions": [
        {"label": "cat",  "probability": 0.8721},
        {"label": "dog",  "probability": 0.0814},
        {"label": "deer", "probability": 0.0231}
      ]
    }
    ```

    **Коды ответа:**
    - 200 — успешная классификация
    - 400 — неверный тип файла или параметр top_k
    - 503 — модель не загружена (нужно запустить ml/train.py)
    """
    # ── Валидация типа файла ──────────────────────────────────────────────────
    # content_type проверяем по MIME-типу, который прислал клиент.
    # Это быстрая проверка — OpenCV дополнительно проверит при декодировании.
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только файлы изображений (image/*)."
        )

    # ── Валидация параметра top_k ─────────────────────────────────────────────
    if not 1 <= top_k <= 10:
        raise HTTPException(
            status_code=400,
            detail="Параметр top_k должен быть в диапазоне 1..10.",
        )

    # ── Чтение и декодирование изображения ───────────────────────────────────
    raw = await file.read()  # читаем байты из multipart-тела запроса

    # np.frombuffer: bytes → одномерный numpy-массив uint8 (плоский поток байт)
    arr = np.frombuffer(raw, dtype=np.uint8)

    # cv2.imdecode: плоский массив байт → матрица BGR (H, W, 3)
    # IMREAD_COLOR всегда возвращает 3-канальное BGR, даже для PNG с альфой
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        # imdecode возвращает None если не смог распознать формат
        raise HTTPException(
            status_code=400,
            detail="Не удалось декодировать изображение.",
        )

    # ── Классификация ─────────────────────────────────────────────────────────
    try:
        # Classifier() — Singleton: модель загружается один раз
        predictions = Classifier().predict(img_bgr, top_k=top_k)
    except FileNotFoundError as exc:
        # Модель не обучена — файл весов отсутствует
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "filename": file.filename,
        "top_k": top_k,
        "predictions": predictions,
    }


@router.get("/info")
def classify_info() -> dict:
    """
    Получить диагностическую информацию о сервисе классификации.

    Возвращает JSON с:
    - `device` — устройство вычислений ("cpu" или "cuda")
    - `num_parameters` — общее число обучаемых параметров модели
    - `last_inference_ms` — время последнего вызова /classify/ в миллисекундах (0 если ещё не было)

    **Пример ответа:**
    ```json
    {
      "device": "cpu",
      "num_parameters": 667178,
      "last_inference_ms": 45.32
    }
    ```
    """
    try:
        return Classifier.get_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
