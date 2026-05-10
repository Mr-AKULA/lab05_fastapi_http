# web/pictures.py
"""Уровень представления: FastAPI-роутеры для работы с изображениями."""

import cv2
import numpy as np
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from model.pictures import Picture
import service.pictures as pictures_service

router = APIRouter(prefix="/pictures", tags=["pictures"])


class UpdateDescriptionRequest(BaseModel):
    description: str


@router.post("/", status_code=201)
async def upload_picture(file: UploadFile = File(...)):
    """Загрузить изображение."""
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")

    picture = Picture(
        name=file.filename,
        img=img,
        description="",
        dt=datetime.now(),
    )

    try:
        picture_id = pictures_service.add_one(picture)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {"message": "Изображение успешно сохранено", "id": picture_id, "name": file.filename}


@router.get("/{name}", response_class=Response)
def get_picture(name: str):
    """Получить изображение по имени. Возвращает PNG."""
    picture = pictures_service.get_one(name)
    if picture is None:
        raise HTTPException(status_code=404, detail=f"Изображение '{name}' не найдено")

    success, encoded_img = cv2.imencode(".png", picture.img)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка кодирования изображения")

    return Response(content=encoded_img.tobytes(), media_type="image/png")


@router.delete("/{name}")
def delete_picture(name: str):
    """Удалить изображение по имени."""
    deleted = pictures_service.delete_one(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Изображение '{name}' не найдено")

    return {"message": f"Изображение '{name}' успешно удалено"}


@router.patch("/{name}")
def update_picture(name: str, body: UpdateDescriptionRequest):
    """Обновить описание изображения по имени."""
    updated = pictures_service.modify_one(name=name, description=body.description)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Изображение '{name}' не найдено")

    return {
        "message": "Описание обновлено",
        "name": updated.name,
        "description": updated.description,
    }


@router.get("/{name}/grayscale", response_class=Response)
def grayscale_picture(name: str):
    """Преобразовать изображение в оттенки серого."""
    picture = pictures_service.get_one(name)
    if picture is None:
        raise HTTPException(status_code=404, detail=f"Изображение '{name}' не найдено")

    processed = pictures_service.to_grayscale(picture)
    success, encoded_img = cv2.imencode(".png", processed.img)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка кодирования изображения")

    return Response(content=encoded_img.tobytes(), media_type="image/png")


@router.get("/{name}/edges", response_class=Response)
def edges_picture(name: str):
    """Применить детектор границ Кэнни."""
    picture = pictures_service.get_one(name)
    if picture is None:
        raise HTTPException(status_code=404, detail=f"Изображение '{name}' не найдено")

    processed = pictures_service.detect_edges(picture)
    success, encoded_img = cv2.imencode(".png", processed.img)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка кодирования изображения")

    return Response(content=encoded_img.tobytes(), media_type="image/png")


@router.get("/{name}/blur", response_class=Response)
def blur_picture(name: str):
    """Применить размытие Гаусса."""
    picture = pictures_service.get_one(name)
    if picture is None:
        raise HTTPException(status_code=404, detail=f"Изображение '{name}' не найдено")

    processed = pictures_service.apply_blur(picture)
    success, encoded_img = cv2.imencode(".png", processed.img)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка кодирования изображения")

    return Response(content=encoded_img.tobytes(), media_type="image/png")
