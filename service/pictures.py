# service/pictures.py
"""Уровень сервиса: бизнес-логика и обработка изображений."""

import cv2
import numpy as np
from datetime import datetime

import data.pictures as pictures_data
from model.pictures import Picture


def get_one(name: str) -> Picture | None:
    return pictures_data.get_one(name=name)


def get_all() -> list[Picture]:
    return pictures_data.get_all()


def add_one(picture: Picture) -> int:
    return pictures_data.add_one(picture)


def modify_one(name: str, description: str) -> Picture | None:
    return pictures_data.modify_one(name=name, description=description)


def delete_one(name: str) -> bool:
    return pictures_data.delete_one(name=name)


def to_grayscale(picture: Picture) -> Picture:
    gray = cv2.cvtColor(picture.img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    new_name = f"{picture.name}_gray"
    new_pic = Picture(
        name=new_name,
        img=gray_bgr,
        description=f"Grayscale of {picture.name}",
        dt=datetime.now(),
    )

    if pictures_data.get_one(new_name):
        pictures_data.delete_one(new_name)
    pictures_data.add_one(new_pic)

    return new_pic


def detect_edges(picture: Picture) -> Picture:
    gray = cv2.cvtColor(picture.img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    new_name = f"{picture.name}_edges"
    new_pic = Picture(
        name=new_name,
        img=edges_bgr,
        description=f"Canny edges of {picture.name}",
        dt=datetime.now(),
    )

    if pictures_data.get_one(new_name):
        pictures_data.delete_one(new_name)
    pictures_data.add_one(new_pic)

    return new_pic


def apply_blur(picture: Picture) -> Picture:
    blurred = cv2.GaussianBlur(picture.img, (15, 15), sigmaX=0)

    new_name = f"{picture.name}_blur"
    new_pic = Picture(
        name=new_name,
        img=blurred,
        description=f"Gaussian blur of {picture.name}",
        dt=datetime.now(),
    )

    if pictures_data.get_one(new_name):
        pictures_data.delete_one(new_name)
    pictures_data.add_one(new_pic)

    return new_pic
