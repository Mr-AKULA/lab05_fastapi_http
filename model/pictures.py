# model/pictures.py
"""Pydantic-модель для изображения."""

from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime


class Picture(BaseModel):
    name: str
    img: np.ndarray
    description: str = ""
    dt: datetime = Field(default_factory=datetime.now)

    model_config = {
        "arbitrary_types_allowed": True
    }
