# error.py
"""Пользовательские исключения для уровней данных и сервиса."""


class Missing(Exception):
    """Запись не найдена в базе данных."""

    def __init__(self, msg: str = ""):
        self.msg = msg
        super().__init__(msg)


class Duplicate(Exception):
    """Запись с таким именем уже существует."""

    def __init__(self, msg: str = ""):
        self.msg = msg
        super().__init__(msg)
