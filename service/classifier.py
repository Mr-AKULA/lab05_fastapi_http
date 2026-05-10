# service/classifier.py
"""
Сервисный слой для нейросетевой классификации изображений.

Отвечает за:
  1. Однократную загрузку обученной модели в память (паттерн Singleton).
  2. Предобработку входного изображения для инференса.
  3. Запуск прямого прохода модели и возврат top-k предсказаний.
  4. Сбор диагностической информации (устройство, кол-во параметров,
     время последнего инференса).

Паттерн Singleton критичен для FastAPI: без него модель (~2.5 MB весов)
загружалась бы заново при каждом HTTP-запросе, что увеличивает latency
на сотни миллисекунд и расходует память.
"""

import time
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from model.classifier import CIFAR10CNN, CIFAR10_CLASSES

# Путь к файлу весов модели, созданному скриптом ml/train.py
WEIGHTS_PATH: Final[Path] = Path("ml/weights/cifar10_cnn.pth")

# Устройство вычислений: GPU (если доступна CUDA) или CPU
# torch.cuda.is_available() вернёт True только при наличии NVIDIA GPU + CUDA toolkit
DEVICE: Final[torch.device] = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Конвейер предобработки изображения для инференса.
# Важно: параметры нормализации (mean, std) совпадают с теми,
# что использовались при обучении — иначе точность упадёт.
# Значения mean=(0.4914, 0.4822, 0.4465) и std=(0.2470, 0.2435, 0.2616)
# — это среднее и стандартное отклонение по каналам RGB для CIFAR-10.
_inference_tf = T.Compose([
    T.Resize((32, 32)),           # любое входное разрешение → 32×32
    T.ToTensor(),                  # PIL Image (H,W,C uint8) → Tensor (C,H,W) float [0,1]
    T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)),  # z-score нормализация по CIFAR-10
])


class Classifier:
    """
    Singleton-обёртка над обученной CNN CIFAR10CNN.

    Гарантирует, что модель загружается в память единственный раз
    за время жизни процесса FastAPI. При первом вызове Classifier()
    выполняется _init_model(); все последующие вызовы возвращают
    тот же объект из _instance.

    Пример использования:
        clf = Classifier()           # 1-й вызов: загружает модель
        clf2 = Classifier()          # 2-й вызов: возвращает тот же объект
        assert clf is clf2           # True
        results = clf.predict(img)   # BGR ndarray → список предсказаний
    """

    _instance: "Classifier | None" = None
    _last_inference_ms: float = 0.0  # время последнего вызова predict() в мс

    def __new__(cls) -> "Classifier":
        """Реализация паттерна Singleton через __new__."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self) -> None:
        """
        Загружает веса модели с диска и переводит её в режим инференса.

        Raises:
            FileNotFoundError: если файл весов не найден (обучение не проводилось).
        """
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Файл весов не найден: {WEIGHTS_PATH}. "
                f"Запустите обучение: python -m ml.train"
            )

        # Создаём экземпляр архитектуры и переносим на нужное устройство
        self.model = CIFAR10CNN(num_classes=10).to(DEVICE)

        # Загружаем только state_dict (словарь весов), не весь объект модели.
        # map_location гарантирует совместимость: модель обученная на GPU
        # корректно загрузится на CPU и наоборот.
        # weights_only=True — рекомендуемый безопасный режим (PyTorch >= 2.0)
        self.model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True),
        )

        # eval() критически важен: переключает BatchNorm на использование
        # накопленных статистик (running_mean/var) вместо статистик батча,
        # и полностью отключает Dropout (вероятность обнуления = 0)
        self.model.eval()

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Классифицирует изображение и возвращает top-k наиболее вероятных классов.

        Конвейер обработки:
          BGR ndarray (OpenCV) → RGB → PIL Image → Tensor [1,3,32,32]
          → модель → логиты [1,10] → softmax → top-k вероятности

        @torch.no_grad() отключает граф автодифференцирования:
          - экономит ~30-40% памяти (не хранятся промежуточные тензоры)
          - ускоряет вычисления (нет накладных расходов на запись в autograd)

        Args:
            img_bgr: изображение в формате BGR (H, W, 3), uint8 — стандартный
                     формат OpenCV (cv2.imdecode возвращает именно BGR).
            top_k:   сколько лучших классов вернуть (1..10).

        Returns:
            Список словарей [{"label": str, "probability": float}, ...],
            отсортированных по убыванию вероятности.
        """
        t0 = time.perf_counter()

        # OpenCV хранит каналы в порядке BGR, PIL и torchvision ожидают RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Применяем преобразования и добавляем batch-измерение: [C,H,W] → [1,C,H,W]
        tensor = _inference_tf(pil_img).unsqueeze(0).to(DEVICE)

        # Прямой проход: [1,3,32,32] → [1,10] (логиты)
        logits = self.model(tensor)

        # Softmax → вероятности; squeeze убирает batch-измерение: [1,10] → [10]
        probs = F.softmax(logits, dim=1).squeeze(0)

        # torch.topk возвращает top_k наибольших значений и их индексы
        top_probs, top_idx = torch.topk(probs, k=top_k)

        Classifier._last_inference_ms = (time.perf_counter() - t0) * 1000

        return [
            {
                "label": CIFAR10_CLASSES[int(idx)],    # индекс → название класса
                "probability": float(prob),             # Tensor → Python float
            }
            for prob, idx in zip(top_probs, top_idx)
        ]

    @staticmethod
    def get_info() -> dict:
        """
        Возвращает диагностическую информацию о сервисе классификации.

        Returns:
            Словарь с полями:
              - device: "cuda" или "cpu" — устройство вычислений
              - num_parameters: общее число обучаемых параметров модели
              - last_inference_ms: время последнего вызова predict() в миллисекундах
        """
        clf = Classifier()
        # sum(p.numel() for p in ...) подсчитывает общее число элементов
        # во всех тензорах весов и смещений модели
        num_params = sum(p.numel() for p in clf.model.parameters())
        return {
            "device": str(DEVICE),
            "num_parameters": num_params,
            "last_inference_ms": round(Classifier._last_inference_ms, 2),
        }
