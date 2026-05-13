# ml/confusion_matrix.py
"""
Задание 5 самооценки: матрица ошибок (confusion matrix) для CIFAR10CNN.

Что делает скрипт:
  1. Загружает обученную модель из ml/weights/cifar10_cnn.pth.
  2. Прогоняет весь тестовый набор CIFAR-10 (10 000 изображений).
  3. Строит матрицу ошибок с помощью sklearn.
  4. Визуализирует её в виде тепловой карты (seaborn heatmap).
  5. Сохраняет изображение в ml/plots/confusion_matrix.png.

Запуск:
    python -m ml.confusion_matrix

Что читать на матрице:
  - Диагональ = правильно классифицированные примеры.
  - Внедиагональные клетки = ошибки: строка = истинный класс,
    столбец = предсказанный класс.
  - Например, клетка [cat, dog] показывает, сколько кошек приняли за собак.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from model.classifier import CIFAR10CNN, CIFAR10_CLASSES
from service.classifier import WEIGHTS_PATH

PLOTS_DIR = ROOT / "ml" / "plots"
DATA_DIR = str(ROOT / "ml" / "data")


@torch.no_grad()
def get_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Запускает инференс на всём датасете, возвращает (y_true, y_pred)."""
    model.eval()
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm: np.ndarray, class_names: tuple[str, ...]) -> None:
    """Рисует нормализованную тепловую карту и сохраняет в файл."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Нормализация по строкам: значение = доля правильных ответов для класса
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm,
        annot=True,          # числа в каждой клетке
        fmt=".2f",           # формат: 0.87
        cmap="Blues",        # синяя цветовая схема
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Предсказанный класс", fontsize=12)
    ax.set_ylabel("Истинный класс", fontsize=12)
    ax.set_title("Матрица ошибок CIFAR10CNN (нормализованная)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Матрица ошибок сохранена: {out_path}")


def main() -> None:
    if not WEIGHTS_PATH.exists():
        print(f"[!] Файл весов не найден: {WEIGHTS_PATH}")
        print("    Запустите обучение: python -m ml.train")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Устройство: {device}")

    # Загружаем тестовую выборку (без аугментации)
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_set, batch_size=256,
                             shuffle=False, num_workers=0)

    # Загружаем модель
    model = CIFAR10CNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device,
                                      weights_only=True))
    print("[i] Запускаем инференс на 10 000 тестовых примерах...")

    y_true, y_pred = get_predictions(model, test_loader, device)

    # Общая точность
    accuracy = (y_true == y_pred).mean() * 100
    print(f"[i] Test accuracy: {accuracy:.2f}%\n")

    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)

    # Текстовый отчёт по классам
    print("--- Classification report ---")
    print(classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES))

    # Самые частые ошибки
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # убираем диагональ
    flat_idx = np.argsort(cm_copy.ravel())[::-1][:5]
    print("--- Top-5 ошибок (истинный -> предсказанный) ---")
    for idx in flat_idx:
        true_cls = idx // len(CIFAR10_CLASSES)
        pred_cls = idx % len(CIFAR10_CLASSES)
        count = cm[true_cls, pred_cls]
        print(f"  {CIFAR10_CLASSES[true_cls]:12s} -> {CIFAR10_CLASSES[pred_cls]:12s}: {count} раз")

    plot_confusion_matrix(cm, CIFAR10_CLASSES)


if __name__ == "__main__":
    main()
