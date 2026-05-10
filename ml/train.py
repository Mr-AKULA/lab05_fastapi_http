# ml/train.py
"""
Скрипт обучения CNN CIFAR10CNN на датасете CIFAR-10.

Запуск из корня проекта:
    python -m ml.train

Что делает скрипт:
  1. Загружает датасет CIFAR-10 (скачивает автоматически при первом запуске, ~170 MB).
  2. Применяет аугментацию к обучающей выборке (RandomCrop + HorizontalFlip).
  3. Обучает CIFAR10CNN в течение NUM_EPOCHS эпох с оптимизатором Adam.
  4. После каждой эпохи оценивает качество на тестовой выборке.
  5. Сохраняет веса лучшей модели (по val_accuracy) в ml/weights/cifar10_cnn.pth.

Ожидаемые результаты (CPU, 15 эпох, ~30-60 минут):
  - val_accuracy ≈ 80-85%
"""

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path чтобы импорты model.* работали
# при запуске как python -m ml.train из любой директории
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from model.classifier import CIFAR10CNN

# ── Гиперпараметры ────────────────────────────────────────────────────────────
BATCH_SIZE = 128      # размер мини-батча: больше → быстрее, но больше памяти
NUM_EPOCHS = 15       # число эпох; 15 достаточно для ~80% точности на CPU
LEARNING_RATE = 1e-3  # начальная скорость обучения для Adam
WEIGHT_DECAY = 1e-4   # L2-регуляризация весов (предотвращает переобучение)
DATA_DIR = str(ROOT / "ml" / "data")          # куда скачать/где искать CIFAR-10
WEIGHTS_PATH = ROOT / "ml" / "weights" / "cifar10_cnn.pth"  # куда сохранять веса


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    """
    Создаёт DataLoader'ы для обучающей и тестовой выборок CIFAR-10.

    Аугментации применяются ТОЛЬКО к обучающей выборке — это стандартная
    практика: тестовая выборка должна отражать реальные условия применения.

    Аугментации для train:
      - RandomCrop(32, padding=4): добавляет 4 пикселя границы и случайно
        вырезает фрагмент 32×32 — имитирует небольшие сдвиги объекта.
      - RandomHorizontalFlip(): отражает изображение с вероятностью 0.5 —
        удваивает эффективный размер датасета для симметричных объектов.

    Нормализация (mean, std) рассчитана по всему обучающему набору CIFAR-10
    и приводит каждый канал к нулевому среднему и единичной дисперсии.

    Returns:
        (train_loader, test_loader) — DataLoader'ы для обучения и оценки.
    """
    # Преобразования для обучающей выборки (с аугментацией)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),         # случайное кадрирование
        T.RandomHorizontalFlip(),             # горизонтальное отражение
        T.ToTensor(),                          # PIL → float Tensor [0,1]
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),  # z-score нормализация
    ])

    # Преобразования для тестовой выборки (без аугментации, только нормализация)
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])

    # torchvision.datasets.CIFAR10 скачивает данные при download=True
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_tf,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_tf,
    )

    # DataLoader: итератор по мини-батчам
    # num_workers=0 — без многопроцессорной загрузки (безопасно на Windows)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    """
    Выполняет одну эпоху обучения модели.

    Классический цикл PyTorch:
      zero_grad → forward → loss → backward → step

    1. zero_grad(): сбрасывает градиенты с предыдущего батча
       (PyTorch накапливает градиенты по умолчанию).
    2. forward: вычисляет предсказания модели.
    3. criterion: считает CrossEntropyLoss между предсказаниями и метками.
    4. backward(): вычисляет градиенты методом обратного распространения ошибки.
    5. optimizer.step(): обновляет веса в направлении антиградиента.

    Args:
        model: нейросеть (CIFAR10CNN).
        loader: DataLoader с обучающими данными.
        criterion: функция потерь (CrossEntropyLoss).
        optimizer: оптимизатор (Adam).
        device: устройство вычислений (cpu/cuda).

    Returns:
        Средняя потеря по всей эпохе (сумма потерь / число примеров).
    """
    model.train()  # включает Dropout и обновление статистик BatchNorm
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # 1. сброс градиентов
        outputs = model(images)        # 2. прямой проход → логиты
        loss = criterion(outputs, labels)  # 3. вычисление потерь
        loss.backward()                # 4. обратное распространение ошибки
        optimizer.step()               # 5. обновление весов

        # loss.item() * batch_size для корректного усреднения по всему датасету
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """
    Оценивает качество модели на тестовой (валидационной) выборке.

    @torch.no_grad() отключает граф autograd — не нужен при инференсе,
    экономит память и ускоряет вычисления.

    Args:
        model: нейросеть в режиме eval().
        loader: DataLoader с тестовыми данными.
        criterion: функция потерь.
        device: устройство вычислений.

    Returns:
        (avg_loss, accuracy) — средняя потеря и точность (0..1).
    """
    model.eval()  # выключает Dropout, BatchNorm использует running stats
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)

        # outputs.max(1) возвращает (values, indices) — берём индексы классов
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()  # число верных ответов
        total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    """
    Главная функция: настройка, обучение и сохранение модели.

    Планировщик скорости обучения StepLR:
      - Каждые step_size=5 эпох умножает lr на gamma=0.5
      - Эпохи 1-5:  lr = 1e-3
      - Эпохи 6-10: lr = 5e-4
      - Эпохи 11-15: lr = 2.5e-4
      Снижение lr в конце обучения помогает модели «дообтачиваться» в
      локальном минимуме и улучшает итоговую точность на 1-2%.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Используется устройство: {device}")

    train_loader, test_loader = get_dataloaders()

    model = CIFAR10CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()  # включает log-softmax + NLLLoss

    # Adam: адаптивный оптимизатор, хорошо работает «из коробки»
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)

    # Планировщик: снижает lr вдвое каждые 5 эпох
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()  # обновляем lr согласно расписанию

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc * 100:.2f}%")

        # Сохраняем только state_dict (словарь весов) — рекомендуемый способ.
        # Он не привязан к конкретной версии Python/PyTorch в отличие от
        # сохранения всего объекта модели через pickle.
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> сохранены веса (acc={best_acc * 100:.2f}%)")

    print(f"[OK] Обучение завершено. Лучшая точность: {best_acc * 100:.2f}%")
    print(f"[OK] Веса сохранены в: {WEIGHTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
