# ml/train_fmnist.py
"""
Задание 4 самооценки: обучение CNN на датасете Fashion-MNIST.

Fashion-MNIST — альтернатива классическому MNIST:
  - 70 000 изображений одежды (28×28, grayscale)
  - 10 классов: T-shirt, Trouser, Pullover, Dress, Coat,
                Sandal, Shirt, Sneaker, Bag, Ankle boot
  - Скачивается автоматически через torchvision (~30 MB)

Ожидаемые результаты (CPU, 10 эпох):
  - val_accuracy ≈ 91-93%

Запуск:
    python -m ml.train_fmnist

Результат:
    ml/weights/fmnist_cnn.pth  — веса лучшей модели
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from model.fmnist import FashionMNISTCNN, FMNIST_CLASSES

BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DATA_DIR = str(ROOT / "ml" / "data")
WEIGHTS_PATH = ROOT / "ml" / "weights" / "fmnist_cnn.pth"

# Fashion-MNIST: mean=0.2860, std=0.3530 (рассчитаны по обучающей выборке)
MEAN = (0.2860,)
STD  = (0.3530,)


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),             # одежда симметрична
        T.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # небольшие повороты и сдвиги
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    train_set = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += outputs.max(1)[1].eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Устройство: {device}")
    print("[i] Датасет: Fashion-MNIST (10 классов одежды)")

    train_loader, test_loader = get_dataloaders()
    model = FashionMNISTCNN(num_classes=10).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[i] Параметров в модели: {num_params:,}")
    print(f"\nКлассы: {', '.join(FMNIST_CLASSES)}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> сохранены веса (acc={best_acc*100:.2f}%)")

    print(f"\n[OK] Обучение завершено. Лучшая точность: {best_acc*100:.2f}%")
    print(f"[OK] Веса: {WEIGHTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
