# ml/train_compare.py
"""
Задание 2 самооценки: сравнение CIFAR10CNN и CIFAR10CNNv2.

Обучает обе модели на одинаковых данных за NUM_EPOCHS эпох и строит
графики val_accuracy и train_loss для сравнения архитектур.

Запуск из корня проекта:
    python -m ml.train_compare

Результат:
    ml/plots/compare_accuracy.png  — кривые точности обеих моделей
    ml/plots/compare_loss.png      — кривые потерь обеих моделей
    ml/weights/cifar10_cnn_v2.pth  — веса лучшей CIFAR10CNNv2
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
import matplotlib
matplotlib.use("Agg")  # без GUI — рисуем в файл
import matplotlib.pyplot as plt

from model.classifier import CIFAR10CNN, CIFAR10CNNv2

BATCH_SIZE = 128
NUM_EPOCHS = 10       # достаточно для наглядного сравнения кривых
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DATA_DIR = str(ROOT / "ml" / "data")
PLOTS_DIR = ROOT / "ml" / "plots"
WEIGHTS_V2 = ROOT / "ml" / "weights" / "cifar10_cnn_v2.pth"


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
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


def train_model(model, name: str, train_loader, test_loader,
                device, save_path: Path | None = None) -> dict[str, list]:
    """Обучает модель NUM_EPOCHS эпох, возвращает метрики по эпохам."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*55}")
    print(f" Модель: {name}  |  Параметров: {num_params:,}")
    print(f"{'='*55}")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc * 100)

        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc*100:.2f}%")

        if save_path and val_acc > best_acc:
            best_acc = val_acc
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    if save_path:
        print(f"  -> Лучшая точность: {best_acc*100:.2f}%, веса: {save_path}")
    return history


def plot_comparison(h1: dict, h2: dict, label1: str, label2: str) -> None:
    """Строит и сохраняет два графика: accuracy и loss."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs = range(1, NUM_EPOCHS + 1)

    # ── График точности ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, h1["val_acc"], "b-o", label=f"{label1} val_acc", linewidth=2)
    ax.plot(epochs, h2["val_acc"], "r-s", label=f"{label2} val_acc", linewidth=2)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Точность, %")
    ax.set_title("Сравнение архитектур: val accuracy на CIFAR-10")
    ax.legend()
    ax.grid(True, alpha=0.3)
    acc_path = PLOTS_DIR / "compare_accuracy.png"
    fig.savefig(acc_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] График точности: {acc_path}")

    # ── График потерь ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, h1["train_loss"], "b--", label=f"{label1} train_loss", linewidth=1.5)
    ax.plot(epochs, h1["val_loss"],   "b-",  label=f"{label1} val_loss",   linewidth=2)
    ax.plot(epochs, h2["train_loss"], "r--", label=f"{label2} train_loss", linewidth=1.5)
    ax.plot(epochs, h2["val_loss"],   "r-",  label=f"{label2} val_loss",   linewidth=2)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss (CrossEntropy)")
    ax.set_title("Сравнение архитектур: loss на CIFAR-10")
    ax.legend()
    ax.grid(True, alpha=0.3)
    loss_path = PLOTS_DIR / "compare_loss.png"
    fig.savefig(loss_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] График потерь:   {loss_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Устройство: {device}")

    train_loader, test_loader = get_dataloaders()

    # Обучаем обе модели с одинаковыми гиперпараметрами
    model_v1 = CIFAR10CNN(num_classes=10).to(device)
    h1 = train_model(model_v1, "CIFAR10CNN (v1, 3 блока)",
                     train_loader, test_loader, device)

    model_v2 = CIFAR10CNNv2(num_classes=10).to(device)
    h2 = train_model(model_v2, "CIFAR10CNNv2 (v2, 4 блока + GAP)",
                     train_loader, test_loader, device, save_path=WEIGHTS_V2)

    plot_comparison(h1, h2, "CIFAR10CNN", "CIFAR10CNNv2")

    print("\n── Итоговое сравнение ──────────────────────────────────────────")
    print(f"  CIFAR10CNN   — лучшая val_acc: {max(h1['val_acc']):.2f}%")
    print(f"  CIFAR10CNNv2 — лучшая val_acc: {max(h2['val_acc']):.2f}%")
    n_v1 = sum(p.numel() for p in CIFAR10CNN(10).parameters())
    n_v2 = sum(p.numel() for p in CIFAR10CNNv2(10).parameters())
    print(f"  CIFAR10CNN   — параметров: {n_v1:,}")
    print(f"  CIFAR10CNNv2 — параметров: {n_v2:,}")


if __name__ == "__main__":
    main()
