# ml/train_resnet.py
"""
Задание 3 самооценки: Transfer Learning на основе ResNet18.

Стратегия:
  1. Загружаем ResNet18 с предобученными весами ImageNet.
  2. Замораживаем все слои, кроме последнего residual-блока (layer4) и fc.
  3. Заменяем выходной слой на Linear(512 → 10) для CIFAR-10.
  4. Дообучаем 5 эпох с Adam — только незамороженные параметры.

Почему замораживаем всё кроме layer4:
  - Первые слои ResNet18 (layer1-layer3) извлекают универсальные признаки
    (края, текстуры), которые одинаково полезны для любого датасета.
  - layer4 и fc адаптируются под конкретную задачу (CIFAR-10 vs ImageNet).
  - Заморозка ≈ 86% параметров ускоряет обучение в 5-7 раз.

Запуск:
    python -m ml.train_resnet

Результат:
    ml/weights/resnet18_cifar10.pth — fine-tuned веса
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
import torchvision.models as models

BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DATA_DIR = str(ROOT / "ml" / "data")
WEIGHTS_PATH = ROOT / "ml" / "weights" / "resnet18_cifar10.pth"

# ResNet18 обучался на ImageNet (224×224), поэтому увеличиваем CIFAR-10 (32×32)
# Нормализация — средние и std от ImageNet, с которым обучался ResNet18
MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET  = (0.229, 0.224, 0.225)


def get_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_tf = T.Compose([
        T.Resize(64),                          # CIFAR-10 32→64: компромисс скорость/качество
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
    ])
    test_tf = T.Compose([
        T.Resize(64),
        T.ToTensor(),
        T.Normalize(MEAN_IMAGENET, STD_IMAGENET),
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


def build_model() -> nn.Module:
    """
    Строит модель для transfer learning.

    ResNet18 состоит из:
      conv1 → bn1 → relu → maxpool → layer1 → layer2 → layer3 → layer4 → avgpool → fc

    Замораживаем: conv1, bn1, layer1, layer2, layer3.
    Дообучаем:    layer4, fc (заменяем на Linear(512→10)).
    """
    # weights='IMAGENET1K_V1': предобученные веса с точностью 69.8% на ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Замораживаем все параметры
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем последний residual-блок
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Заменяем голову на 10-классовую
    model.fc = nn.Linear(model.fc.in_features, 10)
    # fc создаётся с requires_grad=True по умолчанию

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Заморожено параметров:   {frozen:,}")
    print(f"  Обучаемых параметров:    {trainable:,}")
    return model


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
    print("[i] Загружаем ResNet18 (ImageNet)...")

    train_loader, test_loader = get_dataloaders()
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    # Оптимизируем только незамороженные параметры
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    best_acc = 0.0
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f" Fine-tuning ResNet18 → CIFAR-10 ({NUM_EPOCHS} эпох)")
    print(f"{'='*55}")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> сохранены веса (acc={best_acc*100:.2f}%)")

    print(f"\n[OK] Fine-tuning завершён. Лучшая точность: {best_acc*100:.2f}%")
    print(f"[OK] Веса: {WEIGHTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
