# ml/evaluate.py
"""
Оценка качества сохранённой модели на тесте CIFAR-10.
Запуск: python -m ml.evaluate
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from model.classifier import CIFAR10CNN
from service.classifier import WEIGHTS_PATH


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])

    test_set = torchvision.datasets.CIFAR10(
        root="./ml/data", train=False, download=True, transform=test_tf,
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    model = CIFAR10CNN(num_classes=10).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    print(f"Test loss:     {total_loss / total:.4f}")
    print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    main()
