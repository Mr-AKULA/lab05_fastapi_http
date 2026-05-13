# model/fmnist.py
"""
Архитектура CNN для датасета Fashion-MNIST.

Fashion-MNIST (Zalando):
  - 70 000 изображений 28×28 пикселей, одноканальные (grayscale)
  - 10 классов одежды и аксессуаров
  - 60 000 обучающих / 10 000 тестовых

Классы:
  0: T-shirt/top   1: Trouser    2: Pullover  3: Dress    4: Coat
  5: Sandal        6: Shirt      7: Sneaker   8: Bag      9: Ankle boot
"""

import torch
import torch.nn as nn

FMNIST_CLASSES: tuple[str, ...] = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


class FashionMNISTCNN(nn.Module):
    """
    CNN для Fashion-MNIST: одноканальный вход 28×28, 10 классов.

    Архитектура аналогична CIFAR10CNNv2 но адаптирована для:
      - 1-канального входа (grayscale вместо RGB)
      - меньшего размера изображения (28×28 вместо 32×32)

    ┌─────────────────────────────────────────────────────────┐
    │ Блок 1: Conv(1→32,3×3) → BN → ReLU → MaxPool → Drop   │  28→14
    │ Блок 2: Conv(32→64,3×3) → BN → ReLU → MaxPool → Drop  │  14→7
    │ Блок 3: Conv(64→128,3×3) → BN → ReLU → Drop           │  7→7
    │ AdaptiveAvgPool2d(1)  → (B,128,1,1)                    │
    │ Flatten               → (B,128)                         │
    │ Linear(128→10)        → логиты                          │
    └─────────────────────────────────────────────────────────┘

    Параметров: ~130 000.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        # Третий блок без MaxPool — не уменьшаем 7×7 дальше
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)   # (B,128,H,W) → (B,128,1,1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)          # (B,1,28,28) → (B,32,14,14)
        x = self.block2(x)          # (B,32,14,14) → (B,64,7,7)
        x = self.block3(x)          # (B,64,7,7) → (B,128,7,7)
        x = self.gap(x)             # (B,128,7,7) → (B,128,1,1)
        x = torch.flatten(x, 1)     # (B,128,1,1) → (B,128)
        return self.fc(x)           # (B,128) → (B,num_classes)
