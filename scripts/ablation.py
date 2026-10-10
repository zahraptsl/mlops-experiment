import os
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.003602704977876853
EPOCHS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)
mlflow.set_experiment("Fashion-MNIST-Ablation")


class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Fashion-MNIST tanpa augmentasi
transform = transforms.ToTensor()

dataset = datasets.FashionMNIST(
    root="data/fashion-mnist",
    train=True,
    download=False,
    transform=transform,
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# Bandingkan model dengan dropout dan tanpa dropout
experiments = [
    {"name": "ablation-with-dropout", "dropout": 0.165272059739315},
    {"name": "ablation-without-dropout", "dropout": 0.0},
]

for config in experiments:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = SimpleCNN(dropout=config["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    with mlflow.start_run(run_name=config["name"]):
        mlflow.log_params({
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dropout": config["dropout"],
            "epochs": EPOCHS,
            "augmentation": False,
            "dataset": "Fashion-MNIST",
            "architecture": "SimpleCNN",
        })

        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * images.size(0)

            train_loss = total_train_loss / len(train_dataset)

            model.eval()
            total_val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    total_val_loss += loss.item() * images.size(0)
                    predictions = outputs.argmax(dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            val_loss = total_val_loss / len(val_dataset)
            val_accuracy = correct / total

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }, step=epoch + 1)

            print(
                f"{config['name']} | Epoch {epoch + 1}/{EPOCHS} | "
                f"val_accuracy={val_accuracy:.4f}"
            )

    print(f"Eksperimen {config['name']} selesai.")

print("\nABLATION STUDY SELESAI")