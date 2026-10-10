import os
import random

import mlflow
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Konfigurasi eksperimen
SEED = 42
EPOCHS = 3
N_TRIALS = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Dataset Fashion-MNIST
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


# Arsitektur CNN, sama dengan baseline
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


# Pengaturan MLflow
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)
mlflow.set_experiment("Fashion-MNIST-CNN")


def objective(trial):
    # Parameter yang dicoba Optuna
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )
    batch_size = trial.suggest_categorical(
        "batch_size", [64, 128, 256]
    )
    dropout = trial.suggest_float(
        "dropout", 0.1, 0.5
    )

    # Seed sama agar perbandingan trial lebih konsisten
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = SimpleCNN(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate
    )

    with mlflow.start_run(
        run_name=f"trial-{trial.number + 1}",
        nested=True,
    ):
        mlflow.log_params({
            "trial": trial.number + 1,
            "seed": SEED,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "dropout": dropout,
            "epochs": EPOCHS,
            "architecture": "SimpleCNN",
            "optimizer": "Adam",
            "dataset": "Fashion-MNIST",
        })

        best_val_accuracy = 0.0

        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += (
                    loss.item() * images.size(0)
                )

            train_loss = (
                total_train_loss / len(train_dataset)
            )

            model.eval()
            total_val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    total_val_loss += (
                        loss.item() * images.size(0)
                    )
                    predictions = outputs.argmax(dim=1)
                    correct += (
                        predictions == labels
                    ).sum().item()
                    total += labels.size(0)

            val_loss = total_val_loss / len(val_dataset)
            val_accuracy = correct / total
            best_val_accuracy = max(
                best_val_accuracy, val_accuracy
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }, step=epoch + 1)

            print(
                f"Trial {trial.number + 1}/{N_TRIALS} | "
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"val_accuracy={val_accuracy:.4f}"
            )

            trial.report(val_accuracy, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        mlflow.log_metric(
            "best_val_accuracy", best_val_accuracy
        )

        return best_val_accuracy


# Parent run menampung 20 child runs
with mlflow.start_run(run_name="optuna-tuning"):
    mlflow.log_param("n_trials", N_TRIALS)

    study = optuna.create_study(
        direction="maximize",
        study_name="fashion-mnist-cnn-tuning",
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
    )

    print("\nTUNING SELESAI")
    print("Trial terbaik:", study.best_trial.number + 1)
    print("Best validation accuracy:", study.best_value)
    print("Parameter terbaik:", study.best_params)

    mlflow.log_params({
        f"best_{key}": value
        for key, value in study.best_params.items()
    })
    mlflow.log_metric(
        "best_val_accuracy", study.best_value
    )

    with open("best_params.txt", "w", encoding="utf-8") as file:
        file.write(
            f"Best validation accuracy: {study.best_value:.6f}\n"
        )
        file.write(f"Best parameters: {study.best_params}\n")

    mlflow.log_artifact("best_params.txt")

print("Hyperparameter tuning selesai.")