import os

import mlflow
from mlflow import MlflowClient


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

EXPERIMENT_NAME = "Fashion-MNIST-CNN"
MODEL_NAME = "FashionMNIST_CNN"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise RuntimeError("Eksperimen tidak ditemukan.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.val_accuracy DESC"],
    max_results=100,
)

# Pilih run yang benar-benar memiliki artefak model
best_run = None

for run in runs:
    if "val_accuracy" not in run.data.metrics:
        continue

    try:
        client.get_run(run.info.run_id)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run.info.run_id}/model"
        )
        best_run = run
        break
    except Exception:
        continue

if best_run is None:
    raise RuntimeError(
        "Tidak ditemukan run dengan artefak model yang dapat diakses."
    )

model_uri = f"runs:/{best_run.info.run_id}/model"

print("Run terbaik:", best_run.info.run_id)
print("Validation accuracy:", best_run.data.metrics["val_accuracy"])
print("Mendaftarkan model...")

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=model_version.version,
)

print("Model version:", model_version.version)
print("Alias champion berhasil ditetapkan.")