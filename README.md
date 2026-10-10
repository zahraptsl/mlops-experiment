# MLOps Experiment: Fashion-MNIST CNN dengan MLflow

Praktikum MLOps Minggu 6, Sains Data ITERA.

## Cara menjalankan
1. Aktifkan environment dan pasang dependensi: `pip install -r requirements.txt`
2. Ambil dataset: `dvc pull`
3. Jalankan MLflow server:
   `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000`
4. Jalankan skrip berurutan:
   - `python scripts/train.py` (baseline)
   - `python scripts/tune.py` (Optuna, 20 trial)
   - `python scripts/multi_seed.py` (3 seed)
   - `python scripts/ablation.py` (ablation)
   - `python scripts/register_model.py` (registry dan alias champion)
