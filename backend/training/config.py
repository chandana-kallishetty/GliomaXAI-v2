"""
config.py
=========
Configuration parameters for the GliomaXAI training pipeline.
Supports BraTS dataset integration and versioned models.
"""
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "brats"
MODELS_DIR = BASE_DIR / "ml_models"

# Dataset settings
MODALITIES = ["t1", "t1ce", "t2", "flair"]
TARGET_SHAPE = (224, 224, 4)  # 4 modalities stacked
BATCH_SIZE = 16

# Training hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Model Versioning Registry
MODEL_REGISTRY = {
    "v1_baseline": {"desc": "Baseline CNN", "dataset": "Kaggle Generic"},
    "v2_brats": {"desc": "BraTS Multi-modal", "dataset": "BraTS 2021"}
}
