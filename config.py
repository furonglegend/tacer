# config.py
"""
Global configuration and hyperparameters for UCert experiments.
Adjust paths, model choices and hyperparameters here.
"""

from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
FIG_DIR = ROOT / "figures"

# Ensure directories exist (used at import time)
for p in (DATA_DIR, CHECKPOINT_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
HYPERS: Dict = {
    "seed": 42,
    "device": "cuda" if ( __import__("torch").cuda.is_available() ) else "cpu",
    "gnn": {
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-5
    },
    "trainer": {
        "epochs": 50,
        "batch_size": 128,
        "eval_every": 1
    },
    "serialization": {
        "max_nodes": 64,
        "bfs_depth": 2,
        "max_tokens_per_node": 200
    }
}

# Dataset names used in the repo
DATASETS = ["cora", "dbpl", "bookhistory"]
