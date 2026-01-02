# utils.py
"""
General utilities: logging, seeding, IO helpers.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch

# Basic logger
def get_logger(name: str = __name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(ch)
    logger.setLevel(level)
    return logger

logger = get_logger("ucert")

def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # some operations are not deterministic by default; user can enable if desired:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj: Dict[Any, Any], path: str):
    ensure_dir(Path(path).parent.as_posix())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_device(preferred: str = "auto"):
    """Return torch device"""
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)
