# train.py
"""
Training loop and checkpoint utilities for GNN classifiers.

Implements:
    - train_epoch
    - evaluate
    - fit function that trains and saves best model by validation accuracy
"""

from typing import Tuple
import torch
from torch import nn, optim
from torch_geometric.data import Data
from tqdm import trange
from .utils import logger, set_seed, get_device
from .config import HYPERS, CHECKPOINT_DIR
import os

def train_epoch(model: nn.Module, data: Data, optimizer: optim.Optimizer, criterion, device: torch.device):
    model.train()
    optimizer.zero_grad()
    # Basic node classification: forward for all nodes and compute loss on train indices
    x = data.x.to(device) if hasattr(data, "x") else None
    if x is None:
        raise RuntimeError("Data object must contain node feature matrix `x` (e.g., precomputed text embeddings).")
    edge_index = data.edge_index.to(device)
    logits = model(x, edge_index)
    train_mask = data.train_mask.to(device)
    loss = criterion(logits[train_mask], data.y[train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model: nn.Module, data: Data, mask_name: str = "val", device: torch.device = None) -> Tuple[float, float]:
    if device is None:
        device = get_device()
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    logits = model(x, edge_index)
    preds = logits.argmax(dim=-1)
    if mask_name == "val":
        mask = data.val_mask
    elif mask_name == "test":
        mask = data.test_mask
    else:
        mask = data.train_mask
    mask = mask.to(device)
    correct = (preds[mask] == data.y[mask].to(device)).sum().item()
    acc = correct / mask.sum().item()
    return acc, 0.0  # second value is placeholder for additional metrics

def fit(model: nn.Module, data: Data, epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-5, save_name: str = "gnn.pt"):
    device = get_device()
    set_seed(HYPERS["seed"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val = -1.0
    best_path = os.path.join(CHECKPOINT_DIR, save_name)
    for epoch in trange(1, epochs + 1, desc="Train"):
        loss = train_epoch(model, data, optimizer, criterion, device)
        if epoch % HYPERS["trainer"]["eval_every"] == 0:
            val_acc, _ = evaluate(model, data, "val", device)
            logger.info(f"Epoch {epoch} | loss={loss:.4f} | val_acc={val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), best_path)
                logger.info(f"Saved best model to {best_path}")
    # load best
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model
