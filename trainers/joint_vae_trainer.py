"""
Joint VAE Trainer
Training loop for Joint VAE explanation generation model.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional, Dict, Any

from models.vae.joint_vae import JointVAE

try:
    from dataset import CORA
except ImportError:
    CORA = None

try:
    from log import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class VAEDataset(Dataset):
    """Dataset wrapper for VAE training."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # For VAE, use text as pseudo explanation target
        expl_encoding = self.tokenizer(
            f"This node is classified as class {label}.",
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'expl_ids': expl_encoding['input_ids'].squeeze(0)
        }


def prepare_vae_batch(batch, device):
    """Move batch to device."""
    return {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'labels': batch['labels'].to(device),
        'expl_ids': batch['expl_ids'].to(device)
    }


def run_joint_vae(args):
    """
    Main training function for Joint VAE.

    Args:
        args: Command line arguments with dataset, device, etc.
    """
    logger.info("Starting Joint VAE training...")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if args.dataset == 'cora':
        data, texts, n_class = CORA.load(args.seed)
        labels = data.y.numpy().tolist()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Initialize model
    model = JointVAE(
        t5_name='t5-small',
        z_dim=256,
        n_class=n_class
    ).to(device)

    # Create dataset and dataloader
    dataset = VAEDataset(texts, labels, model.tokenizer)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Training parameters
    n_epochs = 10
    beta = getattr(args, 'vae_beta', 1.0)
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = prepare_vae_batch(batch, device)

            # Forward pass
            output = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                expl_ids=batch['expl_ids'],
                beta=beta
            )

            loss = output['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / n_batches
        logger.info(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = prepare_vae_batch(batch, device)

                output = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    expl_ids=batch['expl_ids'],
                    beta=beta
                )

                val_loss += output['loss'].item()

                # Compute accuracy
                preds = output['logits'].argmax(dim=-1)
                val_correct += (preds == batch['labels']).sum().item()
                val_total += batch['labels'].size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        logger.info(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save best model
            ckpt_dir = Path('ckpts')
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / 'joint_vae.pt')
            logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    logger.info("Joint VAE training completed!")
    return model
