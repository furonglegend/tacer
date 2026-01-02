"""
Causal Trainer
Training loop for Causal Do-Generator model.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, Any

from models.causal.causal_do_generator import CausalDoGenerator

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


class CausalDataset(Dataset):
    """Dataset for causal intervention training."""

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

        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Create intervention sequence target
        intervention = f"do(node_{idx}.label='{label}')"
        interv_encoding = self.tokenizer(
            intervention,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': interv_encoding['input_ids'].squeeze(0),
            'class_label': torch.tensor(label, dtype=torch.long)
        }


def run_causal(args):
    """
    Main training function for Causal Do-Generator.

    Args:
        args: Command line arguments
    """
    logger.info("Starting Causal Do-Generator training...")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if args.dataset == 'cora':
        data, texts, n_class = CORA.load(args.seed)
        labels = data.y.numpy().tolist()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Initialize model
    model = CausalDoGenerator(
        t5_name='t5-small',
        hidden_dim=512,
        n_heads=8,
        n_class=n_class
    ).to(device)

    # Create dataset
    dataset = CausalDataset(texts, labels, model.tokenizer)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Training parameters
    n_epochs = 10
    lambda_length = 0.1
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = output['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / n_batches
        logger.info(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += output['loss'].item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_dir = Path('ckpts')
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / 'causal_generator.pt')
            logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")

    logger.info("Causal Do-Generator training completed!")
    return model
