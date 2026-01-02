"""
DSL Trainer
Training loop for Code Generator (graph-to-DSL) model.
"""

import json
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Dict, Any

from models.code_generator import CodeT5Generator

try:
    from log import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class DSLDataset(Dataset):
    """Dataset for DSL code generation training."""

    def __init__(self, data_path: str, tokenizer, max_input_length=512, max_output_length=256):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Load data
        self.samples = []
        if Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    self.samples.append(sample)
        else:
            # Generate synthetic samples if file doesn't exist
            self.samples = self._generate_synthetic_samples()

    def _generate_synthetic_samples(self, n_samples=100):
        """Generate synthetic training samples."""
        samples = []
        templates = [
            {
                "input": "Graph with node 0 connected to nodes 1, 2, 3. Node 0 has text about neural networks.",
                "output": "neighbors = hop(G, 0, k=1)\nfiltered = filter(G, neighbors, 'neural')\nresult = agg(G, filtered, 'mean')"
            },
            {
                "input": "Subgraph centered at node 5 with high importance scores on nodes 5, 7, 9.",
                "output": "nodes = hop(G, 5, k=2)\nimportant = select(G, nodes, top_k=3, criterion='importance')\nresult = agg(G, important, 'sum')"
            },
            {
                "input": "Node 10 classified as Reinforcement Learning based on connections to nodes 11, 12.",
                "output": "neighbors = hop(G, 10, k=1)\nrl_nodes = filter(G, neighbors, 'reinforcement')\nresult = count(rl_nodes)"
            }
        ]

        for i in range(n_samples):
            template = templates[i % len(templates)]
            samples.append({
                "input": template["input"],
                "output": template["output"]
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Tokenize input
        input_encoding = self.tokenizer(
            "Generate DSL: " + sample["input"],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize output
        output_encoding = self.tokenizer(
            sample["output"],
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': output_encoding['input_ids'].squeeze(0)
        }


def run_dsl(args):
    """
    Main training function for DSL Code Generator.

    Args:
        args: Command line arguments
    """
    logger.info("Starting DSL Code Generator training...")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = CodeT5Generator(model_name='Salesforce/codet5-small').to(device)

    # Create dataset
    data_path = Path('dsl/dsl_dataset.jsonl')
    dataset = DSLDataset(str(data_path), model.tokenizer)

    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training parameters
    n_epochs = 10
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
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss

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

                outputs = model(input_ids, attention_mask, labels)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch} - Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_dir = Path('ckpts')
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / 'code_generator.pt')
            logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")

    logger.info("DSL Code Generator training completed!")
    return model
