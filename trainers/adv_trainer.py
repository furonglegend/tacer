"""
Adversarial Trainer
Training loop for Generator vs Critic adversarial explanation optimization.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from torch_geometric.utils import k_hop_subgraph
except ImportError:
    def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False, num_nodes=None):
        """Fallback k-hop subgraph extraction."""
        import numpy as np
        visited = {node_idx}
        frontier = {node_idx}
        edge_np = edge_index.cpu().numpy()
        for _ in range(num_hops):
            new_frontier = set()
            for node in frontier:
                neighbors = edge_np[1, edge_np[0] == node].tolist()
                neighbors += edge_np[0, edge_np[1] == node].tolist()
                new_frontier.update(neighbors)
            frontier = new_frontier - visited
            visited.update(frontier)
        subset = torch.tensor(list(visited))
        return subset, edge_index, None, None

from models.adversarial.critic_prober import Critic, AdversarialExplainer

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


def extract_subgraph(edge_index, node_idx, num_hops=2, num_nodes=None):
    """
    Extract k-hop subgraph around a node.

    Args:
        edge_index: Edge index tensor [2, E]
        node_idx: Center node index
        num_hops: Number of hops
        num_nodes: Total number of nodes

    Returns:
        subset, sub_edge_index, mapping, edge_mask
    """
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes
    )
    return subset, sub_edge_index, mapping, edge_mask


def run_adv(args):
    """
    Main training function for Adversarial Probing.

    Args:
        args: Command line arguments
    """
    logger.info("Starting Adversarial Training...")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load dataset
    if args.dataset == 'cora':
        data, texts, n_class = CORA.load(args.seed)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    data = data.to(device)

    # Initialize Critic
    critic = Critic(
        hidden_dim=256,
        n_class=n_class,
        t5_name='t5-small'
    ).to(device)

    # Initialize tokenizer for explanations
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')

    # Optimizers
    lambda_length = getattr(args, 'adv_lambda', 0.01)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

    # Training parameters
    n_epochs = 10
    n_samples = min(500, len(texts))

    # Training loop
    for epoch in range(n_epochs):
        critic.train()
        total_critic_loss = 0
        n_batches = 0

        # Sample nodes for training
        sample_indices = torch.randperm(len(texts))[:n_samples]

        for idx in sample_indices:
            idx = idx.item()

            # Get node data
            text = texts[idx]
            label = data.y[idx].item()

            # Generate mock explanation (in real training, use generator)
            explanation = f"This node is classified as class {label} based on its features."

            # Tokenize explanation
            expl_encoding = tokenizer(
                explanation,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)

            # Extract subgraph
            subset, sub_edge_index, _, _ = extract_subgraph(
                data.edge_index, idx, num_hops=2, num_nodes=data.num_nodes
            )

            # Get subgraph features
            if hasattr(data, 'x') and data.x is not None:
                subg_x = data.x[subset]
            else:
                # Create simple features
                subg_x = torch.randn(len(subset), 256).to(device)

            # Get original logits (mock)
            original_logits = F.one_hot(torch.tensor([label]), n_class).float().to(device)

            # Critic forward pass
            try:
                critic_logits = critic(
                    expl_ids=expl_encoding['input_ids'],
                    expl_attention_mask=expl_encoding['attention_mask'],
                    subg_x=subg_x,
                    subg_edge_index=sub_edge_index
                )

                # Critic loss: minimize reconstruction error
                critic_loss = F.mse_loss(critic_logits, original_logits)

                # Update critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                total_critic_loss += critic_loss.item()
                n_batches += 1

            except Exception as e:
                logger.warning(f"Error processing node {idx}: {e}")
                continue

        if n_batches > 0:
            avg_critic_loss = total_critic_loss / n_batches
            logger.info(f"Epoch {epoch} - Critic Loss: {avg_critic_loss:.4f}")

    # Save model
    ckpt_dir = Path('ckpts')
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(critic.state_dict(), ckpt_dir / 'critic.pt')

    logger.info("Adversarial training completed!")
    return critic
