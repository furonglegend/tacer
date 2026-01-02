"""
Adversarial Critic Prober for Explanation Quality
Implements Generator vs Critic minimax game for explanation generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel

try:
    from models.GNNs.sage import SAGE
except ImportError:
    # Fallback: define a simple SAGE-like module if torch_geometric not available
    class SAGE(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.1):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.Linear(hidden_channels, out_channels))
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x, edge_index=None):
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x)
                x = F.relu(x)
                x = self.dropout(x)
            return self.layers[-1](x)


class Critic(nn.Module):
    """
    Critic model that tries to reconstruct original GNN logits from explanation and subgraph.
    Used in adversarial training to push generator toward information-bottleneck optimal explanations.
    """

    def __init__(self, hidden_dim=256, n_class=7, t5_name='t5-small'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_class = n_class

        # Text encoder for explanation
        self.text_encoder = T5EncoderModel.from_pretrained(t5_name)
        self.text_proj = nn.Linear(self.text_encoder.config.d_model, hidden_dim)

        # GNN encoder for subgraph
        self.gnn = SAGE(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            dropout=0.1
        )

        # Fusion head for logit reconstruction
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_class)
        )

    def encode_text(self, expl_ids, expl_attention_mask=None):
        """Encode explanation text to embedding."""
        outputs = self.text_encoder(
            input_ids=expl_ids,
            attention_mask=expl_attention_mask
        )
        # Mean pooling
        hidden = outputs.last_hidden_state
        if expl_attention_mask is not None:
            pooled = (hidden * expl_attention_mask.unsqueeze(-1)).sum(1) / expl_attention_mask.sum(-1, keepdim=True)
        else:
            pooled = hidden.mean(dim=1)
        return self.text_proj(pooled)

    def encode_graph(self, node_features, edge_index):
        """Encode subgraph to embedding."""
        # GNN forward pass
        graph_out = self.gnn(node_features, edge_index)
        # Global mean pooling
        graph_emb = graph_out.mean(dim=0, keepdim=True)
        return graph_emb

    def forward(self, expl_ids, expl_attention_mask, subg_x, subg_edge_index):
        """
        Forward pass to reconstruct original GNN logits.

        Args:
            expl_ids: Tokenized explanation [batch, seq_len]
            expl_attention_mask: Explanation attention mask
            subg_x: Subgraph node features [n_nodes, hidden_dim]
            subg_edge_index: Subgraph edge indices [2, n_edges]

        Returns:
            Reconstructed logits [batch, n_class]
        """
        # Encode explanation
        text_emb = self.encode_text(expl_ids, expl_attention_mask)

        # Encode subgraph
        graph_emb = self.encode_graph(subg_x, subg_edge_index)

        # Expand graph embedding to match batch size
        if text_emb.size(0) > 1 and graph_emb.size(0) == 1:
            graph_emb = graph_emb.expand(text_emb.size(0), -1)

        # Fuse and predict
        fused = torch.cat([text_emb, graph_emb], dim=-1)
        logits = self.fusion_head(fused)

        return logits


class AdversarialExplainer(nn.Module):
    """
    Combined Generator-Critic system for adversarial explanation training.
    """

    def __init__(self, generator, critic, lambda_length=0.01):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.lambda_length = lambda_length

    def generator_loss(self, expl_ids, expl_attention_mask, subg_x, subg_edge_index, 
                       original_logits, explanation_length):
        """
        Compute generator loss: maximize Critic error + length penalty.

        Loss = -MSE(critic_logits, original_logits) + lambda * explanation_length
        """
        # Get Critic's reconstruction
        critic_logits = self.critic(expl_ids, expl_attention_mask, subg_x, subg_edge_index)

        # Reconstruction error (we want to MAXIMIZE this)
        recon_error = F.mse_loss(critic_logits, original_logits)

        # Length penalty (we want to MINIMIZE this)
        length_penalty = self.lambda_length * explanation_length.float().mean()

        # Generator wants to maximize error, minimize length
        # So loss = -recon_error + length_penalty
        loss = -recon_error + length_penalty

        return loss, recon_error, length_penalty

    def critic_loss(self, expl_ids, expl_attention_mask, subg_x, subg_edge_index, original_logits):
        """
        Compute critic loss: minimize reconstruction error.

        Loss = MSE(critic_logits, original_logits)
        """
        critic_logits = self.critic(expl_ids, expl_attention_mask, subg_x, subg_edge_index)
        loss = F.mse_loss(critic_logits, original_logits)
        return loss
