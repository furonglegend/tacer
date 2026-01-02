# models.py
"""
GNN backbones: GraphSAGE and GAT classifier wrappers suitable for TAG classification.
Provides a small wrapper class `GNNClassifier` that can be switched between architectures.
"""

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.dropout = dropout
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(max(1, num_layers))])

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GATEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, heads: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GNNClassifier(nn.Module):
    """
    Flexible classifier that accepts node features x (float tensors).
    For text-attributed graphs we typically supply a learned text embedding per node
    (done upstream, e.g., by a BERT encoder), or use simple bag-of-words embeddings.
    """
    def __init__(self, encoder: nn.Module, hidden_dim: int, num_classes: int, pooling="mean"):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, batch=None):
        """
        x: [N, F] node features
        edge_index: [2, E]
        batch: optional node->graph batch assignment (for graph-level tasks)
        For node classification batch is None and classifier operates per-node.
        """
        h = self.encoder(x, edge_index)
        logits = self.classifier(h)
        return logits

    @staticmethod
    def make(name: str, in_channels: int, hidden_dim: int, num_layers: int, num_classes: int, **kwargs):
        name = name.lower()
        if name == "graphsage":
            enc = GraphSAGEEncoder(in_channels, hidden_dim, num_layers=num_layers, dropout=kwargs.get("dropout", 0.2))
        elif name == "gat":
            enc = GATEncoder(in_channels, hidden_dim, heads=kwargs.get("heads", 4), num_layers=num_layers, dropout=kwargs.get("dropout", 0.2))
        else:
            raise ValueError(f"Unknown encoder name: {name}")
        return GNNClassifier(enc, hidden_dim, num_classes)
