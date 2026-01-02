"""
TAG classifier backbone used by UCert.

This file defines a standard text-attributed graph classifier
combining a text encoder with a GNN backbone.

The model is intentionally simple to ensure interpretability
and compatibility with certificate verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import SAGEConv


class TAGClassifier(nn.Module):
    """
    Text-Attributed Graph classifier with GraphSAGE backbone.
    """

    def __init__(
        self,
        text_encoder_name: str,
        hidden_dim: int,
        num_classes: int
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.text_proj = nn.Linear(
            self.text_encoder.config.hidden_size,
            hidden_dim
        )

        self.gnn1 = SAGEConv(hidden_dim, hidden_dim)
        self.gnn2 = SAGEConv(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode_text(self, input_ids, attention_mask):
        """
        Encode node text using a pretrained language model.
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embedding = outputs.last_hidden_state[:, 0]
        return self.text_proj(cls_embedding)

    def forward(self, x_text, edge_index):
        """
        Forward pass on the graph.
        """
        h = F.relu(self.gnn1(x_text, edge_index))
        h = F.relu(self.gnn2(h, edge_index))
        return self.classifier(h)
