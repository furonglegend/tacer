"""
Causal Do-Generator for Intervention-based Explanations
Generates causal intervention sequences as explanations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Any, Optional, Tuple


class CausalGraphAttention(nn.Module):
    """
    Attention mechanism over augmented causal DAG with mechanism variables.
    """

    def __init__(self, hidden_dim: int = 512, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply causal-graph attention.

        Args:
            query: Query tensor [batch, seq_q, hidden]
            key: Key tensor [batch, seq_k, hidden]
            value: Value tensor [batch, seq_v, hidden]
            causal_mask: Mask for causal structure [batch, seq_q, seq_k]

        Returns:
            Attended output [batch, seq_q, hidden]
        """
        batch_size = query.size(0)

        # Project
        q = self.q_proj(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask if provided
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        # Output projection and residual
        output = self.out_proj(attn_output)
        output = self.layer_norm(query + output)

        return output


class CausalDoGenerator(nn.Module):
    """
    Generator for causal intervention sequences.
    Produces do(node.attr=value) sequences as explanations.
    """

    def __init__(self, t5_name: str = 't5-small', hidden_dim: int = 512,
                 n_heads: int = 8, n_class: int = 7, max_interventions: int = 5):
        """
        Initialize causal generator.

        Args:
            t5_name: T5 model name for encoding/decoding
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            n_class: Number of classes
            max_interventions: Maximum interventions per explanation
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.max_interventions = max_interventions

        # Graph encoder
        self.encoder = T5EncoderModel.from_pretrained(t5_name)
        encoder_dim = self.encoder.config.d_model

        # Project encoder output to hidden dim
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)

        # Causal graph attention
        self.causal_attention = CausalGraphAttention(hidden_dim, n_heads)

        # Decoder for intervention sequence generation
        self.decoder = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_name)

        # Intervention predictor head
        self.intervention_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # node_id, attribute_id, value_id
        )

    def encode_graph(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode graph text to hidden representation."""
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = encoder_output.last_hidden_state
        return self.encoder_proj(hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                intervention_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Graph text token IDs
            attention_mask: Attention mask
            labels: Target intervention sequence token IDs
            intervention_labels: Target intervention tuples

        Returns:
            Dict with loss and outputs
        """
        # Encode graph
        graph_hidden = self.encode_graph(input_ids, attention_mask)

        # Apply causal attention
        causal_output = self.causal_attention(graph_hidden, graph_hidden, graph_hidden)

        # Pool for intervention prediction
        pooled = causal_output.mean(dim=1)
        intervention_logits = self.intervention_head(pooled)

        result = {
            'graph_hidden': graph_hidden,
            'causal_output': causal_output,
            'intervention_logits': intervention_logits
        }

        if labels is not None:
            # Decoder loss for intervention sequence generation
            decoder_output = self.decoder(
                inputs_embeds=causal_output,
                labels=labels
            )
            result['loss'] = decoder_output.loss
            result['decoder_logits'] = decoder_output.logits

        return result

    def generate_interventions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                max_length: int = 128) -> List[str]:
        """
        Generate intervention sequence as text.

        Args:
            input_ids: Graph text token IDs
            attention_mask: Attention mask
            max_length: Maximum output length

        Returns:
            List of intervention sequence strings
        """
        self.eval()
        with torch.no_grad():
            graph_hidden = self.encode_graph(input_ids, attention_mask)
            causal_output = self.causal_attention(graph_hidden, graph_hidden, graph_hidden)

            generated_ids = self.decoder.generate(
                inputs_embeds=causal_output,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            interventions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return interventions

    def compute_causal_loss(self, delta_y: torch.Tensor, intervention_length: torch.Tensor,
                            lambda_length: float = 0.1) -> torch.Tensor:
        """
        Compute causal fidelity loss.

        Loss = -delta_y + lambda * intervention_length

        Args:
            delta_y: Change in prediction after intervention
            intervention_length: Number of interventions
            lambda_length: Length penalty weight

        Returns:
            Loss tensor
        """
        # We want large delta_y (effective interventions) and short sequences
        loss = -delta_y.mean() + lambda_length * intervention_length.float().mean()
        return loss


class InterventionParser:
    """Parse intervention strings to structured format."""

    @staticmethod
    def parse(intervention_str: str) -> List[Dict[str, Any]]:
        """
        Parse intervention string to list of interventions.

        Args:
            intervention_str: String like "do(node_42.text='reinforcement')"

        Returns:
            List of intervention dicts
        """
        import re
        interventions = []

        # Pattern for do(node_id.attribute=value)
        pattern = r"do\((\w+)\.(\w+)\s*=\s*['\"]?([^'\"]+)['\"]?\)"
        matches = re.findall(pattern, intervention_str)

        for match in matches:
            node_id, attribute, value = match
            interventions.append({
                'node_id': node_id,
                'attribute': attribute,
                'value': value
            })

        return interventions

    @staticmethod
    def format(interventions: List[Dict[str, Any]]) -> str:
        """
        Format interventions to string.

        Args:
            interventions: List of intervention dicts

        Returns:
            Formatted string
        """
        parts = []
        for interv in interventions:
            parts.append(f"do({interv['node_id']}.{interv['attribute']}='{interv['value']}')")
        return "; ".join(parts)
