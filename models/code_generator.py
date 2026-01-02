"""
Code Generator Model for DSL Explanation Generation
Generates executable Python/DSL code from graph descriptions.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, RobertaTokenizer, T5Tokenizer
from typing import List, Optional


class CodeT5Generator(nn.Module):
    """
    CodeT5-based model for generating DSL code explanations from graphs.
    """

    def __init__(self, model_name: str = 'Salesforce/codet5-small'):
        """
        Initialize CodeT5 generator.

        Args:
            model_name: HuggingFace model name for CodeT5
        """
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # CodeT5 uses RobertaTokenizer
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        except:
            # Fallback to T5 tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        """
        Forward pass for training.

        Args:
            input_ids: Tokenized graph description
            attention_mask: Attention mask
            labels: Target DSL code tokens (for training)

        Returns:
            Model outputs with loss if labels provided
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def generate(self, graph_text: str, max_length: int = 256, 
                 num_beams: int = 4, num_return_sequences: int = 1) -> List[str]:
        """
        Generate DSL code from graph description.

        Args:
            graph_text: Text description of graph/node
            max_length: Maximum output length
            num_beams: Beam search width
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated DSL code strings
        """
        # Prepare input
        prefix = "Generate DSL code for graph explanation: "
        input_text = prefix + graph_text

        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        # Generate
        self.model.to(self.device)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode
        generated_codes = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return generated_codes

    def generate_batch(self, graph_texts: List[str], max_length: int = 256,
                       num_beams: int = 4) -> List[str]:
        """
        Generate DSL code for multiple graphs.

        Args:
            graph_texts: List of graph text descriptions
            max_length: Maximum output length
            num_beams: Beam search width

        Returns:
            List of generated DSL codes
        """
        prefix = "Generate DSL code for graph explanation: "
        input_texts = [prefix + text for text in graph_texts]

        inputs = self.tokenizer(
            input_texts,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)

        self.model.to(self.device)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class DSLCodeGenerator(nn.Module):
    """
    Specialized DSL code generator with graph-aware encoding.
    """

    def __init__(self, vocab_size: int = 32000, hidden_dim: int = 512,
                 n_layers: int = 6, n_heads: int = 8):
        """
        Initialize custom DSL generator.

        Args:
            vocab_size: Vocabulary size
            hidden_dim: Hidden dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(512, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None):
        """
        Forward pass.

        Args:
            src_ids: Source (graph description) token IDs
            tgt_ids: Target (DSL code) token IDs
            src_mask: Source attention mask
            tgt_mask: Target attention mask

        Returns:
            Output logits
        """
        batch_size, src_len = src_ids.shape
        _, tgt_len = tgt_ids.shape

        # Embeddings
        src_pos = torch.arange(src_len, device=src_ids.device).unsqueeze(0).expand(batch_size, -1)
        tgt_pos = torch.arange(tgt_len, device=tgt_ids.device).unsqueeze(0).expand(batch_size, -1)

        src_emb = self.embedding(src_ids) + self.pos_embedding(src_pos)
        tgt_emb = self.embedding(tgt_ids) + self.pos_embedding(tgt_pos)

        # Encode
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)

        # Decode
        # Create causal mask for decoder
        causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt_ids.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask)

        # Project to vocabulary
        logits = self.output_proj(output)

        return logits
