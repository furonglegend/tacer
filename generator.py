"""
Generator model for adversarial (bottlenecked) free-text explanations.

Implements a sequence-to-sequence generator with length control.
"""

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ExplanationGenerator(nn.Module):
    """
    Generator for adversarial probing in UCert.
    """

    def __init__(self, model_name: str, max_len: int = 128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_len = max_len

    def forward(self, input_text: str) -> str:
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True
        )
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_len,
            do_sample=False
        )
        return self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
