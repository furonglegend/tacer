"""
LLM interface for explanation generation and likelihood-based scoring.

This module provides a unified abstraction over:
  - local HuggingFace transformer models
  - optional external API-based LLMs

It supports:
  - deterministic decoding
  - conditional log-probability computation
  - prompt caching for reproducibility
"""

from typing import List, Dict, Optional
import torch
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import logger


class LLMInterface:
    """
    Wrapper for generation and scoring with a causal language model.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_new_tokens: int = 128,
        temperature: float = 0.0
    ):
        self.model_name = model_name
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info(f"Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._cache: Dict[str, str] = {}

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """
        Generate free-text explanation deterministically (unless temperature > 0).
        """
        key = self._hash_prompt(prompt)
        if key in self._cache:
            return self._cache[key]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = text[len(prompt):].strip()
        self._cache[key] = result
        return result

    @torch.no_grad()
    def conditional_logprob(self, context: str, target: str) -> float:
        """
        Compute log p(target | context) using teacher forcing.
        """
        full = context + target
        tokens = self.tokenizer(full, return_tensors="pt").to(self.device)
        input_ids = tokens.input_ids

        logits = self.model(input_ids).logits
        shift_logits = logits[:, :-1]
        shift_labels = input_ids[:, 1:]

        ctx_len = len(self.tokenizer(context)["input_ids"]) - 1
        log_probs = torch.log_softmax(shift_logits, dim=-1)

        score = 0.0
        for i in range(ctx_len, shift_labels.size(1)):
            token_id = shift_labels[0, i]
            score += log_probs[0, i, token_id].item()
        return score
