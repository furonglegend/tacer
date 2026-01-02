"""
DNA Representation for Evolutionary Explanation Generation
Implements genetic encoding of explanations as token sequences.
"""

import random
import numpy as np
from typing import List, Tuple, Optional

try:
    from transformers import T5Tokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    T5Tokenizer = None


class ExplanationDNA:
    """
    DNA representation of an explanation as a fixed-length token sequence.
    Supports mutation, crossover, and encoding/decoding operations.
    """

    def __init__(self, tokens: Optional[List[int]] = None, length: int = 64, 
                 tokenizer_name: str = 't5-small'):
        """
        Initialize explanation DNA.

        Args:
            tokens: Initial token sequence (optional)
            length: Fixed DNA length
            tokenizer_name: Name of tokenizer for encoding/decoding
        """
        self.length = length
        self.tokenizer_name = tokenizer_name
        
        if HAS_TOKENIZER and T5Tokenizer is not None:
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
                self.vocab_size = self.tokenizer.vocab_size
            except Exception:
                self.tokenizer = None
                self.vocab_size = 32000  # Default T5 vocab size
        else:
            self.tokenizer = None
            self.vocab_size = 32000  # Default T5 vocab size

        # Get pad token id
        if self.tokenizer is not None:
            self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
        else:
            self.pad_token_id = 0

        if tokens is not None:
            self.tokens = tokens[:length] if len(tokens) > length else tokens + [self.pad_token_id] * (length - len(tokens))
        else:
            # Random initialization
            self.tokens = self._random_tokens()

        self.fitness = None

    def _random_tokens(self) -> List[int]:
        """Generate random token sequence."""
        # Avoid special tokens (usually first ~100 tokens)
        return [random.randint(100, self.vocab_size - 1) for _ in range(self.length)]

    def mutate(self, mutation_rate: float = 0.1, adaptive: bool = True) -> 'ExplanationDNA':
        """
        Apply mutation to DNA.

        Args:
            mutation_rate: Base probability of mutating each token
            adaptive: If True, adjust mutation rate based on fitness

        Returns:
            New mutated ExplanationDNA
        """
        # Adaptive mutation rate based on fitness
        if adaptive and self.fitness is not None:
            # Higher fitness = lower mutation rate
            adjusted_rate = mutation_rate * (1.0 - min(0.9, self.fitness))
        else:
            adjusted_rate = mutation_rate

        new_tokens = []
        for token in self.tokens:
            if random.random() < adjusted_rate:
                # Mutate: replace with random token
                new_tokens.append(random.randint(100, self.vocab_size - 1))
            else:
                new_tokens.append(token)

        child = ExplanationDNA(tokens=new_tokens, length=self.length)
        return child

    def crossover(self, other: 'ExplanationDNA') -> Tuple['ExplanationDNA', 'ExplanationDNA']:
        """
        Single-point crossover with another DNA.

        Args:
            other: Partner DNA for crossover

        Returns:
            Tuple of two offspring ExplanationDNA
        """
        # Select random crossover point
        crossover_point = random.randint(1, self.length - 1)

        # Create offspring
        child1_tokens = self.tokens[:crossover_point] + other.tokens[crossover_point:]
        child2_tokens = other.tokens[:crossover_point] + self.tokens[crossover_point:]

        child1 = ExplanationDNA(tokens=child1_tokens, length=self.length)
        child2 = ExplanationDNA(tokens=child2_tokens, length=self.length)

        return child1, child2

    def two_point_crossover(self, other: 'ExplanationDNA') -> Tuple['ExplanationDNA', 'ExplanationDNA']:
        """
        Two-point crossover for increased genetic diversity.

        Args:
            other: Partner DNA

        Returns:
            Tuple of two offspring
        """
        # Select two crossover points
        point1 = random.randint(1, self.length - 2)
        point2 = random.randint(point1 + 1, self.length - 1)

        child1_tokens = self.tokens[:point1] + other.tokens[point1:point2] + self.tokens[point2:]
        child2_tokens = other.tokens[:point1] + self.tokens[point1:point2] + other.tokens[point2:]

        child1 = ExplanationDNA(tokens=child1_tokens, length=self.length)
        child2 = ExplanationDNA(tokens=child2_tokens, length=self.length)

        return child1, child2

    def decode(self) -> str:
        """Decode DNA tokens to text string."""
        if self.tokenizer is not None:
            return self.tokenizer.decode(self.tokens, skip_special_tokens=True)
        else:
            # Fallback: convert tokens to simple string representation
            return " ".join([f"t{t}" for t in self.tokens if t != self.pad_token_id])

    @classmethod
    def from_text(cls, text: str, length: int = 64, tokenizer_name: str = 't5-small') -> 'ExplanationDNA':
        """
        Create DNA from text string.

        Args:
            text: Input text to encode
            length: Target DNA length
            tokenizer_name: Tokenizer to use

        Returns:
            ExplanationDNA instance
        """
        if HAS_TOKENIZER and T5Tokenizer is not None:
            try:
                tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
                tokens = tokenizer.encode(text, add_special_tokens=False)
                return cls(tokens=tokens, length=length, tokenizer_name=tokenizer_name)
            except Exception:
                pass
        
        # Fallback: use hash-based encoding
        tokens = [hash(word) % 30000 + 100 for word in text.split()]
        return cls(tokens=tokens, length=length, tokenizer_name=tokenizer_name)

    def copy(self) -> 'ExplanationDNA':
        """Create a copy of this DNA."""
        new_dna = ExplanationDNA(tokens=self.tokens.copy(), length=self.length)
        new_dna.fitness = self.fitness
        return new_dna

    def __len__(self) -> int:
        """Return effective length (non-padding tokens)."""
        count = 0
        for t in self.tokens:
            if t != self.tokenizer.pad_token_id:
                count += 1
        return count

    def __repr__(self) -> str:
        text = self.decode()
        if len(text) > 50:
            text = text[:50] + "..."
        return f"ExplanationDNA(fitness={self.fitness:.3f if self.fitness else 'None'}, text='{text}')"


def compute_edit_distance(dna1: ExplanationDNA, dna2: ExplanationDNA) -> int:
    """Compute Levenshtein edit distance between two DNA sequences."""
    tokens1 = dna1.tokens
    tokens2 = dna2.tokens
    
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]
