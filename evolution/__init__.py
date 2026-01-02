"""Evolution module for genetic algorithm-based explanation generation."""

from .dna_explain import ExplanationDNA, compute_edit_distance
from .ga_engine import GeneticAlgorithm, create_mock_faithfulness_fn

__all__ = [
    'ExplanationDNA', 
    'compute_edit_distance',
    'GeneticAlgorithm',
    'create_mock_faithfulness_fn'
]
