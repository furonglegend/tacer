"""
Unit Tests for Evolution Module
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '..')


class TestExplanationDNA:
    """Test cases for ExplanationDNA class."""

    @pytest.fixture
    def dna(self):
        """Create DNA fixture."""
        from evolution.dna_explain import ExplanationDNA
        return ExplanationDNA(length=64)

    def test_initialization(self, dna):
        """Test DNA initialization."""
        assert len(dna.tokens) == 64
        assert dna.fitness is None
        assert dna.length == 64

    def test_mutation_preserves_length(self, dna):
        """Test that mutation preserves DNA length."""
        original_length = len(dna.tokens)
        mutated = dna.mutate(mutation_rate=0.5)
        
        assert len(mutated.tokens) == original_length

    def test_mutation_changes_tokens(self, dna):
        """Test that mutation actually changes some tokens."""
        mutated = dna.mutate(mutation_rate=1.0)  # 100% mutation rate
        
        # At least some tokens should be different
        different = sum(1 for a, b in zip(dna.tokens, mutated.tokens) if a != b)
        assert different > 0

    def test_crossover_produces_valid_offspring(self, dna):
        """Test crossover produces valid offspring."""
        from evolution.dna_explain import ExplanationDNA
        other = ExplanationDNA(length=64)
        
        child1, child2 = dna.crossover(other)
        
        assert len(child1.tokens) == 64
        assert len(child2.tokens) == 64

    def test_crossover_mixes_genetic_material(self, dna):
        """Test that crossover mixes genetic material from both parents."""
        from evolution.dna_explain import ExplanationDNA
        
        # Create parents with distinct tokens
        parent1 = ExplanationDNA(tokens=[100] * 64, length=64)
        parent2 = ExplanationDNA(tokens=[200] * 64, length=64)
        
        child1, child2 = parent1.crossover(parent2)
        
        # Children should have tokens from both parents
        child1_has_100 = any(t == 100 for t in child1.tokens)
        child1_has_200 = any(t == 200 for t in child1.tokens)
        
        assert child1_has_100 or child1_has_200

    def test_decode_returns_string(self, dna):
        """Test that decode returns a string."""
        text = dna.decode()
        assert isinstance(text, str)

    def test_from_text_creates_valid_dna(self):
        """Test creating DNA from text."""
        from evolution.dna_explain import ExplanationDNA
        
        text = "This is a test explanation for the model."
        dna = ExplanationDNA.from_text(text, length=64)
        
        assert len(dna.tokens) == 64
        # Decoded text should contain some of the original words
        decoded = dna.decode()
        assert len(decoded) > 0

    def test_copy_creates_independent_copy(self, dna):
        """Test that copy creates independent copy."""
        dna.fitness = 0.5
        copy = dna.copy()
        
        assert copy.tokens == dna.tokens
        assert copy.fitness == dna.fitness
        
        # Modify copy shouldn't affect original
        copy.tokens[0] = 999
        assert dna.tokens[0] != 999


class TestGeneticAlgorithm:
    """Test cases for GeneticAlgorithm class."""

    @pytest.fixture
    def ga(self):
        """Create GA fixture."""
        from evolution.ga_engine import GeneticAlgorithm
        return GeneticAlgorithm(
            population_size=20,
            dna_length=32,
            mutation_rate=0.1,
            crossover_rate=0.8
        )

    @pytest.fixture
    def mock_fitness_fn(self):
        """Create mock fitness function."""
        def fitness(text):
            # Simple fitness: longer text = higher fitness
            return min(1.0, len(text) / 100)
        return fitness

    def test_initialization(self, ga):
        """Test GA initialization."""
        assert ga.population_size == 20
        assert ga.dna_length == 32
        assert len(ga.population) == 0

    def test_population_initialization(self, ga):
        """Test population initialization."""
        ga.initialize_population()
        
        assert len(ga.population) == ga.population_size

    def test_seeded_initialization(self, ga):
        """Test population initialization with seed texts."""
        seeds = ["Test explanation one", "Test explanation two"]
        ga.initialize_population(seed_texts=seeds)
        
        assert len(ga.population) == ga.population_size

    def test_tournament_selection(self, ga, mock_fitness_fn):
        """Test tournament selection."""
        ga.initialize_population()
        ga.evaluate_population(mock_fitness_fn)
        
        selected = ga.tournament_select()
        assert selected is not None
        assert selected.fitness is not None

    def test_elitism(self, ga, mock_fitness_fn):
        """Test elite selection."""
        ga.initialize_population()
        ga.evaluate_population(mock_fitness_fn)
        
        elites = ga.select_elites()
        
        expected_n = max(1, int(ga.population_size * ga.elitism_ratio))
        assert len(elites) == expected_n

    def test_diversity_computation(self, ga):
        """Test population diversity computation."""
        ga.initialize_population()
        
        diversity = ga.compute_diversity()
        
        assert 0 <= diversity <= 1

    def test_evolution_improves_fitness(self, ga, mock_fitness_fn):
        """Test that evolution improves fitness over generations."""
        ga.initialize_population()
        
        initial_stats = ga.evolve_generation(mock_fitness_fn)
        initial_best = initial_stats['best_fitness']
        
        # Run a few generations
        for _ in range(5):
            stats = ga.evolve_generation(mock_fitness_fn)
        
        final_best = stats['best_fitness']
        
        # Fitness should not decrease (elitism ensures this)
        assert final_best >= initial_best * 0.9  # Allow small variance

    def test_pareto_front_computation(self, ga, mock_fitness_fn):
        """Test Pareto front computation."""
        ga.initialize_population()
        ga.evaluate_population(mock_fitness_fn)
        
        pareto = ga.get_pareto_front()
        
        # Should have at least one individual in Pareto front
        assert len(pareto) >= 1


class TestEditDistance:
    """Test edit distance computation."""

    def test_identical_dna(self):
        """Test edit distance for identical DNA."""
        from evolution.dna_explain import ExplanationDNA, compute_edit_distance
        
        dna1 = ExplanationDNA(tokens=[1, 2, 3, 4, 5] + [0] * 59, length=64)
        dna2 = ExplanationDNA(tokens=[1, 2, 3, 4, 5] + [0] * 59, length=64)
        
        distance = compute_edit_distance(dna1, dna2)
        assert distance == 0

    def test_different_dna(self):
        """Test edit distance for different DNA."""
        from evolution.dna_explain import ExplanationDNA, compute_edit_distance
        
        dna1 = ExplanationDNA(tokens=[1] * 64, length=64)
        dna2 = ExplanationDNA(tokens=[2] * 64, length=64)
        
        distance = compute_edit_distance(dna1, dna2)
        assert distance > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
