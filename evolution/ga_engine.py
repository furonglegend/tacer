"""
Genetic Algorithm Engine for Evolving Explanations
Implements population-based optimization for explanation generation.
"""

import random
import numpy as np
from typing import List, Callable, Tuple, Optional, Dict, Any
from .dna_explain import ExplanationDNA, compute_edit_distance


class GeneticAlgorithm:
    """
    Genetic Algorithm for evolving graph explanations.
    Optimizes for faithfulness while minimizing explanation length.
    """

    def __init__(
        self,
        population_size: int = 256,
        dna_length: int = 64,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_ratio: float = 0.1,
        tournament_k: int = 3,
        lambda_length: float = 0.01,
        diversity_threshold: float = 0.1
    ):
        """
        Initialize genetic algorithm.

        Args:
            population_size: Number of individuals in population
            dna_length: Fixed length of DNA sequences
            mutation_rate: Probability of mutating each token
            crossover_rate: Probability of applying crossover
            elitism_ratio: Fraction of top individuals to preserve
            tournament_k: Tournament selection group size
            lambda_length: Weight for length penalty in fitness
            diversity_threshold: Minimum diversity before early stopping
        """
        self.population_size = population_size
        self.dna_length = dna_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.tournament_k = tournament_k
        self.lambda_length = lambda_length
        self.diversity_threshold = diversity_threshold

        self.population: List[ExplanationDNA] = []
        self.generation = 0
        self.best_individual: Optional[ExplanationDNA] = None
        self.history: List[Dict[str, Any]] = []

    def initialize_population(self, seed_texts: Optional[List[str]] = None):
        """
        Initialize random population, optionally seeded with existing texts.

        Args:
            seed_texts: Optional list of text strings to seed population
        """
        self.population = []

        # Add seeded individuals
        if seed_texts:
            for text in seed_texts[:self.population_size // 2]:
                dna = ExplanationDNA.from_text(text, length=self.dna_length)
                self.population.append(dna)

        # Fill remaining with random individuals
        while len(self.population) < self.population_size:
            dna = ExplanationDNA(length=self.dna_length)
            self.population.append(dna)

        self.generation = 0

    def compute_fitness(self, individual: ExplanationDNA, 
                        faithfulness_fn: Callable[[str], float]) -> float:
        """
        Compute fitness score for an individual.

        Fitness = faithfulness - lambda * normalized_length

        Args:
            individual: DNA to evaluate
            faithfulness_fn: Function that takes explanation text and returns faithfulness score

        Returns:
            Fitness score
        """
        text = individual.decode()
        faithfulness = faithfulness_fn(text)

        # Normalize length penalty (0-1 range)
        effective_length = len(individual)
        normalized_length = effective_length / self.dna_length

        fitness = faithfulness - self.lambda_length * normalized_length
        individual.fitness = fitness
        return fitness

    def evaluate_population(self, faithfulness_fn: Callable[[str], float]):
        """Evaluate fitness for entire population."""
        for individual in self.population:
            self.compute_fitness(individual, faithfulness_fn)

    def tournament_select(self) -> ExplanationDNA:
        """
        Select individual using tournament selection.

        Returns:
            Selected individual
        """
        tournament = random.sample(self.population, self.tournament_k)
        winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
        return winner

    def select_elites(self) -> List[ExplanationDNA]:
        """
        Select top individuals for elitism.

        Returns:
            List of elite individuals
        """
        n_elites = max(1, int(self.population_size * self.elitism_ratio))
        sorted_pop = sorted(
            self.population, 
            key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
            reverse=True
        )
        return [ind.copy() for ind in sorted_pop[:n_elites]]

    def compute_diversity(self) -> float:
        """
        Compute population diversity using average pairwise edit distance.

        Returns:
            Diversity score (0-1 range)
        """
        if len(self.population) < 2:
            return 1.0

        # Sample pairs for efficiency
        n_samples = min(50, len(self.population))
        sample = random.sample(self.population, n_samples)

        total_distance = 0
        count = 0
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                total_distance += compute_edit_distance(sample[i], sample[j])
                count += 1

        avg_distance = total_distance / count if count > 0 else 0
        # Normalize by DNA length
        diversity = avg_distance / self.dna_length
        return min(1.0, diversity)

    def evolve_generation(self, faithfulness_fn: Callable[[str], float]) -> Dict[str, Any]:
        """
        Evolve population for one generation.

        Args:
            faithfulness_fn: Fitness evaluation function

        Returns:
            Generation statistics
        """
        # Evaluate current population
        self.evaluate_population(faithfulness_fn)

        # Select elites
        elites = self.select_elites()

        # Create next generation
        next_generation = elites.copy()

        while len(next_generation) < self.population_size:
            # Tournament selection
            parent1 = self.tournament_select()
            parent2 = self.tournament_select()

            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            child1 = child1.mutate(self.mutation_rate)
            child2 = child2.mutate(self.mutation_rate)

            next_generation.append(child1)
            if len(next_generation) < self.population_size:
                next_generation.append(child2)

        self.population = next_generation
        self.generation += 1

        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness if x.fitness else float('-inf'))
        if self.best_individual is None or (current_best.fitness and 
            (self.best_individual.fitness is None or current_best.fitness > self.best_individual.fitness)):
            self.best_individual = current_best.copy()

        # Compute statistics
        fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None]
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitness_values) if fitness_values else 0,
            'mean_fitness': np.mean(fitness_values) if fitness_values else 0,
            'std_fitness': np.std(fitness_values) if fitness_values else 0,
            'diversity': self.compute_diversity(),
            'best_text': self.best_individual.decode() if self.best_individual else ""
        }
        self.history.append(stats)

        return stats

    def evolve(self, n_generations: int, faithfulness_fn: Callable[[str], float],
               early_stop: bool = True, verbose: bool = True) -> ExplanationDNA:
        """
        Run evolution for multiple generations.

        Args:
            n_generations: Maximum number of generations
            faithfulness_fn: Fitness evaluation function
            early_stop: Whether to stop early on low diversity
            verbose: Whether to print progress

        Returns:
            Best individual found
        """
        for gen in range(n_generations):
            stats = self.evolve_generation(faithfulness_fn)

            if verbose:
                print(f"Gen {stats['generation']:3d} | "
                      f"Best: {stats['best_fitness']:.4f} | "
                      f"Mean: {stats['mean_fitness']:.4f} | "
                      f"Div: {stats['diversity']:.4f}")

            # Early stopping on low diversity
            if early_stop and stats['diversity'] < self.diversity_threshold:
                if verbose:
                    print(f"Early stopping: diversity {stats['diversity']:.4f} < {self.diversity_threshold}")
                break

        return self.best_individual

    def get_pareto_front(self) -> List[ExplanationDNA]:
        """
        Get Pareto-optimal individuals (faithfulness vs length tradeoff).

        Returns:
            List of non-dominated individuals
        """
        pareto_front = []

        for ind in self.population:
            if ind.fitness is None:
                continue

            dominated = False
            length = len(ind)

            for other in self.population:
                if other.fitness is None or other is ind:
                    continue

                other_length = len(other)

                # Check if other dominates ind
                if (other.fitness >= ind.fitness and other_length <= length and
                    (other.fitness > ind.fitness or other_length < length)):
                    dominated = True
                    break

            if not dominated:
                pareto_front.append(ind)

        return pareto_front


def create_mock_faithfulness_fn(target_keywords: List[str]) -> Callable[[str], float]:
    """
    Create a mock faithfulness function for testing.

    Args:
        target_keywords: Keywords that increase faithfulness score

    Returns:
        Faithfulness evaluation function
    """
    def faithfulness_fn(text: str) -> float:
        text_lower = text.lower()
        score = 0.0
        for keyword in target_keywords:
            if keyword.lower() in text_lower:
                score += 0.2
        return min(1.0, score)
    return faithfulness_fn
