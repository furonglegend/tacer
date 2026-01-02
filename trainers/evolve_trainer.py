"""
Evolution Trainer
Training loop for Genetic Algorithm-based explanation evolution.
"""

import json
import torch
from pathlib import Path
from typing import Callable, Dict, Any, List

from evolution.ga_engine import GeneticAlgorithm
from evolution.dna_explain import ExplanationDNA

try:
    from dataset import CORA
except ImportError:
    CORA = None

try:
    from filter import XAIEvaluator
except ImportError:
    XAIEvaluator = None

try:
    from log import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


def create_faithfulness_evaluator(graph, evaluator: XAIEvaluator) -> Callable[[str], float]:
    """
    Create a faithfulness evaluation function for a specific graph.

    Args:
        graph: TAG graph object
        evaluator: XAI evaluator instance

    Returns:
        Function that takes explanation text and returns faithfulness score
    """
    def evaluate_faithfulness(explanation: str) -> float:
        try:
            _, faithfulness, _ = evaluator.metrics(graph, explanation)
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (faithfulness + 5) / 10))
        except Exception as e:
            return 0.0
    return evaluate_faithfulness


def run_evolve(args):
    """
    Main function for evolutionary explanation generation.

    Args:
        args: Command line arguments
    """
    logger.info("Starting Evolutionary Explanation Generation...")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Parameters
    population_size = getattr(args, 'evolve_pop_size', 256)
    n_generations = getattr(args, 'evolve_generations', 30)

    # Load dataset for getting target keywords
    if args.dataset == 'cora':
        data, texts, n_class = CORA.load(args.seed)
        class_names = [
            'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
            'Probabilistic_Methods', 'Reinforcement_Learning',
            'Rule_Learning', 'Theory'
        ]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Create mock faithfulness function using keywords
    def keyword_faithfulness(explanation: str) -> float:
        """Simple keyword-based faithfulness approximation."""
        score = 0.0
        text_lower = explanation.lower()

        # Check for class-related keywords
        for class_name in class_names:
            if class_name.lower().replace('_', ' ') in text_lower:
                score += 0.15

        # Check for explanation quality indicators
        quality_words = ['because', 'therefore', 'indicates', 'suggests', 
                         'based on', 'due to', 'classified', 'prediction']
        for word in quality_words:
            if word in text_lower:
                score += 0.05

        return min(1.0, score)

    # Initialize GA
    ga = GeneticAlgorithm(
        population_size=population_size,
        dna_length=64,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism_ratio=0.1,
        tournament_k=3,
        lambda_length=0.01,
        diversity_threshold=0.1
    )

    # Seed population with some template explanations
    seed_texts = [
        "This node is classified based on its connections to neural network research papers.",
        "The prediction is reinforcement learning because the text mentions agents and rewards.",
        "Classification as genetic algorithms due to references to evolution and fitness.",
        "The node belongs to probabilistic methods based on Bayesian analysis keywords.",
        "Theory classification is supported by mathematical formulations in the text.",
    ]

    ga.initialize_population(seed_texts=seed_texts)

    # Evolution loop
    logger.info(f"Starting evolution with population size {population_size} for {n_generations} generations")

    best_individual = ga.evolve(
        n_generations=n_generations,
        faithfulness_fn=keyword_faithfulness,
        early_stop=True,
        verbose=True
    )

    # Get Pareto front
    pareto_front = ga.get_pareto_front()

    # Save results
    output_dir = Path('ckpts')
    output_dir.mkdir(exist_ok=True)

    # Save Pareto-optimal explanations
    pareto_results = []
    for ind in pareto_front:
        pareto_results.append({
            'text': ind.decode(),
            'fitness': ind.fitness,
            'length': len(ind)
        })

    with open(output_dir / 'pareto_explanations.json', 'w', encoding='utf-8') as f:
        json.dump(pareto_results, f, indent=2)

    # Save evolution history
    with open(output_dir / 'evolution_history.json', 'w', encoding='utf-8') as f:
        json.dump(ga.history, f, indent=2)

    logger.info(f"Evolution completed!")
    logger.info(f"Best fitness: {best_individual.fitness:.4f}")
    logger.info(f"Best explanation: {best_individual.decode()[:100]}...")
    logger.info(f"Pareto front size: {len(pareto_front)}")

    return ga
