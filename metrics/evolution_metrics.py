"""
Evolution Metrics
Evaluation metrics for Genetic Algorithm explanation generation.
"""

import numpy as np
from typing import List, Dict, Tuple, Any


def compute_pareto_front(population: List[Any], 
                         fitness_scores: List[Tuple[float, float]]) -> List[int]:
    """
    Identify Pareto-optimal individuals.

    Args:
        population: List of individuals
        fitness_scores: List of (faithfulness, -length) tuples

    Returns:
        Indices of Pareto-optimal individuals
    """
    n = len(fitness_scores)
    pareto_indices = []

    for i in range(n):
        dominated = False
        f1_i, f2_i = fitness_scores[i]

        for j in range(n):
            if i == j:
                continue

            f1_j, f2_j = fitness_scores[j]

            # Check if j dominates i
            if f1_j >= f1_i and f2_j >= f2_i and (f1_j > f1_i or f2_j > f2_i):
                dominated = True
                break

        if not dominated:
            pareto_indices.append(i)

    return pareto_indices


def compute_hypervolume(pareto_points: List[Tuple[float, float]], 
                        reference_point: Tuple[float, float] = (0, 0)) -> float:
    """
    Compute hypervolume indicator for Pareto front quality.

    Args:
        pareto_points: List of (faithfulness, -length) points
        reference_point: Reference point for hypervolume calculation

    Returns:
        Hypervolume value
    """
    if not pareto_points:
        return 0.0

    # Sort by first objective descending
    sorted_points = sorted(pareto_points, key=lambda x: x[0], reverse=True)

    hypervolume = 0.0
    prev_x = reference_point[0]

    for point in sorted_points:
        x, y = point
        if x > prev_x and y > reference_point[1]:
            # Rectangle contribution
            hypervolume += (x - prev_x) * (y - reference_point[1])
            prev_x = x

    return hypervolume


def compare_to_baseline(evolved_expls: List[str], baseline_expls: List[str],
                        faithfulness_fn=None) -> Dict[str, float]:
    """
    Compare evolved explanations to baseline.

    Args:
        evolved_expls: List of evolved explanations
        baseline_expls: List of baseline explanations
        faithfulness_fn: Function to compute faithfulness

    Returns:
        Dict with comparison metrics
    """
    # Length comparison
    evolved_lengths = [len(e.split()) for e in evolved_expls]
    baseline_lengths = [len(e.split()) for e in baseline_expls]

    length_improvement = (np.mean(baseline_lengths) - np.mean(evolved_lengths)) / np.mean(baseline_lengths) * 100

    # Faithfulness comparison (if function provided)
    if faithfulness_fn:
        evolved_faith = [faithfulness_fn(e) for e in evolved_expls]
        baseline_faith = [faithfulness_fn(e) for e in baseline_expls]
        faith_improvement = (np.mean(evolved_faith) - np.mean(baseline_faith)) / (np.mean(baseline_faith) + 1e-8) * 100
    else:
        evolved_faith = baseline_faith = [0]
        faith_improvement = 0

    return {
        'length_improvement_pct': length_improvement,
        'faithfulness_improvement_pct': faith_improvement,
        'evolved_avg_length': np.mean(evolved_lengths),
        'baseline_avg_length': np.mean(baseline_lengths),
        'evolved_avg_faithfulness': np.mean(evolved_faith),
        'baseline_avg_faithfulness': np.mean(baseline_faith)
    }


def analyze_evolution_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze evolution history for convergence and diversity.

    Args:
        history: List of generation statistics dicts

    Returns:
        Analysis results
    """
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    mean_fitness = [h['mean_fitness'] for h in history]
    diversity = [h['diversity'] for h in history]

    # Convergence analysis
    if len(best_fitness) > 5:
        # Check if fitness plateaued
        recent_improvement = best_fitness[-1] - best_fitness[-5]
        is_converged = abs(recent_improvement) < 0.01
    else:
        is_converged = False

    # Diversity trend
    diversity_trend = np.polyfit(range(len(diversity)), diversity, 1)[0] if len(diversity) > 1 else 0

    return {
        'total_generations': len(generations),
        'final_best_fitness': best_fitness[-1] if best_fitness else 0,
        'final_mean_fitness': mean_fitness[-1] if mean_fitness else 0,
        'final_diversity': diversity[-1] if diversity else 0,
        'fitness_improvement': best_fitness[-1] - best_fitness[0] if len(best_fitness) > 1 else 0,
        'is_converged': is_converged,
        'diversity_trend': diversity_trend
    }


def compute_selection_pressure(fitness_scores: List[float]) -> Dict[str, float]:
    """
    Analyze selection pressure in the population.

    Args:
        fitness_scores: List of fitness values

    Returns:
        Selection pressure metrics
    """
    scores = np.array(fitness_scores)

    # Selection intensity
    mean_fitness = np.mean(scores)
    std_fitness = np.std(scores)
    selection_intensity = std_fitness / (mean_fitness + 1e-8)

    # Fitness variance ratio
    top_10_pct = np.percentile(scores, 90)
    bottom_10_pct = np.percentile(scores, 10)
    fitness_range = top_10_pct - bottom_10_pct

    return {
        'selection_intensity': selection_intensity,
        'fitness_variance': std_fitness ** 2,
        'fitness_range': fitness_range,
        'top_10_percentile': top_10_pct,
        'bottom_10_percentile': bottom_10_pct
    }
