"""
Causal Fidelity Metrics
Evaluation metrics for causal intervention explanations.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Tuple
import networkx as nx


def compute_causal_fidelity(interventions: List[Dict[str, Any]], 
                            graphs: List[nx.Graph],
                            model_fn: Callable) -> Dict[str, float]:
    """
    Measure how well intervention predictions match counterfactual outcomes.

    Args:
        interventions: List of intervention dicts with 'node_id', 'attribute', 'value'
        graphs: List of graphs to apply interventions to
        model_fn: Model function that takes graph and returns prediction

    Returns:
        Dict with causal fidelity metrics
    """
    fidelity_scores = []
    effect_magnitudes = []

    for interv, graph in zip(interventions, graphs):
        # Get original prediction
        original_pred = model_fn(graph)

        # Apply intervention
        intervened_graph = graph.copy()
        node_id = interv.get('node_id')
        attr = interv.get('attribute')
        value = interv.get('value')

        if node_id in intervened_graph.nodes:
            intervened_graph.nodes[node_id][attr] = value

        # Get intervened prediction
        intervened_pred = model_fn(intervened_graph)

        # Compute effect
        if isinstance(original_pred, (int, float)) and isinstance(intervened_pred, (int, float)):
            effect = abs(intervened_pred - original_pred)
            effect_magnitudes.append(effect)
            fidelity_scores.append(1.0 if effect > 0 else 0.0)
        else:
            # For categorical predictions
            changed = original_pred != intervened_pred
            fidelity_scores.append(1.0 if changed else 0.0)
            effect_magnitudes.append(1.0 if changed else 0.0)

    return {
        'mean_fidelity': np.mean(fidelity_scores),
        'std_fidelity': np.std(fidelity_scores),
        'mean_effect_magnitude': np.mean(effect_magnitudes),
        'effective_intervention_rate': np.mean([1 if e > 0 else 0 for e in effect_magnitudes])
    }


def compute_intervention_minimality(interventions: List[List[Dict]], 
                                     graphs: List[nx.Graph],
                                     model_fn: Callable,
                                     target_effect: float = 0.5) -> Dict[str, float]:
    """
    Measure minimality of intervention sequences.

    Args:
        interventions: List of intervention sequences
        graphs: Corresponding graphs
        model_fn: Model function
        target_effect: Minimum effect threshold

    Returns:
        Minimality metrics
    """
    sequence_lengths = []
    sufficient_lengths = []

    for interv_seq, graph in zip(interventions, graphs):
        sequence_lengths.append(len(interv_seq))

        # Find minimum sufficient intervention
        original_pred = model_fn(graph)
        current_graph = graph.copy()

        for i, interv in enumerate(interv_seq):
            node_id = interv.get('node_id')
            attr = interv.get('attribute')
            value = interv.get('value')

            if node_id in current_graph.nodes:
                current_graph.nodes[node_id][attr] = value

            new_pred = model_fn(current_graph)

            if isinstance(original_pred, (int, float)):
                effect = abs(new_pred - original_pred)
                if effect >= target_effect:
                    sufficient_lengths.append(i + 1)
                    break
            else:
                if new_pred != original_pred:
                    sufficient_lengths.append(i + 1)
                    break
        else:
            sufficient_lengths.append(len(interv_seq))

    return {
        'mean_sequence_length': np.mean(sequence_lengths),
        'mean_sufficient_length': np.mean(sufficient_lengths),
        'minimality_ratio': np.mean(sufficient_lengths) / (np.mean(sequence_lengths) + 1e-8)
    }


def test_causal_significance(effects: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Test statistical significance of causal effects.

    Args:
        effects: List of observed causal effects
        alpha: Significance level

    Returns:
        Statistical test results
    """
    from scipy import stats

    effects = np.array(effects)
    n = len(effects)

    if n < 2:
        return {
            'significant': False,
            'p_value': 1.0,
            't_statistic': 0,
            'mean_effect': np.mean(effects) if n > 0 else 0
        }

    # One-sample t-test against zero effect
    t_stat, p_value = stats.ttest_1samp(effects, 0)

    # Effect size (Cohen's d)
    cohens_d = np.mean(effects) / (np.std(effects) + 1e-8)

    return {
        'significant': p_value < alpha,
        'p_value': p_value,
        't_statistic': t_stat,
        'mean_effect': np.mean(effects),
        'std_effect': np.std(effects),
        'cohens_d': cohens_d,
        'n_samples': n
    }


def compute_counterfactual_validity(interventions: List[Dict],
                                     graphs: List[nx.Graph],
                                     expected_changes: List[int]) -> Dict[str, float]:
    """
    Measure validity of counterfactual explanations.

    Args:
        interventions: Applied interventions
        graphs: Original graphs
        expected_changes: Expected prediction changes

    Returns:
        Validity metrics
    """
    valid_count = 0
    total = len(interventions)

    for interv, graph, expected in zip(interventions, graphs, expected_changes):
        # Check if intervention is well-formed
        if all(k in interv for k in ['node_id', 'attribute', 'value']):
            if interv['node_id'] in graph.nodes:
                valid_count += 1

    validity_rate = valid_count / total if total > 0 else 0

    return {
        'validity_rate': validity_rate,
        'valid_interventions': valid_count,
        'total_interventions': total
    }
