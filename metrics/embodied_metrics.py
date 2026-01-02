"""
Embodied Agent Metrics
Evaluation metrics for trajectory-based explanations.
"""

import numpy as np
from typing import List, Dict, Any, Set
import networkx as nx


def compute_trajectory_fidelity(trajectories: List[List[int]], 
                                 graphs: List[nx.Graph],
                                 saliency_scores: List[Dict[int, float]]) -> Dict[str, float]:
    """
    Measure how well trajectories cover salient nodes.

    Args:
        trajectories: List of node ID trajectories
        graphs: Corresponding graphs
        saliency_scores: Dict mapping node IDs to importance scores

    Returns:
        Fidelity metrics
    """
    coverage_scores = []
    weighted_coverage_scores = []

    for traj, graph, scores in zip(trajectories, graphs, saliency_scores):
        visited = set(traj)

        # Basic coverage: fraction of important nodes visited
        important_nodes = [n for n, s in scores.items() if s > 0.5]
        if important_nodes:
            coverage = len(visited & set(important_nodes)) / len(important_nodes)
            coverage_scores.append(coverage)

        # Weighted coverage: sum of saliency scores of visited nodes
        total_saliency = sum(scores.values())
        visited_saliency = sum(scores.get(n, 0) for n in visited)
        weighted_cov = visited_saliency / (total_saliency + 1e-8)
        weighted_coverage_scores.append(weighted_cov)

    return {
        'mean_coverage': np.mean(coverage_scores) if coverage_scores else 0,
        'mean_weighted_coverage': np.mean(weighted_coverage_scores),
        'std_coverage': np.std(coverage_scores) if coverage_scores else 0
    }


def compute_trajectory_efficiency(trajectories: List[List[int]],
                                   graphs: List[nx.Graph]) -> Dict[str, float]:
    """
    Measure efficiency of trajectories.

    Args:
        trajectories: List of trajectories
        graphs: Corresponding graphs

    Returns:
        Efficiency metrics
    """
    lengths = []
    unique_ratios = []
    redundancy_scores = []

    for traj, graph in zip(trajectories, graphs):
        lengths.append(len(traj))

        # Unique node ratio
        unique = len(set(traj))
        unique_ratios.append(unique / len(traj) if traj else 0)

        # Redundancy: revisits / total moves
        revisits = len(traj) - unique
        redundancy_scores.append(revisits / len(traj) if traj else 0)

    return {
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'mean_unique_ratio': np.mean(unique_ratios),
        'mean_redundancy': np.mean(redundancy_scores)
    }


def compute_trajectory_coherence(trajectories: List[List[int]],
                                  graphs: List[nx.Graph]) -> Dict[str, float]:
    """
    Measure coherence of trajectories (how connected the path is).

    Args:
        trajectories: List of trajectories
        graphs: Corresponding graphs

    Returns:
        Coherence metrics
    """
    valid_transitions = []
    path_connectedness = []

    for traj, graph in zip(trajectories, graphs):
        if len(traj) < 2:
            valid_transitions.append(1.0)
            path_connectedness.append(1.0)
            continue

        # Count valid transitions (consecutive nodes are neighbors)
        valid = 0
        for i in range(len(traj) - 1):
            if graph.has_edge(traj[i], traj[i + 1]):
                valid += 1
        valid_transitions.append(valid / (len(traj) - 1))

        # Check if trajectory forms connected subgraph
        traj_nodes = list(set(traj))
        subgraph = graph.subgraph(traj_nodes)
        if len(traj_nodes) > 0:
            is_connected = nx.is_connected(subgraph) if len(subgraph) > 0 else False
            path_connectedness.append(1.0 if is_connected else 0.0)

    return {
        'mean_valid_transition_rate': np.mean(valid_transitions),
        'mean_path_connectedness': np.mean(path_connectedness)
    }


def compute_user_study_metrics(trajectory_ratings: List[float],
                                text_ratings: List[float]) -> Dict[str, Any]:
    """
    Compare user ratings between trajectory and text explanations.

    Args:
        trajectory_ratings: User ratings for trajectory explanations (1-5)
        text_ratings: User ratings for text explanations (1-5)

    Returns:
        Comparison metrics
    """
    from scipy import stats

    traj = np.array(trajectory_ratings)
    text = np.array(text_ratings)

    # Basic statistics
    traj_mean = np.mean(traj)
    text_mean = np.mean(text)

    # Paired t-test
    if len(traj) == len(text) and len(traj) > 1:
        t_stat, p_value = stats.ttest_rel(traj, text)
    else:
        t_stat, p_value = 0, 1

    # Preference rate
    prefer_traj = np.sum(traj > text)
    prefer_text = np.sum(text > traj)
    ties = np.sum(traj == text)

    return {
        'trajectory_mean_rating': traj_mean,
        'text_mean_rating': text_mean,
        'rating_difference': traj_mean - text_mean,
        't_statistic': t_stat,
        'p_value': p_value,
        'prefer_trajectory': prefer_traj,
        'prefer_text': prefer_text,
        'ties': ties,
        'trajectory_preference_rate': prefer_traj / len(traj) if len(traj) > 0 else 0
    }


def trajectory_to_explanation(trajectory: List[int], graph: nx.Graph) -> str:
    """
    Convert trajectory to natural language explanation.

    Args:
        trajectory: List of node IDs
        graph: Graph with node attributes

    Returns:
        Natural language explanation
    """
    parts = []

    for i, node in enumerate(trajectory):
        if node in graph.nodes:
            text = graph.nodes[node].get('text', f'Node {node}')
            importance = graph.nodes[node].get('importance_score', 0)

            if len(text) > 100:
                text = text[:100] + "..."

            if i == 0:
                parts.append(f"Starting from: {text}")
            elif importance > 0.5:
                parts.append(f"Key node: {text} (importance: {importance:.2f})")
            else:
                parts.append(f"Via: {text}")

    return " → ".join(parts) if parts else "Empty trajectory"
