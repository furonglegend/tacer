"""
Robustness stress tests for explanation faithfulness.

This module implements:
  - counterfactual node removal
  - adversarial distractor injection
"""

import random
from copy import deepcopy


def counterfactual_removal(graph, target_node, k: int = 1):
    """
    Remove k neighbors of the target node to test explanation robustness.
    """
    G = deepcopy(graph)
    neighbors = list(G.neighbors(target_node))
    random.shuffle(neighbors)

    for n in neighbors[:k]:
        if G.has_node(n):
            G.remove_node(n)
    return G


def adversarial_distractors(graph, distractor_nodes):
    """
    Inject distractor nodes with irrelevant text.
    """
    G = deepcopy(graph)
    for node_id, text in distractor_nodes.items():
        G.add_node(node_id, text=text)
    return G


def robustness_suite(
    graph,
    node,
    classifier,
    explanation_fn
) -> Dict[str, bool]:
    """
    Run a standard robustness suite and report failures.
    """
    results = {}

    G_removed = counterfactual_removal(graph, node)
    results["counterfactual"] = (
        explanation_fn(G_removed) != explanation_fn(graph)
    )

    G_adv = adversarial_distractors(
        graph,
        {"adv_node": "irrelevant content unrelated to task"}
    )
    results["adversarial"] = (
        explanation_fn(G_adv) != explanation_fn(graph)
    )

    return results
