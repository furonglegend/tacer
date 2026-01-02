"""
Causal do-Calculus DSL for Intervention-based Explanations
Provides atomic operations for causal interventions on graphs.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np


class Intervention:
    """Represents a single causal intervention."""

    def __init__(self, intervention_id: int, node_id: int, attribute: str, 
                 old_value: Any, new_value: Any):
        self.id = intervention_id
        self.node_id = node_id
        self.attribute = attribute
        self.old_value = old_value
        self.new_value = new_value
        self.applied = False

    def __repr__(self):
        return f"do({self.node_id}.{self.attribute} = {self.new_value})"


class InterventionManager:
    """
    Manages causal interventions on a graph.
    Tracks intervention history and supports rollback.
    """

    def __init__(self, graph: nx.Graph):
        """
        Initialize intervention manager.

        Args:
            graph: Original graph to apply interventions to
        """
        self.original_graph = graph
        self.working_graph = copy.deepcopy(graph)
        self.interventions: List[Intervention] = []
        self.intervention_counter = 0

    def do(self, node_id: int, attribute: str, value: Any) -> int:
        """
        Apply a do-intervention: set node attribute to fixed value.

        Args:
            node_id: Target node ID
            attribute: Attribute name to intervene on
            value: New value to set

        Returns:
            Intervention ID for reference
        """
        if node_id not in self.working_graph.nodes:
            raise ValueError(f"Node {node_id} not in graph")

        # Store old value
        old_value = self.working_graph.nodes[node_id].get(attribute, None)

        # Create intervention record
        intervention = Intervention(
            intervention_id=self.intervention_counter,
            node_id=node_id,
            attribute=attribute,
            old_value=old_value,
            new_value=value
        )

        # Apply intervention
        self.working_graph.nodes[node_id][attribute] = value
        intervention.applied = True
        self.interventions.append(intervention)
        self.intervention_counter += 1

        return intervention.id

    def see(self, node_id: int, attribute: str) -> Any:
        """
        Observe the current value of a node attribute.

        Args:
            node_id: Target node ID
            attribute: Attribute name to observe

        Returns:
            Current attribute value
        """
        if node_id not in self.working_graph.nodes:
            raise ValueError(f"Node {node_id} not in graph")
        return self.working_graph.nodes[node_id].get(attribute, None)

    def undo(self, intervention_id: int) -> bool:
        """
        Reverse a previous intervention.

        Args:
            intervention_id: ID of intervention to undo

        Returns:
            True if successful, False otherwise
        """
        for i, intervention in enumerate(self.interventions):
            if intervention.id == intervention_id and intervention.applied:
                # Restore old value
                self.working_graph.nodes[intervention.node_id][intervention.attribute] = intervention.old_value
                intervention.applied = False
                return True
        return False

    def undo_all(self):
        """Undo all interventions and reset to original graph."""
        self.working_graph = copy.deepcopy(self.original_graph)
        for intervention in self.interventions:
            intervention.applied = False

    def get_intervention_history(self) -> List[str]:
        """Get human-readable intervention history."""
        return [str(i) for i in self.interventions if i.applied]

    def get_current_graph(self) -> nx.Graph:
        """Get current graph with all applied interventions."""
        return self.working_graph


def do(graph: nx.Graph, node_id: int, attribute: str, value: Any) -> nx.Graph:
    """
    Standalone do-intervention function.

    Args:
        graph: Input graph
        node_id: Target node ID
        attribute: Attribute to intervene on
        value: New value

    Returns:
        Modified graph copy
    """
    new_graph = copy.deepcopy(graph)
    if node_id in new_graph.nodes:
        new_graph.nodes[node_id][attribute] = value
    return new_graph


def see(graph: nx.Graph, node_id: int, attribute: str) -> Any:
    """
    Observe a node attribute value.

    Args:
        graph: Input graph
        node_id: Target node ID
        attribute: Attribute to observe

    Returns:
        Attribute value
    """
    if node_id in graph.nodes:
        return graph.nodes[node_id].get(attribute, None)
    return None


def compute_causal_effect(graph: nx.Graph, intervention: Dict[str, Any], 
                          model_fn, target_node: int = 0) -> Tuple[float, Any, Any]:
    """
    Compute causal effect of an intervention.

    Args:
        graph: Original graph
        intervention: Dict with keys 'node_id', 'attribute', 'value'
        model_fn: Function that takes graph and returns prediction
        target_node: Node to compute prediction for

    Returns:
        Tuple of (delta_y, y_original, y_intervened)
    """
    # Get original prediction
    y_original = model_fn(graph, target_node)

    # Apply intervention
    intervened_graph = do(
        graph, 
        intervention['node_id'], 
        intervention['attribute'], 
        intervention['value']
    )

    # Get intervened prediction
    y_intervened = model_fn(intervened_graph, target_node)

    # Compute effect
    if isinstance(y_original, (int, float)) and isinstance(y_intervened, (int, float)):
        delta_y = float(y_intervened - y_original)
    elif hasattr(y_original, '__sub__'):
        delta_y = y_intervened - y_original
    else:
        # For categorical predictions, return indicator of change
        delta_y = 0.0 if y_original == y_intervened else 1.0

    return delta_y, y_original, y_intervened


def generate_intervention_sequence(important_nodes: List[int], 
                                    attributes: List[str],
                                    n_interventions: int = 3) -> List[Dict[str, Any]]:
    """
    Generate a sequence of interventions for explanation.

    Args:
        important_nodes: List of important node IDs (ranked by importance)
        attributes: List of possible attributes to intervene on
        n_interventions: Number of interventions to generate

    Returns:
        List of intervention dictionaries
    """
    interventions = []
    for i in range(min(n_interventions, len(important_nodes))):
        for attr in attributes[:1]:  # Use most important attribute
            interventions.append({
                'node_id': important_nodes[i],
                'attribute': attr,
                'value': None  # Will be set based on counterfactual logic
            })
    return interventions


# DSL operation registry for causal operations
CAUSAL_OPERATIONS = {
    'do': do,
    'see': see,
    'compute_causal_effect': compute_causal_effect,
}
