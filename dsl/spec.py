"""
DSL Specification for Executable Graph Explanations
Provides atomic operations for graph querying and explanation generation.
"""

import networkx as nx
import numpy as np
from typing import List, Callable, Union, Any


def hop(G: nx.Graph, node_id: int, k: int = 1) -> List[int]:
    """
    Get k-hop neighbors of a node.

    Args:
        G: NetworkX graph
        node_id: Starting node ID
        k: Number of hops (default 1)

    Returns:
        List of node IDs within k hops
    """
    if node_id not in G:
        return []
    lengths = nx.single_source_shortest_path_length(G, node_id, cutoff=k)
    return list(lengths.keys())


def filter_nodes(G: nx.Graph, nodes: List[int], keyword: str) -> List[int]:
    """
    Filter nodes by keyword presence in their text attribute.

    Args:
        G: NetworkX graph with 'text' node attribute
        nodes: List of node IDs to filter
        keyword: Keyword to search for (case-insensitive)

    Returns:
        List of node IDs containing the keyword
    """
    result = []
    keyword_lower = keyword.lower()
    for n in nodes:
        if n in G.nodes:
            text = G.nodes[n].get('text', '')
            if keyword_lower in text.lower():
                result.append(n)
    return result


def agg(G: nx.Graph, nodes: List[int], func: str = 'mean') -> float:
    """
    Aggregate node features using specified function.

    Args:
        G: NetworkX graph with 'feat' node attribute
        nodes: List of node IDs to aggregate
        func: Aggregation function ('mean', 'max', 'sum', 'min')

    Returns:
        Aggregated feature value (scalar)
    """
    feats = []
    for n in nodes:
        if n in G.nodes and 'feat' in G.nodes[n]:
            feat = G.nodes[n]['feat']
            if isinstance(feat, (list, np.ndarray)):
                feats.append(np.array(feat).mean())
            else:
                feats.append(float(feat))

    if not feats:
        return 0.0

    feats = np.array(feats)
    if func == 'mean':
        return float(feats.mean())
    elif func == 'max':
        return float(feats.max())
    elif func == 'sum':
        return float(feats.sum())
    elif func == 'min':
        return float(feats.min())
    else:
        raise ValueError(f"Unknown aggregation function: {func}")


def select(G: nx.Graph, nodes: List[int], top_k: int, criterion: str = 'importance') -> List[int]:
    """
    Select top-k nodes based on specified criterion.

    Args:
        G: NetworkX graph
        nodes: List of node IDs to select from
        top_k: Number of nodes to select
        criterion: Selection criterion ('importance', 'degree', 'centrality')

    Returns:
        List of top-k node IDs
    """
    if not nodes:
        return []

    scores = {}
    for n in nodes:
        if n not in G:
            continue

        if criterion == 'importance':
            # Use importance_score attribute if available
            scores[n] = G.nodes[n].get('importance_score', 0.0)
        elif criterion == 'degree':
            scores[n] = G.degree(n)
        elif criterion == 'centrality':
            # Compute betweenness centrality for the subgraph
            subgraph = G.subgraph(nodes)
            centrality = nx.betweenness_centrality(subgraph)
            scores[n] = centrality.get(n, 0.0)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    # Sort by score descending and take top_k
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in sorted_nodes[:top_k]]


def classify(features: Union[List[float], np.ndarray], weights: np.ndarray) -> int:
    """
    Perform simple linear classification on features.

    Args:
        features: Feature vector
        weights: Weight matrix [n_features, n_classes]

    Returns:
        Predicted class index
    """
    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    logits = features @ weights
    return int(np.argmax(logits, axis=-1)[0])


def get_node_text(G: nx.Graph, node_id: int) -> str:
    """Get text attribute of a node."""
    if node_id in G.nodes:
        return G.nodes[node_id].get('text', '')
    return ''


def get_node_label(G: nx.Graph, node_id: int) -> Any:
    """Get label attribute of a node."""
    if node_id in G.nodes:
        return G.nodes[node_id].get('label', None)
    return None


def count_nodes(nodes: List[int]) -> int:
    """Count number of nodes in a list."""
    return len(nodes)


def intersect(nodes1: List[int], nodes2: List[int]) -> List[int]:
    """Get intersection of two node lists."""
    return list(set(nodes1) & set(nodes2))


def union(nodes1: List[int], nodes2: List[int]) -> List[int]:
    """Get union of two node lists."""
    return list(set(nodes1) | set(nodes2))


def difference(nodes1: List[int], nodes2: List[int]) -> List[int]:
    """Get nodes in nodes1 but not in nodes2."""
    return list(set(nodes1) - set(nodes2))


# Registry of all DSL operations
DSL_OPERATIONS = {
    'hop': hop,
    'filter': filter_nodes,
    'agg': agg,
    'select': select,
    'classify': classify,
    'get_node_text': get_node_text,
    'get_node_label': get_node_label,
    'count': count_nodes,
    'intersect': intersect,
    'union': union,
    'difference': difference,
}
