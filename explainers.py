# explainers.py
"""
Wrappers for GNN explainers:
- GNNExplainer (from torch_geometric)
- PGExplainer (if installed)
- SubgraphX placeholder (simple greedy search)
Each wrapper returns:
    node_importance: dict[node_id -> score]
    edge_mask: dict[(u,v) -> score]
    token_importance: dict[node_id -> {token_idx:score}]  (optional)
"""

from typing import Tuple, Dict, Any
import networkx as nx
import numpy as np
from .utils import logger
import torch

try:
    from torch_geometric.nn import GNNExplainer
    _has_gnnexplainer = True
except Exception:
    _has_gnnexplainer = False

# PGExplainer optional import (may need external repo)
try:
    from pg_explainer import PGExplainer  # placeholder: actual package name may differ
    _has_pg = True
except Exception:
    _has_pg = False

def explain_with_gnnexplainer(model: torch.nn.Module, data, node_idx: int, epochs: int = 100) -> Dict:
    """
    Use PyG's GNNExplainer to produce node/edge importance.
    `data` is a PyG Data object.
    Returns a dict with keys: 'node_imp', 'edge_imp'
    """
    if not _has_gnnexplainer:
        logger.warning("GNNExplainer not available. Returning random masks as fallback.")
        return {"node_imp": {}, "edge_imp": {}}
    explainer = GNNExplainer(model, epochs=epochs)
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
    # node_feat_mask: per-feature mask; we don't return token-level by default
    edge_mask = edge_mask.detach().cpu().numpy()
    edge_index = data.edge_index.cpu().numpy().T
    edge_imp = {(int(u), int(v)): float(edge_mask[i]) for i, (u, v) in enumerate(edge_index)}
    # produce simple node importance from edge importances by summing incident edges
    node_imp = {}
    for (u, v), s in edge_imp.items():
        node_imp[u] = node_imp.get(u, 0.0) + s
        node_imp[v] = node_imp.get(v, 0.0) + s
    return {"node_imp": node_imp, "edge_imp": edge_imp}

def explain_with_pgexplainer(model: torch.nn.Module, data, node_idx: int, **kwargs) -> Dict:
    """
    Placeholder wrapper for PGExplainer. If PGExplainer is not installed,
    return a simple local-attribution estimate based on gradient magnitudes.
    """
    if _has_pg:
        # The real PGExplainer API usage depends on the installed version.
        expl = PGExplainer(model, **kwargs)
        edge_mask = expl.explain_node(node_idx, data.x, data.edge_index)
        # convert to dict similar to above
        edge_mask = edge_mask.detach().cpu().numpy()
        edge_index = data.edge_index.cpu().numpy().T
        edge_imp = {(int(u), int(v)): float(edge_mask[i]) for i, (u, v) in enumerate(edge_index)}
        node_imp = {}
        for (u, v), s in edge_imp.items():
            node_imp[u] = node_imp.get(u, 0.0) + s
            node_imp[v] = node_imp.get(v, 0.0) + s
        return {"node_imp": node_imp, "edge_imp": edge_imp}
    else:
        logger.warning("PGExplainer not available. Using gradient-saliency fallback.")
        return explain_with_gradients(model, data, node_idx)

def explain_with_gradients(model: torch.nn.Module, data, node_idx: int) -> Dict:
    """
    A lightweight gradient-based saliency that approximates token/node importance.
    This returns a node_importance map using the gradient norm of the target logit
    wrt node features.
    """
    model.eval()
    x = data.x.clone().detach().requires_grad_(True)
    edge_index = data.edge_index
    logits = model(x, edge_index)
    target = logits[node_idx, data.y[node_idx].item()] if data.y is not None else logits[node_idx].max()
    model.zero_grad()
    target.backward(retain_graph=False)
    grads = x.grad.abs().sum(dim=1).cpu().numpy()  # per-node grad norm
    node_imp = {int(i): float(grads[i]) for i in range(len(grads))}
    return {"node_imp": node_imp, "edge_imp": {}}

def explain_with_subgraphx(model: torch.nn.Module, G: nx.Graph, root: int, budget_nodes: int = 10) -> Dict:
    """
    Simplified SubgraphX-like greedy search:
      - start from root; greedily add neighbor whose embedding similarity to root is maximized
      - stop when `budget_nodes` reached
    This is a heuristic stand-in for a full SubgraphX combinatorial search.
    """
    # user must supply node embeddings; fallback to random if missing
    embeddings = nx.get_node_attributes(G, "embedding")
    if not embeddings:
        logger.warning("No node embeddings found; using random scores for subgraph selection.")
        emb = {n: np.random.randn(16) for n in G.nodes()}
    else:
        emb = embeddings
    selected = {root}
    frontier = set(G.neighbors(root))
    root_vec = emb[root]
    while len(selected) < budget_nodes and frontier:
        best_node = None
        best_sim = -1e9
        for cand in sorted(frontier):
            sim = float(np.dot(root_vec, emb[cand]) / (np.linalg.norm(root_vec) * (np.linalg.norm(emb[cand]) + 1e-12)))
            if sim > best_sim:
                best_sim = sim
                best_node = cand
        if best_node is None:
            break
        selected.add(best_node)
        frontier.update(set(G.neighbors(best_node)) - selected)
        frontier.discard(best_node)
    # construct subgraph importance masks
    node_imp = {n: (1.0 if n in selected else 0.0) for n in G.nodes()}
    edge_imp = {}
    for u, v in G.edges():
        edge_imp[(u, v)] = 1.0 if (u in selected and v in selected) else 0.0
    return {"node_imp": node_imp, "edge_imp": edge_imp}
