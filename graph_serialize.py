# graph_serialize.py
"""
Deterministic serialization of an ego-graph for LLM conditioning and evaluator scoring.

Functions:
    serialize_ego_graph(G, root, depth, max_nodes, saliency_map=None, mask_topk=None)
Returns:
    serialized_text (str), token_index_map (list of (node_id, token_idx) tuples)
"""

from typing import Tuple, List, Optional, Dict
import networkx as nx
import re
from collections import deque
from .config import HYPERS
from .utils import logger

_token_re = re.compile(r"\S+")

def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer; replace or integrate a real tokenizer as needed."""
    return _token_re.findall(text)

def _clean_text(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text

def serialize_ego_graph(
    G: nx.Graph,
    root: int,
    depth: int = None,
    max_nodes: int = None,
    saliency_map: Optional[Dict[int, Dict[int, float]]] = None,
    mask_topk: Optional[int] = None
) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Serialize an ego-graph rooted at `root` deterministically.
    - BFS traversal, sorted neighbor order for deterministic behavior.
    - Each node is emitted as: NODE_<id>: <text>
    - returns (serialized_string, token_index_map) where token_index_map lists (node_id, token_pos)
      for each token in the serialized string in order (useful for masking).
    saliency_map: {node_id: {token_idx: score}} if available.
    mask_topk: if provided, mask the top-K tokens globally by saliency (useful for masked-lm metrics).
    """
    if depth is None:
        depth = HYPERS["serialization"]["bfs_depth"]
    if max_nodes is None:
        max_nodes = HYPERS["serialization"]["max_nodes"]

    visited = set([root])
    q = deque([(root, 0)])
    node_order = []
    while q and len(node_order) < max_nodes:
        n, d = q.popleft()
        node_order.append(n)
        if d < depth:
            nbrs = sorted(G.neighbors(n))
            for nb in nbrs:
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))

    # collect token lists and global saliency ranking
    tokens_by_node = {}
    token_index_map = []
    for n in node_order:
        text = G.nodes[n].get("text", "")
        text = _clean_text(text)
        tokens = _tokenize(text)[: HYPERS["serialization"]["max_tokens_per_node"]]
        tokens_by_node[n] = tokens

    # compute global saliency ordering if requested
    global_saliency = []
    if saliency_map is not None:
        for n, tok_map in saliency_map.items():
            for t_idx, score in tok_map.items():
                global_saliency.append((score, n, t_idx))
        global_saliency.sort(reverse=True, key=lambda x: x[0])

    # determine which tokens to mask (if mask_topk given)
    mask_set = set()
    if mask_topk and global_saliency:
        for _, n, t_idx in global_saliency[:mask_topk]:
            mask_set.add((n, t_idx))

    # Build serialized text
    parts = []
    for n in node_order:
        parts.append(f"NODE_{n}:")
        toks = tokens_by_node[n]
        token_strs = []
        for i, t in enumerate(toks):
            if (n, i) in mask_set:
                token_strs.append("[MASK]")
            else:
                token_strs.append(t)
            token_index_map.append((n, i))
        # join tokens with spaces
        parts.append(" ".join(token_strs) if token_strs else "")
    serialized = "\n".join(parts)
    return serialized, token_index_map
