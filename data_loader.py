# data_loader.py
"""
Dataset loaders for text-attributed graphs (TAGs).
- Cora: uses PyG Planetoid wrapper if available
- DBLP, BookHistory: expects edge list + node text CSV (user-provided)
Exports:
    load_dataset(name: str) -> tuple(pyg.data.Data, networkx.Graph, dict_splits)
"""

from typing import Tuple, Dict, Optional
import os
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import pandas as pd
from pathlib import Path
from .utils import logger, ensure_dir

def _nx_from_pyg(data: Data) -> nx.Graph:
    """Convert a small PyG Data object into networkx Graph with node attributes."""
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes
    G.add_nodes_from(range(num_nodes))
    for u, v in edge_index.T:
        G.add_edge(int(u), int(v))
    # attach node text placeholders if available (we expect 'text' attr)
    if hasattr(data, "text") and data.text is not None:
        for i, t in enumerate(data.text):
            G.nodes[i]["text"] = t
    else:
        for i in range(num_nodes):
            G.nodes[i]["text"] = ""
    if hasattr(data, "y"):
        for i, y in enumerate(data.y.tolist()):
            G.nodes[i]["label"] = int(y)
    return G

def load_cora(root: str = "data/cora") -> Tuple[Data, nx.Graph, Dict]:
    """Load Cora via PyG Planetoid; attach empty text fields if not present."""
    root = Path(root)
    ensure_dir(root.as_posix())
    dataset = Planetoid(root.as_posix(), name="Cora")
    data = dataset[0]
    # If dataset has no textual field, create a placeholder using node id
    if not hasattr(data, "text") or data.text is None:
        text_list = [f"Document_{i}" for i in range(data.num_nodes)]
        data.text = text_list
    G = _nx_from_pyg(data)
    # standard split (for reproducibility, we return indices)
    # PyG Planetoid provides masks; extract them if present
    splits = {}
    if hasattr(data, "train_mask"):
        splits["train"] = data.train_mask.nonzero(as_tuple=True)[0].tolist()
    if hasattr(data, "val_mask"):
        splits["val"] = data.val_mask.nonzero(as_tuple=True)[0].tolist()
    if hasattr(data, "test_mask"):
        splits["test"] = data.test_mask.nonzero(as_tuple=True)[0].tolist()
    return data, G, splits

def load_tag_from_files(edge_list_path: str, node_texts_path: str, label_col: Optional[str] = None) -> Tuple[Data, nx.Graph, Dict]:
    """
    Load a TAG from an edge list and node texts file (CSV).
    node_texts_path must contain columns: node_id, text, optionally label
    """
    edge_df = pd.read_csv(edge_list_path)
    node_df = pd.read_csv(node_texts_path)
    G = nx.Graph()
    for _, row in node_df.iterrows():
        nid = int(row["node_id"])
        G.add_node(nid)
        G.nodes[nid]["text"] = str(row["text"])
        if label_col and label_col in row:
            G.nodes[nid]["label"] = int(row[label_col])
    for _, r in edge_df.iterrows():
        u = int(r["u"]); v = int(r["v"])
        G.add_edge(u, v)
    # produce a PyG Data object with empty features (we assume textual pipeline handles tokenization)
    node_ids = sorted(G.nodes())
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    edge_index = []
    for u, v in G.edges():
        edge_index.append([id_map[u], id_map[v]])
        edge_index.append([id_map[v], id_map[u]])
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(edge_index=edge_index, num_nodes=len(node_ids))
    # store node text list in same order
    texts = [G.nodes[nid]["text"] for nid in node_ids]
    data.text = texts
    # labels if present
    if all("label" in G.nodes[nid] for nid in node_ids):
        labels = [G.nodes[nid]["label"] for nid in node_ids]
        data.y = torch.tensor(labels, dtype=torch.long)
    # create naive splits (train/val/test) if not provided
    n = data.num_nodes
    train_idx = list(range(int(0.6 * n)))
    val_idx = list(range(int(0.6 * n), int(0.8 * n)))
    test_idx = list(range(int(0.8 * n), n))
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    return data, G, splits

def load_dataset(name: str, **kwargs):
    """
    High-level loader. name in {'cora','dbpl','bookhistory'}.
    For 'dbpl' and 'bookhistory' you must supply file paths via kwargs.
    Returns (pyg_data, nx_graph, splits)
    """
    name = name.lower()
    if name == "cora":
        return load_cora(kwargs.get("root", "data/cora"))
    elif name == "dbpl" or name == "dbpl":  # allow 'dbpl'
        # expect kwargs: edge_list, node_texts
        return load_tag_from_files(kwargs["edge_list"], kwargs["node_texts"], label_col=kwargs.get("label_col"))
    elif name == "bookhistory":
        return load_tag_from_files(kwargs["edge_list"], kwargs["node_texts"], label_col=kwargs.get("label_col"))
    else:
        raise ValueError(f"Unsupported dataset name: {name}")
