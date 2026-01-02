"""
Graph Environment for Embodied Agent Explanations
Implements gym-compatible environment for trajectory-based explanations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        # Fallback: create minimal gym-like interface
        class spaces:
            class Discrete:
                def __init__(self, n): self.n = n
            class Box:
                def __init__(self, low, high, shape, dtype):
                    self.low, self.high, self.shape, self.dtype = low, high, shape, dtype
        
        class gym:
            class Env:
                metadata = {}
                def reset(self, **kwargs): pass
                def step(self, action): pass
                def render(self, mode='human'): pass


class GraphEnv(gym.Env):
    """
    Gym environment where an agent walks on a graph to generate explanations.
    The trajectory of visited nodes forms the explanation.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, graph: nx.Graph, embedding_dim: int = 768,
                 max_steps: int = 10, target_node: int = 0):
        """
        Initialize graph environment.

        Args:
            graph: NetworkX graph to navigate
            embedding_dim: Dimension of node embeddings
            max_steps: Maximum trajectory length
            target_node: Node to explain prediction for
        """
        super().__init__()
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.max_steps = max_steps
        self.target_node = target_node

        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        # Action space: select neighbor index (max degree in graph)
        self.max_degree = max(dict(graph.degree()).values()) if self.n_nodes > 0 else 1
        self.action_space = spaces.Discrete(self.max_degree)

        # Observation space: current node embedding + trajectory info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(embedding_dim + self.max_steps,),
            dtype=np.float32
        )

        # State variables
        self.current_node = None
        self.trajectory = []
        self.visited = set()
        self.steps = 0
        self.target_label = None
        self.saliency_scores = {}

    def _get_node_embedding(self, node_id: int) -> np.ndarray:
        """Get embedding for a node."""
        if node_id in self.graph.nodes and 'embedding' in self.graph.nodes[node_id]:
            emb = self.graph.nodes[node_id]['embedding']
            if isinstance(emb, np.ndarray):
                return emb
            return np.array(emb)
        # Return zero embedding if not available
        return np.zeros(self.embedding_dim, dtype=np.float32)

    def _get_neighbors(self, node_id: int) -> List[int]:
        """Get list of neighbor node IDs."""
        if node_id in self.graph:
            return list(self.graph.neighbors(node_id))
        return []

    def _get_observation(self) -> np.ndarray:
        """Construct observation from current state."""
        # Node embedding
        node_emb = self._get_node_embedding(self.current_node)

        # Trajectory encoding (one-hot of visited positions)
        traj_encoding = np.zeros(self.max_steps, dtype=np.float32)
        for i, node in enumerate(self.trajectory[:self.max_steps]):
            traj_encoding[i] = 1.0

        obs = np.concatenate([node_emb, traj_encoding])
        return obs.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        # Start from target node or random node
        if options and 'start_node' in options:
            self.current_node = options['start_node']
        else:
            self.current_node = self.target_node

        self.trajectory = [self.current_node]
        self.visited = {self.current_node}
        self.steps = 0

        # Get target label from node attribute
        if self.target_node in self.graph.nodes:
            self.target_label = self.graph.nodes[self.target_node].get('label', 0)
        else:
            self.target_label = 0

        # Load saliency scores if available
        for node in self.nodes:
            if node in self.graph.nodes:
                self.saliency_scores[node] = self.graph.nodes[node].get('importance_score', 0.0)

        obs = self._get_observation()
        info = {
            'trajectory': self.trajectory.copy(),
            'target_label': self.target_label
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action in environment.

        Args:
            action: Index of neighbor to move to

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        neighbors = self._get_neighbors(self.current_node)

        # Handle invalid action
        if len(neighbors) == 0:
            # No neighbors, stay in place
            reward = -0.1
            terminated = True
        elif action >= len(neighbors):
            # Invalid action index, use modulo
            action = action % len(neighbors)
            next_node = neighbors[action]
            reward = -0.05  # Small penalty for invalid action
            self.current_node = next_node
            terminated = False
        else:
            next_node = neighbors[action]
            self.current_node = next_node
            reward = 0.0
            terminated = False

        # Update trajectory
        self.trajectory.append(self.current_node)
        is_new_visit = self.current_node not in self.visited
        self.visited.add(self.current_node)
        self.steps += 1

        # Compute reward
        reward += self._compute_reward(is_new_visit)

        # Check termination
        truncated = self.steps >= self.max_steps

        obs = self._get_observation()
        info = {
            'trajectory': self.trajectory.copy(),
            'visited_count': len(self.visited),
            'current_node': self.current_node
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, is_new_visit: bool) -> float:
        """
        Compute reward for current step.

        Reward components:
        - Saliency: visiting important nodes
        - Novelty: visiting new nodes
        - Efficiency: penalty for trajectory length

        Args:
            is_new_visit: Whether current node was not previously visited

        Returns:
            Reward value
        """
        reward = 0.0

        # Saliency reward: visiting important nodes
        saliency = self.saliency_scores.get(self.current_node, 0.0)
        reward += saliency * 0.5

        # Novelty bonus for exploring new nodes
        if is_new_visit:
            reward += 0.1

        # Efficiency penalty (encourage shorter trajectories)
        reward -= 0.02 * self.steps

        # Bonus for returning to target node with good coverage
        if self.current_node == self.target_node and len(self.visited) > 1:
            coverage = len(self.visited) / min(self.n_nodes, self.max_steps)
            reward += 0.2 * coverage

        return reward

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment (placeholder)."""
        if mode == 'human':
            print(f"Step {self.steps}: Node {self.current_node}")
            print(f"Trajectory: {self.trajectory}")
            print(f"Visited: {len(self.visited)} nodes")
        return None

    def get_trajectory_explanation(self) -> str:
        """Convert trajectory to natural language explanation."""
        parts = []
        for i, node in enumerate(self.trajectory):
            if node in self.graph.nodes:
                text = self.graph.nodes[node].get('text', f'Node {node}')
                if len(text) > 50:
                    text = text[:50] + "..."
                parts.append(f"Step {i+1}: Visit '{text}'")
        return "\n".join(parts)


def create_graph_env_from_data(data, node_id: int, embedding_fn=None) -> GraphEnv:
    """
    Create GraphEnv from PyG data object.

    Args:
        data: PyTorch Geometric data object
        node_id: Target node to explain
        embedding_fn: Function to compute node embeddings

    Returns:
        GraphEnv instance
    """
    # Convert to NetworkX
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()

    # Add nodes
    n_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
    for i in range(n_nodes):
        attrs = {'label': data.y[i].item() if hasattr(data, 'y') else 0}
        if hasattr(data, 'x'):
            attrs['embedding'] = data.x[i].cpu().numpy()
        G.add_node(i, **attrs)

    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst)

    return GraphEnv(G, target_node=node_id)
