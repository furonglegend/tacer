"""
Unit Tests for DSL Module
"""

import pytest
import networkx as nx
import numpy as np

import sys
sys.path.insert(0, '..')


class TestDSLOperations:
    """Test cases for DSL atomic operations."""

    @pytest.fixture
    def sample_graph(self):
        """Create sample graph for testing."""
        G = nx.Graph()
        
        # Add nodes with attributes
        G.add_node(0, text="Neural networks for classification", feat=[1, 0, 0], importance_score=0.9)
        G.add_node(1, text="Deep learning methods", feat=[0.8, 0.2, 0], importance_score=0.7)
        G.add_node(2, text="Reinforcement learning agents", feat=[0, 1, 0], importance_score=0.5)
        G.add_node(3, text="Probabilistic models", feat=[0, 0, 1], importance_score=0.3)
        G.add_node(4, text="Graph neural networks", feat=[0.5, 0.5, 0], importance_score=0.8)
        
        # Add edges
        G.add_edges_from([(0, 1), (0, 2), (1, 4), (2, 3), (3, 4)])
        
        return G

    def test_hop_operation(self, sample_graph):
        """Test k-hop neighbor retrieval."""
        from dsl.spec import hop
        
        # 1-hop neighbors of node 0
        neighbors_1 = hop(sample_graph, 0, k=1)
        assert 0 in neighbors_1
        assert 1 in neighbors_1
        assert 2 in neighbors_1
        assert 3 not in neighbors_1
        
        # 2-hop neighbors of node 0
        neighbors_2 = hop(sample_graph, 0, k=2)
        assert 4 in neighbors_2
        assert 3 in neighbors_2

    def test_filter_operation(self, sample_graph):
        """Test node filtering by keyword."""
        from dsl.spec import filter_nodes
        
        all_nodes = list(sample_graph.nodes())
        
        # Filter for "neural"
        neural_nodes = filter_nodes(sample_graph, all_nodes, "neural")
        assert 0 in neural_nodes
        assert 4 in neural_nodes
        assert 2 not in neural_nodes
        
        # Filter for "learning"
        learning_nodes = filter_nodes(sample_graph, all_nodes, "learning")
        assert 1 in learning_nodes
        assert 2 in learning_nodes

    def test_agg_operation(self, sample_graph):
        """Test feature aggregation."""
        from dsl.spec import agg
        
        nodes = [0, 1, 4]
        
        # Mean aggregation
        mean_val = agg(sample_graph, nodes, func='mean')
        assert isinstance(mean_val, float)
        
        # Max aggregation
        max_val = agg(sample_graph, nodes, func='max')
        assert max_val >= mean_val
        
        # Sum aggregation
        sum_val = agg(sample_graph, nodes, func='sum')
        assert sum_val >= mean_val

    def test_select_operation(self, sample_graph):
        """Test top-k node selection."""
        from dsl.spec import select
        
        all_nodes = list(sample_graph.nodes())
        
        # Select top 2 by importance
        top_2 = select(sample_graph, all_nodes, top_k=2, criterion='importance')
        assert len(top_2) == 2
        assert 0 in top_2  # Highest importance
        assert 4 in top_2  # Second highest
        
        # Select top 3 by degree
        top_3 = select(sample_graph, all_nodes, top_k=3, criterion='degree')
        assert len(top_3) == 3

    def test_set_operations(self, sample_graph):
        """Test set operations on node lists."""
        from dsl.spec import intersect, union, difference
        
        list1 = [0, 1, 2]
        list2 = [2, 3, 4]
        
        inter = intersect(list1, list2)
        assert 2 in inter
        assert len(inter) == 1
        
        uni = union(list1, list2)
        assert len(uni) == 5
        
        diff = difference(list1, list2)
        assert 0 in diff
        assert 1 in diff
        assert 2 not in diff


class TestDSLExecutor:
    """Test cases for DSL executor."""

    @pytest.fixture
    def executor(self):
        """Create executor fixture."""
        from dsl.executor import DSLExecutor
        return DSLExecutor(timeout=5.0)

    @pytest.fixture
    def sample_graph(self):
        """Create sample graph."""
        G = nx.Graph()
        G.add_node(0, text="test node", feat=[1, 0])
        G.add_node(1, text="another node", feat=[0, 1])
        G.add_edges_from([(0, 1)])
        return G

    def test_valid_script_execution(self, executor, sample_graph):
        """Test execution of valid script."""
        script = """
neighbors = hop(G, 0, k=1)
result = count(neighbors)
"""
        result, error = executor.execute(script, sample_graph)
        assert error is None
        assert result == 2  # Node 0 and its neighbor

    def test_invalid_script_security(self, executor, sample_graph):
        """Test that dangerous scripts are blocked."""
        dangerous_scripts = [
            "import os",
            "__import__('os')",
            "eval('1+1')",
            "open('/etc/passwd')"
        ]
        
        for script in dangerous_scripts:
            result, error = executor.execute(script, sample_graph)
            assert error is not None, f"Script should be blocked: {script}"

    def test_script_timeout(self, sample_graph):
        """Test that infinite loops timeout."""
        from dsl.executor import DSLExecutor
        executor = DSLExecutor(timeout=1.0)
        
        script = """
while True:
    pass
"""
        result, error = executor.execute(script, sample_graph)
        assert error is not None
        assert "timeout" in error.lower()

    def test_syntax_validation(self):
        """Test syntax validation."""
        from dsl.executor import validate_script_syntax
        
        valid, error = validate_script_syntax("x = 1 + 2")
        assert valid
        assert error is None
        
        valid, error = validate_script_syntax("x = 1 +")
        assert not valid
        assert error is not None


class TestDSLDataGenerator:
    """Test cases for DSL data generation."""

    def test_synthetic_sample_generation(self):
        """Test synthetic sample generation."""
        # This would test dsl/data_generator.py if implemented
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
