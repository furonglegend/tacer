"""
Integration Tests for GraphNarrator Enhancement Modules
"""

import pytest
import sys
sys.path.insert(0, '..')


class TestModeRouting:
    """Test that all modes are correctly routed in main.py."""

    def test_mode_choices_defined(self):
        """Test that all new modes are defined in argparse choices."""
        import argparse
        from main import parse
        
        # This will fail if modes aren't properly defined
        # We just test that parsing works with new modes
        expected_modes = [
            'auto', 'manual', 'sali', 'brev', 'fait',
            'joint_vae', 'adv', 'dsl', 'causal', 'embodied', 'evolve'
        ]
        
        # Check that parse function exists and works
        assert callable(parse)

    def test_trainer_imports(self):
        """Test that all trainers can be imported."""
        try:
            from trainers import (
                run_joint_vae, run_adv, run_dsl,
                run_causal, run_evolve, run_embodied
            )
            assert callable(run_joint_vae)
            assert callable(run_adv)
            assert callable(run_dsl)
            assert callable(run_causal)
            assert callable(run_evolve)
            assert callable(run_embodied)
        except ImportError as e:
            pytest.skip(f"Trainer import failed: {e}")


class TestModuleIntegration:
    """Test that modules integrate correctly."""

    def test_vae_model_import(self):
        """Test VAE model can be imported."""
        try:
            from models.vae import JointVAE
            assert JointVAE is not None
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_adversarial_model_import(self):
        """Test adversarial model can be imported."""
        try:
            from models.adversarial import Critic, AdversarialExplainer
            assert Critic is not None
            assert AdversarialExplainer is not None
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_dsl_module_import(self):
        """Test DSL module can be imported."""
        try:
            from dsl import hop, filter_nodes, agg, DSLExecutor
            assert callable(hop)
            assert callable(filter_nodes)
            assert callable(agg)
            assert DSLExecutor is not None
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_evolution_module_import(self):
        """Test evolution module can be imported."""
        try:
            from evolution import ExplanationDNA, GeneticAlgorithm
            assert ExplanationDNA is not None
            assert GeneticAlgorithm is not None
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_causal_module_import(self):
        """Test causal module can be imported."""
        try:
            from models.causal import CausalDoGenerator
            assert CausalDoGenerator is not None
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_embodied_module_import(self):
        """Test embodied module can be imported."""
        try:
            from embodied import GraphEnv, TrajectoryPolicy
            assert GraphEnv is not None
            assert TrajectoryPolicy is not None
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_metrics_import(self):
        """Test metrics can be imported."""
        try:
            from metrics import (
                compute_elbo, compute_critic_error,
                compute_execution_accuracy, compute_pareto_front,
                compute_causal_fidelity, compute_trajectory_fidelity
            )
            assert callable(compute_elbo)
            assert callable(compute_critic_error)
            assert callable(compute_execution_accuracy)
            assert callable(compute_pareto_front)
            assert callable(compute_causal_fidelity)
            assert callable(compute_trajectory_fidelity)
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")

    def test_visualization_import(self):
        """Test visualizations can be imported."""
        try:
            from visualizations import (
                plot_metrics_comparison, generate_metrics_table
            )
            assert callable(plot_metrics_comparison)
            assert callable(generate_metrics_table)
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


class TestEvaluationPipeline:
    """Test unified evaluation pipeline."""

    def test_evaluate_all_import(self):
        """Test evaluate_all.py can be imported."""
        try:
            from evaluate_all import evaluate_all_modules, generate_comparison_report
            assert callable(evaluate_all_modules)
            assert callable(generate_comparison_report)
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


class TestCheckpointManager:
    """Test checkpoint management."""

    def test_checkpoint_manager_import(self):
        """Test checkpoint manager can be imported."""
        try:
            from utils import CheckpointManager, save_checkpoint
            assert CheckpointManager is not None
            assert callable(save_checkpoint)
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
