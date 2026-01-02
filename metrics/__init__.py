"""Metrics module for all evaluation functions."""

from .vae_metrics import (
    compute_elbo, compute_explanation_diversity, 
    sample_latent_space, compute_latent_statistics
)
from .adv_metrics import (
    compute_critic_error, compute_explanation_compression_ratio,
    compute_information_bottleneck_metrics, compute_adversarial_equilibrium
)
from .code_acc import (
    compute_execution_accuracy, compute_script_validity,
    compute_script_complexity, compute_script_diversity
)
from .evolution_metrics import (
    compute_pareto_front, compute_hypervolume, compare_to_baseline,
    analyze_evolution_history, compute_selection_pressure
)
from .causal_fidelity import (
    compute_causal_fidelity, compute_intervention_minimality,
    test_causal_significance, compute_counterfactual_validity
)
from .embodied_metrics import (
    compute_trajectory_fidelity, compute_trajectory_efficiency,
    compute_trajectory_coherence, compute_user_study_metrics,
    trajectory_to_explanation
)

__all__ = [
    # VAE metrics
    'compute_elbo', 'compute_explanation_diversity',
    'sample_latent_space', 'compute_latent_statistics',
    # Adversarial metrics
    'compute_critic_error', 'compute_explanation_compression_ratio',
    'compute_information_bottleneck_metrics', 'compute_adversarial_equilibrium',
    # Code/DSL metrics
    'compute_execution_accuracy', 'compute_script_validity',
    'compute_script_complexity', 'compute_script_diversity',
    # Evolution metrics
    'compute_pareto_front', 'compute_hypervolume', 'compare_to_baseline',
    'analyze_evolution_history', 'compute_selection_pressure',
    # Causal metrics
    'compute_causal_fidelity', 'compute_intervention_minimality',
    'test_causal_significance', 'compute_counterfactual_validity',
    # Embodied metrics
    'compute_trajectory_fidelity', 'compute_trajectory_efficiency',
    'compute_trajectory_coherence', 'compute_user_study_metrics',
    'trajectory_to_explanation'
]
