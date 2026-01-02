"""
Adversarial Metrics
Evaluation metrics for Generator vs Critic adversarial training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any


def compute_critic_error(critic, explanations: List[str], graphs: List[Any],
                         original_logits: torch.Tensor, tokenizer,
                         device: str = 'cuda') -> Dict[str, float]:
    """
    Compute Critic's reconstruction error.

    Args:
        critic: Critic model
        explanations: List of explanation texts
        graphs: List of graph objects
        original_logits: Original GNN logits [N, n_class]
        tokenizer: Tokenizer for explanations
        device: Computation device

    Returns:
        Dict with error metrics
    """
    critic.eval()
    all_mse = []
    all_mae = []

    with torch.no_grad():
        for i, (expl, graph) in enumerate(zip(explanations, graphs)):
            # Tokenize explanation
            encoding = tokenizer(
                expl,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)

            # Get subgraph features (mock for now)
            subg_x = torch.randn(10, 256).to(device)
            subg_edge_index = torch.tensor([[0,1,2,3], [1,2,3,4]]).to(device)

            # Critic prediction
            pred_logits = critic(
                expl_ids=encoding['input_ids'],
                expl_attention_mask=encoding['attention_mask'],
                subg_x=subg_x,
                subg_edge_index=subg_edge_index
            )

            # Compute errors
            target = original_logits[i:i+1].to(device)
            mse = F.mse_loss(pred_logits, target).item()
            mae = F.l1_loss(pred_logits, target).item()

            all_mse.append(mse)
            all_mae.append(mae)

    return {
        'mean_mse': np.mean(all_mse),
        'std_mse': np.std(all_mse),
        'mean_mae': np.mean(all_mae),
        'std_mae': np.std(all_mae)
    }


def compute_explanation_compression_ratio(explanations: List[str], 
                                           graph_texts: List[str]) -> Dict[str, float]:
    """
    Measure information compression ratio of explanations.

    Args:
        explanations: List of generated explanations
        graph_texts: List of original graph text descriptions

    Returns:
        Dict with compression metrics
    """
    ratios = []

    for expl, graph_text in zip(explanations, graph_texts):
        expl_len = len(expl.split())
        graph_len = len(graph_text.split())

        if graph_len > 0:
            ratio = expl_len / graph_len
            ratios.append(ratio)

    return {
        'mean_compression_ratio': np.mean(ratios),
        'std_compression_ratio': np.std(ratios),
        'min_compression_ratio': np.min(ratios),
        'max_compression_ratio': np.max(ratios)
    }


def compute_information_bottleneck_metrics(explanations: List[str],
                                            original_texts: List[str],
                                            predictions: List[int]) -> Dict[str, float]:
    """
    Compute information bottleneck related metrics.

    Args:
        explanations: Generated explanations
        original_texts: Original graph texts
        predictions: Model predictions

    Returns:
        Dict with IB metrics
    """
    # Compute mutual information approximations

    # I(E; Y) - how much explanation preserves about prediction
    # Approximated by checking if prediction-related keywords are preserved
    pred_info_preserved = []
    for expl, pred in zip(explanations, predictions):
        # Simple heuristic: check if class-related terms appear
        expl_lower = expl.lower()
        info_score = 0
        if str(pred) in expl_lower or 'class' in expl_lower:
            info_score = 1
        pred_info_preserved.append(info_score)

    # I(E; X) - how much explanation preserves about input
    # Approximated by word overlap
    input_info_preserved = []
    for expl, orig in zip(explanations, original_texts):
        expl_words = set(expl.lower().split())
        orig_words = set(orig.lower().split())
        overlap = len(expl_words & orig_words) / len(orig_words) if orig_words else 0
        input_info_preserved.append(overlap)

    return {
        'prediction_info_preserved': np.mean(pred_info_preserved),
        'input_info_preserved': np.mean(input_info_preserved),
        'compression_efficiency': np.mean(pred_info_preserved) / (np.mean(input_info_preserved) + 1e-8)
    }


def compute_adversarial_equilibrium(generator_losses: List[float],
                                     critic_losses: List[float]) -> Dict[str, float]:
    """
    Analyze adversarial training equilibrium.

    Args:
        generator_losses: History of generator losses
        critic_losses: History of critic losses

    Returns:
        Dict with equilibrium metrics
    """
    gen_losses = np.array(generator_losses)
    crit_losses = np.array(critic_losses)

    # Check for convergence
    if len(gen_losses) > 10:
        gen_trend = np.polyfit(range(len(gen_losses[-10:])), gen_losses[-10:], 1)[0]
        crit_trend = np.polyfit(range(len(crit_losses[-10:])), crit_losses[-10:], 1)[0]
    else:
        gen_trend = 0
        crit_trend = 0

    return {
        'generator_final_loss': gen_losses[-1] if len(gen_losses) > 0 else 0,
        'critic_final_loss': crit_losses[-1] if len(crit_losses) > 0 else 0,
        'generator_trend': gen_trend,
        'critic_trend': crit_trend,
        'loss_ratio': gen_losses[-1] / (crit_losses[-1] + 1e-8) if len(gen_losses) > 0 else 0
    }
