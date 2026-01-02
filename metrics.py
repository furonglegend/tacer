"""
Evaluation metrics for UCert.

This module implements:
  - Simulatability
  - PMI-style saliency
  - Brevity
  - Unexplainability Certificate (UC) coverage metrics

All metrics are model-agnostic and reproducible.
"""

import math
from typing import Dict, List


def brevity(explanation: str) -> float:
    """
    Brevity score defined as inverse token length.
    Higher is better.
    """
    length = max(len(explanation.split()), 1)
    return 1.0 / length


def pmi_score(
    logp_with: float,
    logp_without: float
) -> float:
    """
    Pointwise Mutual Information proxy used in saliency recovery.
    """
    return logp_with - logp_without


def simulatability(
    logp_label_given_expl: float
) -> float:
    """
    Simulatability proxy.
    Measures how well the explanation allows recovery of the label.
    """
    return logp_label_given_expl


def uc_coverage(
    certificate_results: List[bool]
) -> float:
    """
    UC-Coverage metric.

    Measures the fraction of instances for which a valid
    Unexplainability Certificate was successfully issued.
    """
    if not certificate_results:
        return 0.0
    return sum(certificate_results) / len(certificate_results)


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metric dictionary across runs or seeds.
    """
    agg = {}
    for key in results[0]:
        agg[key] = sum(r[key] for r in results) / len(results)
    return agg
