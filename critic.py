"""
Critic for adversarial explanation optimization.

Scores explanations for:
  - faithfulness
  - brevity
Used to compute generator loss.
"""

from typing import Dict
import torch
from .evaluator import XAIEvaluator


class ExplanationCritic:
    """
    Critic that evaluates explanations using XAIEvaluator metrics.
    """

    def __init__(self, evaluator: XAIEvaluator, lambda_len: float = 0.01):
        self.evaluator = evaluator
        self.lambda_len = lambda_len

    def score(
        self,
        serialized_graph: str,
        explanation: str,
        label_token: str
    ) -> Dict[str, float]:
        faith = self.evaluator.faithfulness_to_prediction(
            serialized_graph, explanation, label_token
        )
        length_penalty = self.lambda_len * len(explanation.split())
        return {
            "faithfulness": faith,
            "length_penalty": length_penalty,
            "total": faith - length_penalty
        }
