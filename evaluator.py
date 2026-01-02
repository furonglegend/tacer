"""
XAIEvaluator implementation for UCert.

Provides LM-based proxy metrics:
  - MLM-style saliency (PMI estimator)
  - Faithfulness-to-prediction
  - Simulatability proxy

All metrics are deterministic given a fixed evaluator LM.
"""

from typing import List
import numpy as np
from .llm_interface import LLMInterface


class XAIEvaluator:
    """
    Implements proxy evaluation metrics used in UCert experiments.
    """

    def __init__(self, evaluator_lm: LLMInterface):
        self.lm = evaluator_lm

    def mlm_saliency(
        self,
        serialized_graph: str,
        explanation: str,
        masked_graph: str,
        salient_tokens: List[str]
    ) -> float:
        """
        PMI-style saliency recovery score.

        Estimates how much the explanation helps recover masked salient tokens.
        """
        deltas = []
        for tok in salient_tokens:
            with_exp = self.lm.conditional_logprob(
                masked_graph + explanation,
                tok
            )
            without_exp = self.lm.conditional_logprob(
                masked_graph,
                tok
            )
            deltas.append(with_exp - without_exp)
        return float(np.mean(deltas)) if deltas else 0.0

    def faithfulness_to_prediction(
        self,
        serialized_graph: str,
        explanation: str,
        label_token: str
    ) -> float:
        """
        Faithfulness score measuring change in likelihood of the predicted label
        when conditioning on the explanation.
        """
        with_exp = self.lm.conditional_logprob(
            serialized_graph + explanation,
            label_token
        )
        without_exp = self.lm.conditional_logprob(
            serialized_graph,
            label_token
        )
        return with_exp - without_exp

    def simulatability(
        self,
        serialized_graph: str,
        explanation: str,
        label_token: str
    ) -> float:
        """
        Simulatability proxy: probability that the evaluator LM
        recovers the predicted label from the explanation.
        """
        return self.lm.conditional_logprob(
            explanation + serialized_graph,
            label_token
        )
