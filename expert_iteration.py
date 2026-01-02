"""
GraphNarrator-style expert iteration loop.

This module performs pseudo-label refinement by iteratively:
  1) generating explanations
  2) scoring them with XAIEvaluator
  3) filtering or rewriting explanations
"""

from typing import List, Dict
from .llm_interface import LLMInterface
from .evaluator import XAIEvaluator
from .utils import logger


class ExpertIteration:
    """
    Implements expert iteration for explanation refinement.
    """

    def __init__(
        self,
        generator: LLMInterface,
        evaluator: XAIEvaluator,
        num_rounds: int = 5,
        thresholds: Dict[str, float] = None
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.num_rounds = num_rounds
        self.thresholds = thresholds or {
            "saliency": 0.0,
            "faithfulness": 0.0
        }

    def run(
        self,
        serialized_graph: str,
        label_token: str,
        masked_graph: str,
        salient_tokens: List[str]
    ) -> str:
        """
        Run the expert-iteration loop and return the final explanation.
        """
        explanation = self.generator.generate(
            f"Explain the prediction {label_token} given:\n{serialized_graph}\nExplanation:"
        )

        for t in range(self.num_rounds):
            s = self.evaluator.mlm_saliency(
                serialized_graph, explanation, masked_graph, salient_tokens
            )
            f = self.evaluator.faithfulness_to_prediction(
                serialized_graph, explanation, label_token
            )

            if s >= self.thresholds["saliency"] and f >= self.thresholds["faithfulness"]:
                logger.info(f"Iteration {t}: explanation accepted.")
                break

            prompt = (
                "Refine the explanation to improve faithfulness and reduce verbosity.\n"
                f"Graph:\n{serialized_graph}\n"
                f"Previous explanation:\n{explanation}\n"
                "Rewritten explanation:"
            )
            explanation = self.generator.generate(prompt)

        return explanation
