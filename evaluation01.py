"""
Unified evaluation harness for UCert.

This module orchestrates:
  - explanation generation
  - metric computation
  - certificate issuance
  - result aggregation

It provides a single entry point for experiments.
"""

from typing import List, Dict
from .metrics import (
    brevity,
    simulatability,
    pmi_score,
    uc_coverage
)
from .utils import logger


class EvaluationHarness:
    """
    Unified evaluation pipeline for TAG explainability experiments.
    """

    def __init__(
        self,
        explainer,
        evaluator,
        certificate_issuer
    ):
        self.explainer = explainer
        self.evaluator = evaluator
        self.certificate_issuer = certificate_issuer

    def evaluate_instance(
        self,
        serialized_graph: str,
        masked_graph: str,
        salient_tokens: List[str],
        label_token: str
    ) -> Dict[str, float]:
        """
        Evaluate a single graph instance.
        """
        explanation = self.explainer(serialized_graph)

        logp_with = self.evaluator.lm.conditional_logprob(
            masked_graph + explanation,
            salient_tokens[0]
        )
        logp_without = self.evaluator.lm.conditional_logprob(
            masked_graph,
            salient_tokens[0]
        )

        sim = simulatability(
            self.evaluator.lm.conditional_logprob(
                explanation + serialized_graph,
                label_token
            )
        )

        return {
            "brevity": brevity(explanation),
            "pmi": pmi_score(logp_with, logp_without),
            "simulatability": sim
        }

    def evaluate_dataset(
        self,
        dataset
    ) -> Dict[str, float]:
        """
        Run evaluation over a dataset split.
        """
        metric_records = []
        uc_results = []

        for item in dataset:
            metrics = self.evaluate_instance(**item)
            metric_records.append(metrics)

            cert = self.certificate_issuer(**item)
            uc_results.append(cert is not None)

        aggregated = {
            **{k: sum(m[k] for m in metric_records) / len(metric_records)
               for k in metric_records[0]},
            "uc_coverage": uc_coverage(uc_results)
        }

        logger.info(f"Evaluation summary: {aggregated}")
        return aggregated
