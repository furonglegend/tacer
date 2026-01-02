"""
Command-line interface for running UCert experiments.

Provides a reproducible entry point for:
  - training
  - explanation generation
  - evaluation
  - certificate issuance
"""

import argparse
from .evaluation import EvaluationHarness
from .utils import set_seed, logger


def main():
    parser = argparse.ArgumentParser(
        description="UCert: Formal Unexplainability Experiments"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="eval",
                        choices=["train", "eval", "certify"])
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info(f"Running UCert in {args.mode} mode")

    if args.mode == "eval":
        logger.info("Launching evaluation pipeline")
        # Instantiate models, evaluators, dataset here
    elif args.mode == "certify":
        logger.info("Issuing Unexplainability Certificates")
    else:
        logger.info("Training TAG classifier")


if __name__ == "__main__":
    main()
