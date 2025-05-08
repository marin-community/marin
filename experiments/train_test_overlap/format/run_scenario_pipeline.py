#!/usr/bin/env python3
import logging

from experiments.train_test_overlap.format.consolidate_scenario_jsonl import get_consolidation_step, get_conversion_step
from marin.execution.executor import executor_main

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Run the full scenario conversion and consolidation pipeline."""
    # Get the conversion step
    conversion_step = get_conversion_step()

    # Get the consolidation step
    consolidation_step = get_consolidation_step(conversion_step)

    # Create a list of all steps to run in sequence
    steps = [
        conversion_step,
        consolidation_step,
    ]

    # Run the executor with all steps
    executor_main(steps=steps, description="Run the full scenario conversion and consolidation pipeline")


if __name__ == "__main__":
    main()
