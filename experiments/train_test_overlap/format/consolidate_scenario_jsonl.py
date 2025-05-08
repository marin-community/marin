import json
import logging
import os
from dataclasses import dataclass

import fsspec

from experiments.train_test_overlap.format.test_scenario_conversation import conversion_step
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.utils import fsspec_glob, fsspec_mkdirs

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConsolidateScenarioConfig:
    """Configuration for consolidating scenario JSONL files."""

    input_step: ExecutorStep
    output_path: str
    output_filename: str = "consolidated_scenarios.jsonl"


def consolidate_scenario_files(config: ConsolidateScenarioConfig) -> str:
    """Find all scenario JSONL files and consolidate them into a single file."""
    input_path = output_path_of(config.input_step)
    print(f"input_path: {input_path}", flush=True)
    logger.info(f"Looking for scenario files in: {input_path}")

    output_file = os.path.join(config.output_path, config.output_filename)
    fsspec_mkdirs(config.output_path)

    # Find all scenario JSONL files recursively
    scenario_files = []
    # Look for files with scenario in the name and .jsonl extension
    scenario_files.extend(fsspec_glob(os.path.join(input_path.step, "**", "scenario_*.jsonl")))

    if not scenario_files:
        logger.warning(f"No scenario files found in {input_path}")
        return "No files found to consolidate"

    logger.info(f"Found {len(scenario_files)} scenario files to consolidate")

    # Consolidate the files
    scenario_count = 0
    instance_count = 0

    with fsspec.open(output_file, "wt") as out_f:
        for file_path in scenario_files:
            with fsspec.open(file_path, "rt") as in_f:
                for line in in_f:
                    # Parse and validate each scenario
                    scenario = json.loads(line)
                    # Make sure it has the expected structure
                    if "scenario_key" in scenario and "instances" in scenario:
                        out_f.write(line)
                        scenario_count += 1
                        instance_count += len(scenario["instances"])
                    else:
                        logger.warning(f"Skipping invalid scenario in {file_path}")

    logger.info(f"Consolidated {scenario_count} scenarios with {instance_count} instances into {output_file}")
    return f"Successfully consolidated {scenario_count} scenarios with {instance_count} instances"


# Create a ConsolidateScenarioConfig using the conversion_step
consolidate_config = ConsolidateScenarioConfig(
    input_step=conversion_step,
    output_filename="consolidated_scenarios.jsonl",
    output_path=this_output_path(),
)

# Create the consolidation step using the proper config type
consolidation_step = ExecutorStep(
    name="scenarios/consolidated_eval_scenarios",
    fn=lambda cfg: consolidate_scenario_files(cfg),
    config=consolidate_config,
)


if __name__ == "__main__":
    executor_main(steps=[consolidation_step], description="Consolidate scenario JSONL files into a single file")
