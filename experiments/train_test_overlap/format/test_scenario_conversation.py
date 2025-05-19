import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import fsspec

from experiments.train_test_overlap.eval_datasets_overlap import (
    bbh_convert_dolma,
    gpqa_convert_dolma,
    gsm8k_convert_dolma,
    instruction_following_convert_dolma,
    math_convert_dolma,
    mmlu_convert_dolma,
    mmlu_pro_convert_dolma,
    musr_convert_dolma,
    truthful_qa_convert_dolma,
    winograd_wsc_convert_dolma,
    commonsense_qa_convert_dolma,
    lambada_openai_convert_dolma,
    openbookqa_convert_dolma,
    piqa_convert_dolma,
    hellaswag_convert_dolma,
    ai2_arc_convert_dolma,
    boolq_convert_dolma,
    piqa_convert_dolma,
)
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.utils import fsspec_get_curr_subdirectories, fsspec_glob, fsspec_mkdirs

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ScenarioConversionConfig:
    """Configuration for converting decontamination data to scenario format."""

    input_steps: list[ExecutorStep]
    output_path: str
    scenario_class_name_prefix: str = "helm.benchmark.scenarios"


@dataclass
class LightScenarioKey:
    """Key for LightScenario."""

    scenario_spec: dict[str, Any]
    split: str


@dataclass
class LightInstance:
    """A lighter Instance with only text fields."""

    input: str
    references: list[str]
    id: str | None = None


@dataclass
class LightScenario:
    """A lighter Scenario."""

    scenario_key: LightScenarioKey
    instances: list[LightInstance]


def get_scenario_class_name(dataset_name: str, prefix: str = "helm.benchmark.scenarios") -> str:
    """Determine the scenario class name based on the dataset name."""
    dataset_name_lowered = dataset_name.lower()

    if "mmlu" in dataset_name_lowered:
        return f"{prefix}.mmlu_scenario.MMLUScenario"
    elif "gsm8k" in dataset_name_lowered:
        return f"{prefix}.gsm8k_scenario.GSM8KScenario"
    elif "math" in dataset_name_lowered:
        return f"{prefix}.math_scenario.MathScenario"
    elif "truthful_qa" in dataset_name_lowered:
        return f"{prefix}.truthful_qa_scenario.TruthfulQAScenario"
    elif "bbh" in dataset_name_lowered:
        return f"{prefix}.bbh_scenario.BBHScenario"
    elif "gpqa" in dataset_name_lowered:
        return f"{prefix}.gpqa_scenario.GpqaScenario"
    elif "instruction_following" in dataset_name_lowered:
        return f"{prefix}.instruction_following_scenario.InstructionFollowingScenario"
    elif "musr" in dataset_name_lowered:
        return f"{prefix}.musr_scenario.MusrScenario"
    elif "winograd_wsc" in dataset_name_lowered:
        return f"{prefix}.winograd_wsc_scenario.WinogradWscScenario"
    elif "commonsense_qa" in dataset_name_lowered:
        return f"{prefix}.commonsense_qa_scenario.CommonsenseQAScenario"
    elif "lambada_openai" in dataset_name_lowered:
        return f"{prefix}.lambada_openai_scenario.LambadaOpenaiScenario"
    elif "openbookqa" in dataset_name_lowered:
        return f"{prefix}.openbookqa_scenario.OpenbookqaScenario"
    elif "piqa" in dataset_name_lowered:
        return f"{prefix}.piqa_scenario.PiqaScenario"
    elif "hellaswag" in dataset_name_lowered:
        return f"{prefix}.hellaswag_scenario.HellaswagScenario"
    elif "ai2_arc" in dataset_name_lowered:
        return f"{prefix}.ai2_arc_scenario.Ai2ArcScenario"
    elif "boolq" in dataset_name_lowered:
        return f"{prefix}.boolq_scenario.BoolqScenario"
    elif "mmlu_pro" in dataset_name_lowered:
        return f"{prefix}.mmlu_pro_scenario.MmluProScenario"
    elif "mmlu_pro" in dataset_name_lowered:
        return f"{prefix}.mmlu_pro_scenario.MmluProScenario"
        
        
    else:
        return f"{prefix}.{dataset_name_lowered}_scenario.{dataset_name}Scenario"


def extract_references(item: dict[str, Any]) -> list[str]:
    """Extract references from metadata, prioritizing answer_labels, then options, then answer."""
    metadata = item.get("metadata", {})

    # First check for answer_labels + options
    if "answer_labels" in metadata and "options" in metadata:
        return metadata["options"]

    # Then check for just options
    if "options" in metadata:
        return metadata["options"]

    # Finally, use the answer as a single reference
    if "answer" in metadata:
        return [metadata["answer"]]

    # If nothing is found, return an empty list
    return []


def create_scenario_instance(item: dict[str, Any], instance_id: str) -> LightInstance:
    """Create a scenario instance from a decontamination item."""
    return LightInstance(input=item.get("text", ""), references=extract_references(item), id=instance_id)


def extract_subset_and_args(item: dict[str, Any]) -> dict[str, Any]:
    """Extract subset and create args dictionary for scenario spec."""
    metadata = item.get("metadata", {})
    subset = metadata.get("subset", "unknown")

    # For MMLU, the subject is the subset
    if "mmlu" in item.get("source", "").lower():
        return {"subject": subset}

    # For other datasets, we might need different mappings
    return {"subset": subset}


def process_decontamination_file(input_file: str, output_file: str, scenario_class_name_prefix: str) -> None:
    """Process a decontamination file and convert it to scenario format."""
    logger.info(f"Processing file {input_file} -> {output_file}")

    # Create a unique output file per input file to ensure parallel processing
    file_basename = os.path.basename(input_file).split(".")[0]
    output_dir = os.path.dirname(output_file)
    unique_output_file = os.path.join(output_dir, f"scenario_{file_basename}.jsonl")

    # Read the input file
    with fsspec.open(input_file, "rt", compression="infer") as f:
        items = [json.loads(line) for line in f]

    if not items:
        logger.warning(f"No items found in {input_file}")
        return

    # Group items by subset
    items_by_subset = {}
    for item in items:
        metadata = item.get("metadata", {})
        subset = metadata.get("subset", "unknown")
        split = metadata.get("split", "test")
        if subset not in items_by_subset:
            items_by_subset[subset] = {"subset": subset, "split": split, "items": []}
        items_by_subset[subset]["items"].append(item)

    # Create scenarios
    scenarios = []
    for subset_info in items_by_subset.values():
        subset = subset_info["subset"]
        split = subset_info["split"]
        subset_items = subset_info["items"]

        # Get first item to extract source and create scenario spec
        first_item = subset_items[0]
        source = first_item.get("source", "unknown")

        # Create scenario key
        args = extract_subset_and_args(first_item)
        scenario_class_name = get_scenario_class_name(source, scenario_class_name_prefix)

        scenario_key = LightScenarioKey(scenario_spec={"class_name": scenario_class_name, "args": args}, split=split)

        # Create instances
        instances = []
        for i, item in enumerate(subset_items):
            instance_id = item.get("id", f"id{i}")
            if isinstance(instance_id, str) and "decontamination" in instance_id:
                # Extract just the number part if it's a decontamination ID
                instance_id = f"id{instance_id.split('-')[-1]}"
            instances.append(create_scenario_instance(item, instance_id))

        # Create and add scenario
        scenario = LightScenario(scenario_key=scenario_key, instances=instances)
        scenarios.append(scenario)

    # Write output
    fsspec_mkdirs(output_dir)

    with fsspec.open(unique_output_file, "wt") as f:
        for scenario in scenarios:
            f.write(
                json.dumps(
                    {
                        "scenario_key": {
                            "scenario_spec": scenario.scenario_key.scenario_spec,
                            "split": scenario.scenario_key.split,
                        },
                        "instances": [
                            {"input": instance.input, "references": instance.references, "id": instance.id}
                            for instance in scenario.instances
                        ],
                    }
                )
                + "\n"
            )

    logger.info(f"Created scenario file {unique_output_file} with {len(scenarios)} scenarios")


def convert_datasets_to_scenarios(config: ScenarioConversionConfig) -> str:
    """Convert decontamination datasets to scenario format."""
    logger.info(f"Starting conversion of {len(config.input_steps)} datasets")

    # Process each input step
    for step in config.input_steps:
        step_name = step.split("/")[-1]  # Extract the last part of the step name
        output_directory = os.path.join(config.output_path, step_name)
        output_file = os.path.join(output_directory, "scenario.jsonl")

        # Find input files in the step's output path
        input_path = output_path_of(step)
        logger.info(f"Processing input_path: {input_path}")

        # Use pattern matching to find relevant files
        # First try *.jsonl.gz files
        input_files = fsspec_glob(os.path.join(input_path.step, "*.jsonl.gz"))
        logger.info(f"Found {len(input_files)} .jsonl.gz files")

        # If no .jsonl.gz files, try *.jsonl files
        if not input_files:
            input_files = fsspec_glob(os.path.join(input_path.step, "*.jsonl"))
            logger.info(f"Found {len(input_files)} .jsonl files")

        # If still no files, try all subdirectories
        if not input_files:
            logger.info(f"Checking subdirectories in {input_path}")
            subdirs = fsspec_get_curr_subdirectories(input_path.step)
            logger.info(f"Found subdirectories: {subdirs}")

            for subdir in subdirs:
                subdir_files = fsspec_glob(os.path.join(subdir, "*.jsonl.gz"))
                if not subdir_files:
                    subdir_files = fsspec_glob(os.path.join(subdir, "*.jsonl"))
                input_files.extend(subdir_files)

            logger.info(f"Found {len(input_files)} files in subdirectories")

        if not input_files:
            logger.warning(f"No input files found for step {step_name} at {input_path}")
            continue

        # Process each input file - now creates unique output files
        for input_file in input_files:
            process_decontamination_file(input_file, output_file, config.scenario_class_name_prefix)

        # Log information about created scenario files
        created_files = fsspec_glob(os.path.join(output_directory, "scenario_*.jsonl"))
        logger.info(f"Created {len(created_files)} scenario files for step {step_name}")

    return "Conversion to scenarios completed, with unique files per input source"


# Define the configuration with the executor steps
scenario_conversion_config = ScenarioConversionConfig(
    input_steps=[
        bbh_convert_dolma,
        gpqa_convert_dolma,
        gsm8k_convert_dolma,
        instruction_following_convert_dolma,
        math_convert_dolma,
        mmlu_convert_dolma,
        mmlu_pro_convert_dolma,
        musr_convert_dolma,
        truthful_qa_convert_dolma,
        winograd_wsc_convert_dolma,
        commonsense_qa_convert_dolma,
        lambada_openai_convert_dolma,
        openbookqa_convert_dolma,
        piqa_convert_dolma,
        hellaswag_convert_dolma,
        ai2_arc_convert_dolma,
        boolq_convert_dolma,
        piqa_convert_dolma,
    ],
    output_path=this_output_path(),
    scenario_class_name_prefix="helm.benchmark.scenarios",
)

# Create an executor step for the conversion
conversion_step = ExecutorStep(
    name="scenarios/eval_scenarios",
    fn=lambda cfg: convert_datasets_to_scenarios(cfg),
    config=scenario_conversion_config,
)

if __name__ == "__main__":
    executor_main(steps=[conversion_step], description="Convert decontamination datasets to scenario format")
