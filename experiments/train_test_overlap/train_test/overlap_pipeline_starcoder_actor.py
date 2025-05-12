import logging
import os

import ray

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob
from train_test_overlap.run_data_overlap import DataOverlapPipelineConfig
from train_test_overlap.run_data_overlap_actor import run_data_overlap

# NOTE: We wrap the Ray `run_data_overlap` remote function in a local Python callable
# `run_data_overlap_task` so that the Executor framework sees it as a normal function (not a RemoteFunction).
# This causes the Executor to reserve a full CPU per task (num_cpus=1), throttling concurrency
# and preventing the explosion of Python worker processes when running many shards.

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Wrap the remote function in a local Python callable so the Executor will reserve 1 CPU
def run_data_overlap_task(config):
    return ray.get(run_data_overlap.remote(config))


# Generate one ExecutorStep per compressed JSONL shard (file) under the base_input_dir
def generate_overlap_steps(base_input_dir, scenario_data, n_values, processes):
    """
    Generates one ExecutorStep per compressed JSONL shard under the base_input_dir,
    allowing each file to be processed independently.
    """
    steps = []
    # Patterns matching compressed JSONL files
    input_patterns = [
        "**/*.jsonl.gz",
        "**/*.jsonl.zst",
        "**/*.jsonl.gs",
        "**/*.json.gz",
        "**/*.json.zst",
        "**/*.jsonl",
    ]
    all_files = set()
    # Discover every shard file under the base directory
    for patt in input_patterns:
        pattern = os.path.join(base_input_dir.rstrip("/"), patt)
        files = fsspec_glob(pattern)
        logger.info(f"Found {len(files)} files for pattern {patt}")
        all_files.update(files)
    logger.info(f"Total {len(all_files)} shard files found under {base_input_dir}")

    # Create one step for each file
    for file_path in sorted(all_files):
        lang = os.path.basename(os.path.dirname(file_path.rstrip("/")))
        shard = os.path.basename(file_path)
        if lang != "ada":
            continue
        step_name = f"train_test_overlap/starcoder_data_overlap_per_file/{lang}/{shard}"
        config = DataOverlapPipelineConfig(
            input_data=file_path,
            scenario_data=scenario_data,
            output_path=this_output_path(),
            N=n_values,
            processes=processes,
        )
        # Use the local wrapper (Callable) so Executor throttles to 1 CPU
        steps.append(ExecutorStep(name=step_name, fn=run_data_overlap_task, config=config))
    return steps


if __name__ == "__main__":
    base_input_dir = "gs://marin-us-central2/documents/starcoderdata-720c8c/9fc30b5/"
    scenario_data = "gs://marin-us-central2/scenarios/consolidated_eval_scenarios-d3f040/consolidated_scenarios.jsonl"
    # generate one step per shard file under the base input directory
    steps = generate_overlap_steps(base_input_dir, scenario_data, [1, 3, 5, 10, 15, 20], 64)
    executor_main(
        steps=steps,
        description="Run n-gram data overlap pipeline on StarCoder data for each shard file",
    )
