import logging
import os

from experiments.pretraining_datasets import dclm_baseline  # This is an InputName
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    get_executor_step,
    this_output_path,
)
from marin.utils import fsspec_get_curr_subdirectories, fsspec_glob
from train_test_overlap.run_data_overlap import (
    DataOverlapPipelineConfig,
    run_data_overlap,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_dclm_overlap_steps(
    base_path_for_local_shards_str: str,
    scenario_data_path: str,
    n_values: list[int],
    processes: int,
):
    """
    Generates ExecutorSteps for processing each compressed JSONL shard file under every local shard.
    """
    steps = []
    print(f"Attempting to find global shards under: {base_path_for_local_shards_str}", flush=True)
    global_shard_dirs = fsspec_get_curr_subdirectories(base_path_for_local_shards_str)
    print(f"Found {len(global_shard_dirs)} potential global shard directories: {global_shard_dirs}", flush=True)

    for global_shard_dir in global_shard_dirs:
        global_shard_name = os.path.basename(global_shard_dir.rstrip("/"))
        if not global_shard_name.startswith("global-shard_"):
            print(f"Skipping non-global-shard directory: {global_shard_dir}", flush=True)
            continue

        print(f"Processing global shard: {global_shard_dir}", flush=True)
        local_shard_dirs = fsspec_get_curr_subdirectories(global_shard_dir)
        print(
            f"Found {len(local_shard_dirs)} local shard directories under {global_shard_dir}: {local_shard_dirs}",
            flush=True,
        )

        for local_shard_dir in local_shard_dirs:
            local_shard_name = os.path.basename(local_shard_dir.rstrip("/"))
            if not local_shard_name.startswith("local-shard_"):
                print(f"Skipping non-local-shard directory: {local_shard_dir}", flush=True)
                continue

            # discover compressed JSONL shards under this local shard
            input_patterns = [
                "**/*.jsonl.gz",
                "**/*.jsonl.zst",
                "**/*.jsonl.gs",
                "**/*.json.gz",
                "**/*.json.zst",
                "**/*.jsonl",
            ]
            all_files = set()
            for patt in input_patterns:
                pattern = os.path.join(local_shard_dir.rstrip("/"), patt)
                files = fsspec_glob(pattern)
                print(f"Found {len(files)} files for pattern {patt} under {local_shard_dir}", flush=True)
                all_files.update(files)
            print(f"Total {len(all_files)} shard files under {local_shard_dir}", flush=True)

            # create one step per file
            for file_path in sorted(all_files):
                shard = os.path.basename(file_path)
                step_name = (
                    f"train_test_overlap/dclm_data_overlap_per_file/{global_shard_name}/{local_shard_name}/{shard}"
                )
                config = DataOverlapPipelineConfig(
                    input_data=file_path,
                    scenario_data=scenario_data_path,
                    output_path=this_output_path(),
                    N=n_values,
                    processes=processes,
                )
                steps.append(ExecutorStep(name=step_name, fn=run_data_overlap, config=config))

    print(f"Total overlap steps generated: {len(steps)}", flush=True)
    return steps


if __name__ == "__main__":
    print("Starting DCLM overlap pipeline script...", flush=True)

    marin_prefix = os.environ.get("MARIN_PREFIX")
    if not marin_prefix:
        raise ValueError("MARIN_PREFIX environment variable not set. Please set it (e.g., 'gs://marin-us-central2').")
    print(f"Using MARIN_PREFIX: {marin_prefix}", flush=True)

    # dclm_baseline is an InputName. We need the underlying ExecutorStep
    # that actually produces/downloads the data to construct paths.
    dclm_producing_step = get_executor_step(dclm_baseline)
    print(f"dclm_baseline (InputName): {dclm_baseline}", flush=True)
    print(f"dclm_producing_step (ExecutorStep): {dclm_producing_step.name}", flush=True)
    print(f"  dclm_producing_step.override_output_path: {dclm_producing_step.override_output_path}", flush=True)
    print(f"  dclm_baseline.name (from .cd()): {dclm_baseline.name}", flush=True)
    print(f"  dclm_producing_step.config.hf_dataset_id: {dclm_producing_step.config.hf_dataset_id}", flush=True)
    print(f"  dclm_producing_step.config.revision: {dclm_producing_step.config.revision}", flush=True)

    # Construct the path where the dclm_baseline step downloads the data
    # This is MARIN_PREFIX / override_output_path / dclm_baseline.name (which is the revision from .cd())
    resolved_dclm_output_base_path = os.path.join(
        marin_prefix,
        dclm_producing_step.override_output_path,  # Should be "raw/dclm"
        dclm_baseline.name,  # Should be "a3b142c" (revision from .cd())
    )
    print(
        f"Resolved DCLM output base path (where HF files are downloaded): {resolved_dclm_output_base_path}", flush=True
    )

    # The actual global shards are located deeper, following the Hugging Face structure
    # MARIN_PREFIX/raw/dclm/a3b142c/huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/resolve/a3b142c/
    base_path_for_dclm_shards_str = os.path.join(
        resolved_dclm_output_base_path,
        "huggingface.co/datasets",
        dclm_producing_step.config.hf_dataset_id,  # "mlfoundations/dclm-baseline-1.0"
        "resolve",
        dclm_producing_step.config.revision,  # "a3b142c"
    )
    print(f"Constructed base path for DCLM global shards: {base_path_for_dclm_shards_str}", flush=True)

    # Define scenario data (can be the same as starcoder for now, or a different one)
    scenario_data = "gs://marin-us-central2/scenarios/consolidated_eval_scenarios-d3f040/consolidated_scenarios.jsonl"
    print(f"Using scenario data: {scenario_data}", flush=True)

    # N-gram values and processes
    n_gram_values = [5, 10, 15, 20]
    num_processes = 4
    print(f"Using N-gram values: {n_gram_values}, Processes: {num_processes}", flush=True)

    # Generate one overlap analysis step per local shard
    dclm_overlap_steps = generate_dclm_overlap_steps(
        base_path_for_dclm_shards_str,
        scenario_data,
        n_gram_values,
        num_processes,
    )

    # The dclm_producing_step must be part of the executor's knowledge
    # to ensure data is downloaded/available if not already present.
    all_steps = [dclm_producing_step, *dclm_overlap_steps]
    print(
        f"Total steps for executor_main: {len(all_steps)} (1 for DCLM download + {len(dclm_overlap_steps)} for overlap)",
        flush=True,
    )
    if not dclm_overlap_steps:
        print("WARNING: No DCLM overlap steps generated Check paths & bucket contents.", flush=True)
        print(
            f"Check subdirectories matching 'global-shard_*/local-shard_*' exist under: {base_path_for_dclm_shards_str}",
            flush=True,
        )

    executor_main(
        steps=all_steps,
        description="Run n-gram data overlap pipeline on DCLM dataset for each shard file",
        # prefix and executor_info_base_path will be handled by executor_main's default logic
        # or ExecutorMainConfig if you choose to use draccus for this script too.
    )
    print("DCLM overlap pipeline script finished.", flush=True)
