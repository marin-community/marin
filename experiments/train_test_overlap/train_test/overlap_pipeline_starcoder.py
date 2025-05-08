import logging
import os

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_get_curr_subdirectories
from train_test_overlap.run_data_overlap import DataOverlapPipelineConfig, run_data_overlap

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Replace static ExecutorStep with dynamic generation of steps per subdirectory
def generate_overlap_steps(base_input_dir, scenario_data, n_values, processes):
    steps = []
    subdirs = fsspec_get_curr_subdirectories(base_input_dir)
    print(f"Found {len(subdirs)} subdirectories under {base_input_dir}", flush=True)
    for subdir in subdirs:
        # list JSONL files under each subdirectory
        if "fortran" not in subdir:
            continue
        print(f"Processing subdir {subdir}", flush=True)
        sub_dir_name = os.path.basename(subdir.rstrip("/"))
        step_name = f"train_test_overlap/starcoder_data_overlap_pipeline_subset/{sub_dir_name}"
        print(f"Step name: {step_name}", flush=True)
        config = DataOverlapPipelineConfig(
            input_data=subdir,
            scenario_data=scenario_data,
            output_path=this_output_path(),
            N=n_values,
            processes=processes,
        )
        steps.append(ExecutorStep(name=step_name, fn=run_data_overlap, config=config))
    return steps


if __name__ == "__main__":
    base_input_dir = "gs://marin-us-central2/documents/starcoderdata-720c8c/9fc30b5/"
    scenario_data = "gs://marin-us-central2/scenario_data/scenario_data.jsonl"
    # generate one step per subdir under the base input directory
    steps = generate_overlap_steps(base_input_dir, scenario_data, [5, 10, 15, 20], 4)
    executor_main(
        steps=steps,
        description="Run n-gram data overlap pipeline on StarCoder data for each subdir",
    )
