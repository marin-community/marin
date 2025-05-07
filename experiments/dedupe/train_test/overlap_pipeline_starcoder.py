import logging
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from operations.data_overlap.run_data_overlap_pipeline import DataOverlapPipelineConfig, run_data_overlap
from marin.utils import fsspec_get_curr_subdirectories, fsspec_glob
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Replace static ExecutorStep with dynamic generation of steps per subdirectory
def generate_overlap_steps(base_input_dir, scenario_data, output_base_path, n_values, processes):
    steps = []
    subdirs = fsspec_get_curr_subdirectories(base_input_dir)
    for subdir in subdirs:
        # list JSONL files under each subdirectory
        jsonl_files = fsspec_glob(os.path.join(subdir, "*.jsonl"))
        if not jsonl_files:
            continue
        model_name = os.path.basename(subdir.rstrip("/"))
        step_name = f"train_test_overlap/starcoder_data_overlap_pipeline/{model_name}"
        config = DataOverlapPipelineConfig(
            input_data=subdir,
            scenario_data=scenario_data,
            output_path=os.path.join(output_base_path, model_name),
            input_format="the_pile",
            normalization="default",
            N=n_values,
            processes=processes,
        )
        steps.append(ExecutorStep(name=step_name, fn=run_data_overlap, config=config))
    return steps

if __name__ == "__main__":
    base_input_dir = "gs://marin-us-central2/documents/starcoderdata-720c8c/9fc30b5/"
    scenario_data = "gs://marin-us-central2/scenario_data/scenario_data.jsonl"
    output_base_path = this_output_path()
    # generate one step per subdir under the base input directory
    steps = generate_overlap_steps(base_input_dir, scenario_data, output_base_path, [8, 10, 13], 4)
    executor_main(
        steps=steps,
        description="Run n-gram data overlap pipeline on StarCoder data for each subdir",
    ) 