"""
python marin/run/ray_run.py --env_vars HF_TOKEN -- python experiments/227_sft.py --force_run '["olmo_sft"]'
"""

import dataclasses
import draccus
from marin.execution.executor import Executor, ExecutorStep, executor_main, this_output_path, output_path_of
from marin.training.training import TrainSFTOnPodConfig, run_levanter_sft
from instruction_datasets import get_instruction_dataset

# Load base config 
training_config = draccus.load(TrainSFTOnPodConfig, open("config/training/standard_sft.yaml"))

# Get instruction dataset
instruction_dataset = get_instruction_dataset("allenai/tulu-v2-sft-mixture")
dataset_path = output_path_of(instruction_dataset)


# Create executor instance to resolve the paths
executor = Executor(prefix="gs://marin-us-central2", executor_info_base_path="gs://marin-us-central2/experiments")
executor.compute_version(instruction_dataset)  # This will populate output_paths
actual_gcs_path = executor.output_paths[instruction_dataset]


print(f"Instruction dataset path: {actual_gcs_path}")

train_step = ExecutorStep(
    name=f"olmo_sft",
    fn=run_levanter_sft,
    config=dataclasses.replace(
        training_config,
        output_path=this_output_path(),
        tpu_type="v4-64",
        # Don't override cache_dir since it's already set correctly in the YAML
        # Just update the chat_train_urls if needed
        chat_train_urls=f"{actual_gcs_path}/**/*.jsonl.gz",
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[
            train_step
        ]
    )