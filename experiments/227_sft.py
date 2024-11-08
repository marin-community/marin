"""
python marin/run/ray_run.py --env_vars HF_TOKEN -- python experiments/227_sft.py --force_run '["olmo_sft"]'
"""
import jmp
import dataclasses
import draccus
from marin.execution.executor import Executor, ExecutorStep, executor_main, this_output_path, output_path_of
from marin.training.training import TrainSFTOnPodConfig, run_levanter_sft
from levanter.checkpoint import CheckpointerConfig
from instruction_datasets import get_instruction_dataset
from datetime import timedelta

from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

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
        tpu_type="v4-128",
        chat_train_urls=[f"{actual_gcs_path}/**/*.jsonl.gz"],
        # Modify the nested trainer config by creating a new one
        trainer=dataclasses.replace(
            training_config.trainer,  # Start with the existing trainer config
            tracker=WandbConfig(
                project="marin",
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=128,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=25000)],
            ),
            initialize_from="gs://levanter-checkpoints/marin/olmoish7b_v4_1024_0627/dlwh_7b0627/step-510000/",  # Add initialize_from here
        ),
    ),
)
# train_step = ExecutorStep(
#     name=f"olmo_sft",
#     fn=run_levanter_sft,
#     config=dataclasses.replace(
#         training_config,
#         output_path=this_output_path(),
#         tpu_type="v4-64",
#         # Don't override cache_dir since it's already set correctly in the YAML
#         # Just update the chat_train_urls if needed
#         chat_train_urls=[f"{actual_gcs_path}/**/*.jsonl.gz"],
#         trainer=TrainerConfig(
#                 tracker=WandbConfig(
#                     project="marin",
#                 ),
#                 mp=jmp.get_policy("p=f32,c=bfloat16"),
#                 train_batch_size=128,
#                 checkpointer=CheckpointerConfig(
#                     save_interval=timedelta(minutes=10),
#                     keep=[dict(every=25000)],
#                 ),
#             ),
#             initialize_from="gs://levanter-checkpoints/marin/olmoish7b_v4_1024_0627/dlwh_7b0627/step-510000/",
#     ),
# )

if __name__ == "__main__":
    executor_main(
        steps=[
            train_step
        ]
    )