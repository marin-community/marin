"""
python marin/run/ray_run.py --env_vars HF_TOKEN -- python experiments/227_sft.py --force_run '["olmo_sft"]'

export to HF format with
export HF_TOKEN=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  && python -m levanter.main.export_lm_to_hf
--output_dir "gs://marin-us-central2/checkpoints/tulu_sft_3epshf/"
--checkpoint_path "gs://marin-us-central2/checkpoints/tulu_sft_3eps-4e30ee/checkpoints/step-938/"
--config_path levanter/config/llama_sft_hf_ckpt.yaml
"""

import dataclasses
from datetime import timedelta

import draccus
import jmp
from instruction_datasets import get_instruction_dataset
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig

from marin.execution.executor import Executor, ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.tokenize import TokenizeConfig, levanter_tokenize_sft
from marin.training.training import TrainSFTOnPodConfig, run_levanter_sft

# Load base config
training_config = draccus.load(TrainSFTOnPodConfig, open("config/training/standard_sft.yaml"))

# Get instruction dataset
instruction_dataset = get_instruction_dataset("allenai/tulu-v2-sft-mixture")
dataset_path = output_path_of(instruction_dataset)


# Create executor instance to resolve the paths
executor = Executor(prefix="gs://marin-us-central2", executor_info_base_path="gs://marin-us-central2/experiments")
executor.compute_version(instruction_dataset)  # This will populate output_paths
actual_gcs_path = executor.output_paths[instruction_dataset]

# Add tokenization step
tokenize_step = ExecutorStep(
    name="tokenized/tulu_sft_3eps",
    fn=levanter_tokenize_sft,
    config=TokenizeConfig(
        train_paths=[f"{actual_gcs_path}/**/*.jsonl.gz"],
        validation_paths=[],
        cache_path=training_config.supervised_data.cache_dir,
        tokenizer=training_config.tokenizer,
        input_field=training_config.input_role,
        output_field=training_config.output_role,  # Or whatever tokenizer you're using
        seq_len=training_config.max_seq_len,
    ),
    description="Tokenize chat SFT data",
)

train_step = ExecutorStep(
    name="checkpoints/tulu_sft_3eps",
    fn=run_levanter_sft,
    config=dataclasses.replace(
        training_config,
        output_path=this_output_path(),
        tpu_type="v4-128",
        # number of epochs over the dataset set to reproduce Olmo SFT
        epoch=3,
        supervised_data=dataclasses.replace(
            training_config.supervised_data,
            cache_dir=training_config.supervised_data.cache_dir,
        ),
        # Modify the nested trainer config by creating a new one
        trainer=dataclasses.replace(
            training_config.trainer,  # Start with the existing trainer config
            tracker=WandbConfig(
                project="marin",
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=1024,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=25000)],
            ),
            initialize_from="gs://levanter-checkpoints/marin/olmoish7b_v4_1024_0627/dlwh_7b0627/step-510000/",
        ),
    ),
)



if __name__ == "__main__":
    executor_main(steps=[tokenize_step, train_step])
