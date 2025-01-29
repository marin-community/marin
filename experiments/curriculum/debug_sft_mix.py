"""
Test continued training from checkpoint to support different mixtures.
Issue: https://github.com/stanford-crfm/marin/issues/702
"""

from itertools import chain
from typing import List, Optional
import random

from levanter.optim import AdamConfig

from experiments.defaults import _prepare_data_config
from experiments.llama import llama_150m, llama_300m, llama_600m

from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config

from experiments.curriculum.curriculum_stages import train_executor_step, tokenize_train_validation, tokenize_train_validation_sft
from experiments.instruction_datasets import get_instruction_dataset

BASE_DIR_STACK_PYTHON = "gs://marin-us-central2/raw/the-stack-dedup-4ba450/17cad72/data/python"
BASE_DIR_STACK_CPP = "gs://marin-us-central2/raw/the-stack-dedup-4ba450/17cad72/data/cpp"
BASE_DIR_DOLMA = "gs://marin-us-central2/raw/dolma/v1.7"
BASE_DIR_TULU3 = "gs://marin-us-central2/documents/allenai--tulu-3-sft-mixture-0a99cb/data"

# randomly split stack python parquet files into two seperate groups
stack_file_ids = list(range(144))
random.seed(42)
random.shuffle(stack_file_ids)
stack_file_ids_stage1 = stack_file_ids[0:72]
stack_file_ids_stage2 = stack_file_ids[72:143]
stack_file_ids_validation = stack_file_ids[143:144]

# randomly split dolma c4 json.gz files into two seperate groups
dolma_file_ids = list(range(171))
random.shuffle(dolma_file_ids)
dolma_file_ids_stage1 = dolma_file_ids[0:85]
dolma_file_ids_stage2 = dolma_file_ids[85:170]
dolma_file_ids_validation = dolma_file_ids[170:171]

# randomly split stack cpp parquet files into two seperate groups
stack_cpp_file_ids = list(range(110))
random.shuffle(stack_cpp_file_ids)
stack_cpp_file_ids_stage1 = stack_cpp_file_ids[0:55]
stack_cpp_file_ids_stage2 = stack_cpp_file_ids[55:109]
stack_cpp_file_ids_validation = stack_cpp_file_ids[109:110]

tulu_file_ids = list(range(6))
random.shuffle(tulu_file_ids)
tulu_file_ids_stage1 = tulu_file_ids[0:1]
tulu_file_ids_stage2 = tulu_file_ids[1:5]
tulu_file_ids_validation = tulu_file_ids[5:6]

stack_dedup_stage1_tokenized = tokenize_train_validation(
    train_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_stage1],
    validation_files=[f"{BASE_DIR_STACK_PYTHON}/data-{id:05d}-of-00144.parquet" for id in stack_file_ids_validation],
    name="stack_dedup_stage1",
    text_key="content"
)

tulu_stage1_tokenized = tokenize_train_validation_sft(
    train_files=[f"{BASE_DIR_TULU3}/train-{id:05d}-of-00006/**.jsonl.gz" for id in tulu_file_ids_stage1],
    validation_files=[f"{BASE_DIR_TULU3}/train-{id:05d}-of-00006/**.jsonl.gz" for id in tulu_file_ids_validation],
    # train_files=["gs://marin-us-central2/documents/allenai--tulu-3-sft-mixture-0a99cb/data/train-00000-of-00006/shard_00000.jsonl.gz"],
    # validation_files=["gs://marin-us-central2/documents/allenai--tulu-3-sft-mixture-0a99cb/data/train-00001-of-00006/shard_00000.jsonl.gz"],
    name="tulu_stage1_debug4",
)

stage_data = {
    "stack_dedup": {
        "stage1": stack_dedup_stage1_tokenized,
    },    
    "tulu": {
        "stage1": tulu_stage1_tokenized,
    },
}

def full_training_stage_baseline_sweep(data1_name, data2_name, learning_rate, schedule_type, cooldown_frac=None, num_train_steps=3000, model_size="150m", additional_tags=[], data1_portion=0.005, train_batch_size=1024, version_tag=None):
    data_config = lm_mixture_data_config(
        components={data1_name: stage_data[data1_name]["stage1"], data2_name: stage_data[data2_name]["stage1"]},
        weights={data1_name: data1_portion, data2_name: 1 - data1_portion},
    )

    pretraining_data, evaluation_data = _prepare_data_config(data_config, use_default_validation=True, use_default_evaluation=True)

    # Construct executor steps for training

    tpu_type="v4-128"
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
    }[model_size]
    weight_decay=0.1
    steps_per_eval=num_train_steps // 20
    steps_per_export=num_train_steps // 2
    name_prefix = f"debug-sft-mix-{version_tag}"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )

        schedule_type = f"linear-{cooldown_frac}"
    elif schedule_type == "cosine":
        optimizer_config = None
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")
    
    train_step = train_executor_step(
        name=name_prefix,
        pretraining_data=pretraining_data,
        evaluation_data=evaluation_data,
        model=model,
        model_checkpoint=None,
        train_batch_size=train_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export_list=[steps_per_export],
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags,
    )

    return [train_step]

############################################################

if __name__ == "__main__":
    stage_pairs = [
        full_training_stage_baseline_sweep(
            data1_name="tulu",
            data2_name="stack_dedup",
            data1_portion=0.5,
            learning_rate=1e-3,
            schedule_type="linear",
            cooldown_frac=0.05,
            num_train_steps=1000,
            model_size="150m",
            additional_tags=["debug-sft-mix"],
            version_tag="v2",
        )
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description=f"Test training with varying mixtures",
    )
