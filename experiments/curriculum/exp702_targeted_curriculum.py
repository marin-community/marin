"""
Test continued training from checkpoint to support different mixtures.
Issue: https://github.com/stanford-crfm/marin/issues/702
"""

import os
from itertools import chain
from typing import List, Optional
import random
import dataclasses

from levanter.optim import AdamConfig

from experiments.defaults import _prepare_data_config
from experiments.llama import llama_150m, llama_300m, llama_600m, llama_1_9b, llama_8b

from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config, lm_varying_mixture_data_config

from experiments.curriculum.curriculum_stages import train_executor_step, tokenize_train_validation, tokenize_train_validation_sft
from experiments.instruction_datasets import get_instruction_dataset

marin_prefix = os.environ["MARIN_PREFIX"]

llama_8b_1024 = dataclasses.replace(llama_8b, seq_len=1024)

print("Launching experiment from:", marin_prefix)

if 'us-central2' in marin_prefix:
    STACK_PYTHON = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/python/data-{id:05d}-of-00144.parquet"
    STACK_CPP = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/cpp/data-{id:05d}-of-00110.parquet"
    DOLMA_C4 = marin_prefix + "/raw/dolma/v1.7/c4-{id:04d}.json.gz" # different across regions
    DOLMA_TULU_FLAN = marin_prefix + "/raw/dolma/v1.7/tulu_flan-{id:04d}.json.gz" # different across regions
    SPJ6B = marin_prefix + "/raw/SlimPajama-6B-be35b7/b5f90f4/huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/b5f90f4/data/train-{id:05d}-of-00048*.parquet"
    tpu_type = "v4-128"
    job_suffix = "usc2"
elif 'eu-west4' in marin_prefix:
    STACK_PYTHON = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/python/data-{id:05d}-of-00144.parquet"
    STACK_CPP = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/cpp/data-{id:05d}-of-00110.parquet"
    DOLMA_C4 = marin_prefix + "/raw/dolma-c4-split/c4-{id:04d}.json.gz" # different across regions
    DOLMA_TULU_FLAN = marin_prefix + "/raw/dolma-tulu_flan-split/tulu_flan-{id:04d}.json.gz" # different across regions
    SPJ6B = marin_prefix + "/raw/SlimPajama-6B-be35b7/b5f90f4/huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/b5f90f4/data/train-{id:05d}-of-00048*.parquet"
    # tpu_type = "v6e-256"
    tpu_type = "v5litepod-256"
    job_suffix = "euw4"
else:
    raise ValueError("Unknown prefix")

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

spj6b_file_ids = list(range(48))
random.shuffle(spj6b_file_ids)
spj6b_file_ids_stage1 = spj6b_file_ids[0:23]
spj6b_file_ids_stage2 = spj6b_file_ids[23:47]
spj6b_file_ids_validation = spj6b_file_ids[47:48]

flan_file_ids = list(range(66))
random.shuffle(flan_file_ids)
flan_file_ids_stage1 = flan_file_ids[0:32]
flan_file_ids_stage2 = flan_file_ids[32:65]
flan_file_ids_validation = flan_file_ids[65:66]

stack_dedup_stage1_tokenized = tokenize_train_validation(
    train_files=[STACK_PYTHON.format(id=id) for id in stack_file_ids_stage1],
    validation_files=[STACK_PYTHON.format(id=id) for id in stack_file_ids_validation],
    name="stack_dedup_stage1",
    text_key="content"
)

dolma_c4_stage1_tokenized = tokenize_train_validation(
    train_files=[DOLMA_C4.format(id=id) for id in dolma_file_ids_stage1],
    validation_files=[DOLMA_C4.format(id=id) for id in dolma_file_ids_validation],
    name="dolma_c4_stage1",
    text_key="text"
)

stack_cpp_stage1_tokenized = tokenize_train_validation(
    train_files=[STACK_CPP.format(id=id) for id in stack_cpp_file_ids_stage1],
    validation_files=[STACK_CPP.format(id=id) for id in stack_cpp_file_ids_validation],
    name="stack_cpp_stage1",
    text_key="content"
)

spj6b_stage1_tokenized = tokenize_train_validation(
    train_files=[SPJ6B.format(id=id) for id in spj6b_file_ids_stage1],
    validation_files=[SPJ6B.format(id=id) for id in spj6b_file_ids_validation],
    name="spj6b_stage1",
    text_key="text"
)

flan_stage1_tokenized = tokenize_train_validation(
    train_files=[DOLMA_TULU_FLAN.format(id=id) for id in flan_file_ids_stage1],
    validation_files=[DOLMA_TULU_FLAN.format(id=id) for id in flan_file_ids_validation],
    name="flan_stage1",
    text_key="text"
)

stack_dedup_stage2_tokenized = tokenize_train_validation(
    train_files=[STACK_PYTHON.format(id=id) for id in stack_file_ids_stage2],
    validation_files=[STACK_PYTHON.format(id=id) for id in stack_file_ids_validation],
    name="stack_dedup_stage2",
    text_key="content"
)

dolma_c4_stage2_tokenized = tokenize_train_validation(
    train_files=[DOLMA_C4.format(id=id) for id in dolma_file_ids_stage2],
    validation_files=[DOLMA_C4.format(id=id) for id in dolma_file_ids_validation],
    name="dolma_c4_stage2",
    text_key="text"
)

stack_cpp_stage2_tokenized = tokenize_train_validation(
    train_files=[STACK_CPP.format(id=id) for id in stack_cpp_file_ids_stage2],
    validation_files=[STACK_CPP.format(id=id) for id in stack_cpp_file_ids_validation],
    name="stack_cpp_stage2",
    text_key="content"
)

spj6b_stage2_tokenized = tokenize_train_validation(
    train_files=[SPJ6B.format(id=id) for id in spj6b_file_ids_stage2],
    validation_files=[SPJ6B.format(id=id) for id in spj6b_file_ids_validation],
    name="spj6b_stage2",
    text_key="text"
)

flan_stage2_tokenized = tokenize_train_validation(
    train_files=[DOLMA_TULU_FLAN.format(id=id) for id in flan_file_ids_stage2],
    validation_files=[DOLMA_TULU_FLAN.format(id=id) for id in flan_file_ids_validation],
    name="flan_stage2",
    text_key="text"
)

stage_data = {
    "stack_dedup": {
        "stage1": stack_dedup_stage1_tokenized,
        "stage2": stack_dedup_stage2_tokenized,
    },
    "c4": {
        "stage1": dolma_c4_stage1_tokenized,
        "stage2": dolma_c4_stage2_tokenized,
    },
    "stack_cpp": {
        "stage1": stack_cpp_stage1_tokenized,
        "stage2": stack_cpp_stage2_tokenized,
    },
    "spj6b": {
        "stage1": spj6b_stage1_tokenized,
        "stage2": spj6b_stage2_tokenized,
    },
    "flan": {
        "stage1": flan_stage1_tokenized,
        "stage2": flan_stage2_tokenized,
    }
}

def full_training_varying_mixture(
    data1_name: str,
    data2_name: str,
    total_data1_portion: float,
    duration_frac_stage2: float,
    data1_frac_alloc_stage2: float,
    learning_rate: float = 3e-3,
    cooldown_frac: Optional[float] = None,
    schedule_type: str = "cosine",
    model_size: str = "150m",
    num_train_steps: int = 3000,
    num_eval: int = 20,
    version_tag: str = "",
    additional_tags: List[str] = [],
):
    """
    Two-stage training using varying mixture weights, similar to varsched but without checkpointing.
    
    Args:
        data1_name: Name of first dataset (e.g. "stack_dedup", "stack_cpp")
        data2_name: Name of second dataset (e.g. "c4") 
        total_data1_portion: Total portion of data1 across both stages
        duration_frac_stage2: Fraction of total steps to spend in stage 2
        data1_frac_alloc_stage2: Fraction of data1's total portion to allocate to stage 2
        learning_rate: Learning rate to use
        cooldown_frac: Fraction of training to cool down (for linear schedule)
        schedule_type: "cosine" or "linear"
        model_size: Size of model to train
        num_train_steps: Total number of training steps
        num_eval: Number of evaluation steps
        version_tag: Optional version tag for experiment
        additional_tags: Additional tags for experiment
    """
    duration_frac_stage1 = 1 - duration_frac_stage2
    data1_frac_alloc_stage1 = 1 - data1_frac_alloc_stage2

    data1_weight_stage1 = round(total_data1_portion * data1_frac_alloc_stage1 / duration_frac_stage1, 7)
    data1_weight_stage2 = round(total_data1_portion * data1_frac_alloc_stage2 / duration_frac_stage2, 7)

    print('-' * 100)
    print(f"total_data1_portion: {total_data1_portion}")
    print(f"duration_frac_stage1: {duration_frac_stage1}, data1_frac_alloc_stage1: {data1_frac_alloc_stage1}, data1_weight_stage1: {data1_weight_stage1}")
    print(f"duration_frac_stage2: {duration_frac_stage2}, data1_frac_alloc_stage2: {data1_frac_alloc_stage2}, data1_weight_stage2: {data1_weight_stage2}")

    assert 0 <= data1_weight_stage1 <= 1, f"data1_weight_stage1: {data1_weight_stage1}"
    assert 0 <= data1_weight_stage2 <= 1, f"data1_weight_stage2: {data1_weight_stage2}"

    # Calculate transition point
    num_sequences = num_train_steps * 1024  # batch size
    transition_seq_idx = int(duration_frac_stage1 * num_sequences)
    # Ensure sequence index is multiple of 2048
    assert transition_seq_idx % 2048 == 0, f"transition_seq_idx: {transition_seq_idx}"

    # Create data config with varying mixture
    data_config = lm_varying_mixture_data_config(
        components={
            data1_name: stage_data[data1_name]["stage1"],
            data2_name: stage_data[data2_name]["stage1"]
        },
        weights_list=[
            (0, {data1_name: data1_weight_stage1, data2_name: 1 - data1_weight_stage1}),  # Stage 1
            (transition_seq_idx, {data1_name: data1_weight_stage2, data2_name: 1 - data1_weight_stage2})  # Stage 2
        ],
    )

    pretraining_data = _prepare_data_config(data_config, use_default_validation=True)

    # Configure model and training
    model = {
        "150m": llama_150m,
        "300m": llama_300m,
        "600m": llama_600m,
        "1_9b": llama_1_9b,
        # "8b": llama_8b,
        "8b_1024": llama_8b_1024,
    }[model_size]

    weight_decay = 0.1
    steps_per_eval = num_train_steps // num_eval
    name_prefix = f"{data1_name}-{data2_name}-onevs{data1_frac_alloc_stage2}-dur{duration_frac_stage2}-{job_suffix}-{model_size}"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
        )
        name_prefix += f"-{schedule_type}-{cooldown_frac}"
    elif schedule_type == "cosine":
        optimizer_config = None
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")
    
    if num_train_steps != 3000:
        name_prefix += f"-{num_train_steps // 1000}B"

    train_step = train_executor_step(
        name=f"{name_prefix}{version_tag}",
        pretraining_data=pretraining_data,
        model=model,
        model_checkpoint=None,
        train_batch_size=1024,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export_list=[],
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags + [job_suffix],
    )

    return [train_step]

############################################################

if __name__ == "__main__":
    # stage_pairs = [
    #     full_training_varying_mixture(
    #         data1_name="flan",
    #         data2_name="c4",
    #         total_data1_portion=0.005,
    #         duration_frac_stage2=duration_frac_stage2,
    #         data1_frac_alloc_stage2=1.0,
    #         schedule_type="linear",
    #         cooldown_frac=0.05,
    #         model_size=model_size,
    #         num_eval=1,
    #         num_train_steps=100,
    #         additional_tags=["debug-extension-modules"],
    #         version_tag="-debug-v7"
    #     )
    #     for model_size in ["150m"]
    #     for duration_frac_stage2 in [0.4]
    # ]

    learning_rate_dict = {
        "150m": 3e-3,
        "300m": 3e-3,
        "600m_0.001": 1e-3,
        "600m_0.003": 3e-3,
        "1_9b": 3e-4,
        "8b_1024": 3e-4,
    }

    chinchilla_steps = {
        "150m": 3000,
        "300m": 6000,
        "600m_0.001": 12000,
        "600m_0.003": 12000,
        "1_9b": 38000,
        "8b_1024": 160000,
    }

    def version_tag(lr):
        return f"-lr{lr}" if lr != 3e-3 else ""
    
    def correct_model_size(model_size):
        if model_size == "600m_0.003" or model_size == "600m_0.001":
            return "600m"
        return model_size

    # Model scaling

    # stage_pairs = [
    #     full_training_varying_mixture(
    #         data1_name="flan",
    #         data2_name="c4",
    #         total_data1_portion=0.005,
    #         duration_frac_stage2=duration_frac_stage2,
    #         data1_frac_alloc_stage2=1.0,
    #         schedule_type="linear",
    #         cooldown_frac=0.05,
    #         model_size=correct_model_size(model_size),
    #         num_train_steps=3000,
    #         learning_rate=learning_rate_dict[model_size],
    #         additional_tags=["flan-c4-eu-model-scaling"],
    #         version_tag=version_tag(learning_rate_dict[model_size]) + "-v1"
    #     )
    #     for model_size in ["600m_0.001"]
    #     for duration_frac_stage2 in [0.4]
    # ]

    # Chinchilla scaling 

    # stage_pairs = [
    #     full_training_varying_mixture(
    #         data1_name="flan",
    #         data2_name="c4",
    #         total_data1_portion=0.005,
    #         duration_frac_stage2=duration_frac_stage2,
    #         data1_frac_alloc_stage2=1.0,
    #         schedule_type="linear",
    #         cooldown_frac=0.05,
    #         model_size=correct_model_size(model_size),
    #         num_train_steps=chinchilla_steps[model_size],
    #         learning_rate=learning_rate_dict[model_size],
    #         additional_tags=["flan-c4-eu-chinchilla-model-scaling"],
    #         version_tag=version_tag(learning_rate_dict[model_size])
    #     )
    #     for model_size in ["1_9b"]
    #     for duration_frac_stage2 in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    # ]

    # Token scaling

    stage_pairs = [
        full_training_varying_mixture(
            data1_name="flan",
            data2_name="c4",
            total_data1_portion=0.005,
            duration_frac_stage2=duration_frac_stage2,
            data1_frac_alloc_stage2=1.0,
            schedule_type="linear",
            cooldown_frac=0.05,
            model_size=correct_model_size(model_size),
            num_train_steps=num_train_steps,
            learning_rate=learning_rate_dict[model_size],
            additional_tags=["flan-c4-eu-token-scaling"],
            version_tag=version_tag(learning_rate_dict[model_size])
        )
        for model_size in ["600m_0.001"]
        for num_train_steps in [48000]
        for duration_frac_stage2 in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    ]

    # steps = list(chain(*stage_pairs))

    # Confirm optimal learning rate schedule

    # stage_pairs = [
    #     full_training_varying_mixture(
    #         data1_name="flan",
    #         data2_name="c4",
    #         total_data1_portion=0.005,
    #         duration_frac_stage2=duration_frac_stage2,
    #         data1_frac_alloc_stage2=1.0,
    #         schedule_type="linear",
    #         cooldown_frac=cooldown_frac,
    #         model_size=correct_model_size(model_size),
    #         num_train_steps=num_train_steps,
    #         learning_rate=learning_rate_dict[model_size],
    #         additional_tags=["flan-c4-eu-confirming-0.05-lr-decay"],
    #         version_tag=version_tag(learning_rate_dict[model_size])
    #     )
    #     for model_size in ["600m_0.003"]
    #     for num_train_steps in [1200]
    #     for cooldown_frac in [0.02, 0.05, 0.1, 0.2]
    #     for duration_frac_stage2 in [0.02, 0.05, 0.1, 0.2]
    # ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description=f"Test training with varying mixtures",
    )
