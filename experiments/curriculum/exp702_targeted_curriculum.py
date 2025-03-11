"""
Test continued training from checkpoint to support different mixtures.
Issue: https://github.com/stanford-crfm/marin/issues/702
"""

import os
import numpy as np
from itertools import chain
from typing import List, Optional
import random
import dataclasses

from levanter.optim import AdamConfig

from experiments.defaults import _prepare_data_config
from experiments.llama import llama_150m, llama_300m, llama_600m, llama_1_4b, llama_1_9b, llama_8b
from experiments.pretraining_datasets import dclm_baseline
from experiments.midtraining_datasets import finemath_3_plus_tokenized

from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config, lm_varying_mixture_data_config
from experiments.defaults import default_tokenize

from experiments.curriculum.curriculum_stages import train_executor_step, tokenize_train_validation, tokenize_two_stages
from experiments.instruction_datasets import get_instruction_dataset
from experiments.pretraining_datasets import slimpajama_6b, starcoderdata
from experiments.llama import llama3_tokenizer

marin_prefix = os.environ["MARIN_PREFIX"]

llama_150m_4096 = dataclasses.replace(llama_150m, seq_len=4096)
llama_600m_4096 = dataclasses.replace(llama_600m, seq_len=4096)
llama_1_4b_1024 = dataclasses.replace(llama_1_4b, seq_len=1024)
llama_1_9b_1024 = dataclasses.replace(llama_1_9b, seq_len=1024)
llama_8b_1024 = dataclasses.replace(llama_8b, seq_len=1024)

print("Launching experiment from:", marin_prefix)

if 'us-central2' in marin_prefix:
    STACK_PYTHON = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/python/data-{id:05d}-of-00144.parquet"
    STACK_CPP = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/cpp/data-{id:05d}-of-00110.parquet"
    DOLMA_C4 = marin_prefix + "/raw/dolma/v1.7/c4-{id:04d}.json.gz" # different across regions
    DOLMA_TULU_FLAN = marin_prefix + "/raw/dolma/v1.7/tulu_flan-{id:04d}.json.gz" # different across regions
    SPJ6B = marin_prefix + "/raw/SlimPajama-6B-be35b7/b5f90f4/huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/b5f90f4/data/train-{id:05d}-of-00048*.parquet"
    WIKI = marin_prefix + "/raw/dolmino-mix-1124-157960/bb54cab/data/wiki/wiki-{id:04d}.json.gz"
    tpu_type = "v4-128"
    region_suffix = "usc2"
elif 'eu-west4' in marin_prefix:
    STACK_PYTHON = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/python/data-{id:05d}-of-00144.parquet"
    STACK_CPP = marin_prefix + "/raw/the-stack-dedup-4ba450/17cad72/data/cpp/data-{id:05d}-of-00110.parquet"
    DOLMA_C4 = marin_prefix + "/raw/dolma-c4-split/c4-{id:04d}.json.gz" # different across regions
    DOLMA_TULU_FLAN = marin_prefix + "/raw/dolma-tulu_flan-split/tulu_flan-{id:04d}.json.gz" # different across regions
    SPJ6B = marin_prefix + "/raw/SlimPajama-6B-be35b7/b5f90f4/huggingface.co/datasets/DKYoon/SlimPajama-6B/resolve/b5f90f4/data/train-{id:05d}-of-00048*.parquet"
    WIKI = None
    # tpu_type = "v6e-256"
    tpu_type = "v5litepod-256"
    region_suffix = "euw4"
else:
    raise ValueError("Unknown prefix")

# TODO: Remove stage 2 tokenization

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

wiki_file_ids_stage1 = [0]
wiki_file_ids_stage2 = []
wiki_file_ids_validation = [1]

stack_dedup_stage1_tokenized, stack_dedup_stage2_tokenized = tokenize_two_stages(
    base_file_path=STACK_PYTHON,
    train_file_ids_stage1=stack_file_ids_stage1,
    train_file_ids_stage2=stack_file_ids_stage2,
    validation_file_ids=stack_file_ids_validation,
    name="stack_dedup",
    text_key="content"
)

dolma_c4_stage1_tokenized, dolma_c4_stage2_tokenized = tokenize_two_stages(
    base_file_path=DOLMA_C4,
    train_file_ids_stage1=dolma_file_ids_stage1,
    train_file_ids_stage2=dolma_file_ids_stage2,
    validation_file_ids=dolma_file_ids_validation,
    name="dolma_c4",
    text_key="text"
)

stack_cpp_stage1_tokenized, stack_cpp_stage2_tokenized = tokenize_two_stages(
    base_file_path=STACK_CPP,
    train_file_ids_stage1=stack_cpp_file_ids_stage1,
    train_file_ids_stage2=stack_cpp_file_ids_stage2,
    validation_file_ids=stack_cpp_file_ids_validation,
    name="stack_cpp",
    text_key="content"
)

spj6b_stage1_tokenized, spj6b_stage2_tokenized = tokenize_two_stages(
    base_file_path=SPJ6B,
    train_file_ids_stage1=spj6b_file_ids_stage1,
    train_file_ids_stage2=spj6b_file_ids_stage2,
    validation_file_ids=spj6b_file_ids_validation,
    name="spj6b",
    text_key="text"
)

flan_stage1_tokenized, flan_stage2_tokenized = tokenize_two_stages(
    base_file_path=DOLMA_TULU_FLAN,
    train_file_ids_stage1=flan_file_ids_stage1,
    train_file_ids_stage2=flan_file_ids_stage2,
    validation_file_ids=flan_file_ids_validation,
    name="flan",
    text_key="text"
)

if WIKI is not None:
    wiki_stage1_tokenized, wiki_stage2_tokenized = tokenize_two_stages(
        base_file_path=WIKI,
        train_file_ids_stage1=wiki_file_ids_stage1,
        train_file_ids_stage2=wiki_file_ids_stage2,
        validation_file_ids=wiki_file_ids_validation,
        name="wiki",
        text_key="text"
    )
else:
    wiki_stage1_tokenized = None
    wiki_stage2_tokenized = None

dclm_tokenized = dataclasses.replace(
    default_tokenize(
        name="dclm_baseline",
        dataset=dclm_baseline,
        tokenizer=llama3_tokenizer,
    ),
    # make consistent with path in eu (they differ b/c of the cd)
    override_output_path="tokenized/dclm_baseline-0206f1/",
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
    },
    "wiki": {
        "stage1": wiki_stage1_tokenized,
        "stage2": wiki_stage2_tokenized,
    },
    "dclm": {
        "stage1": dclm_tokenized,
    },
    "finemath": {
        "stage1": finemath_3_plus_tokenized,
    }
}

model_dict = {
    "150m": llama_150m,
    "150m_4096": llama_150m_4096,
    "300m": llama_300m,
    "600m": llama_600m,
    "600m_4096": llama_600m_4096,
    "1_4b_1024": llama_1_4b_1024,
    # "1_9b": llama_1_9b,
    "1_9b_1024": llama_1_9b_1024,
    "8b_1024": llama_8b_1024,
}

def get_pretraining_data(
    data1_name: str,
    data2_name: str,
    data1_weight_stage1: float,
    data1_weight_stage2: float,
    transition_seq_idx: int,
    num_data1_sequences: int,
    num_data1_repetitions: Optional[int] = None,
    num_validation_sequences: Optional[int] = 10 * 1024,
):
    # Create data config with varying mixture
    # Only using stage 1 because we're not using checkpointing
    components = {
        data1_name: stage_data[data1_name]["stage1"],
        data2_name: stage_data[data2_name]["stage1"]
    }

    if transition_seq_idx == 0.0:
        weights_list = [
            (0, {data1_name: data1_weight_stage2, data2_name: 1 - data1_weight_stage2}),  # Stage 2
        ]
    else:
        weights_list = [
            (0, {data1_name: data1_weight_stage1, data2_name: 1 - data1_weight_stage1}),  # Stage 1
            (transition_seq_idx, {data1_name: data1_weight_stage2, data2_name: 1 - data1_weight_stage2})  # Stage 2
        ]

    data_config = lm_varying_mixture_data_config(
        components=components,
        weights_list=weights_list,
        max_sequences_dict=None if num_data1_repetitions is None else {
            data1_name: num_data1_sequences
        },
        num_validation_sequences_dict=None if num_validation_sequences is None else {
            data1_name: num_validation_sequences,
            data2_name: num_validation_sequences
        }
    )

    pretraining_data = _prepare_data_config(data_config, use_default_validation=True)

    return pretraining_data

def full_training_varying_mixture(
    data1_name: str,
    data2_name: str,
    total_data1_portion: float,
    duration_frac_stage2: float,
    data1_frac_alloc_stage2: float,
    learning_rate: float = 3e-3,
    sft_learning_rate: float = None,
    cooldown_frac: Optional[float] = None,
    schedule_type: str = "cosine",
    model_size: str = "150m",
    num_train_steps: int = 3000,
    sft_steps: int = None,
    num_eval: int = 20,
    num_data1_repetitions: int = 1,
    version_tag: str = "",
    additional_tags: List[str] = [],
    num_lm_eval_harness: Optional[int] = None,
    tpu_type: str = tpu_type,
    batch_size: int = 1024,
    min_lr_ratio: float = 0.1,
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
    num_sequences = num_train_steps * batch_size
    num_data1_sequences = int(num_sequences * total_data1_portion)

    duration_frac_stage1 = round(1 - duration_frac_stage2, 7)
    data1_frac_alloc_stage1 = round(1 - data1_frac_alloc_stage2, 7)

    if duration_frac_stage1 == 0.0:
        data1_weight_stage1 = 0.0
    else:
        data1_weight_stage1 = round(total_data1_portion * data1_frac_alloc_stage1 * num_data1_repetitions / duration_frac_stage1, 7)
    data1_weight_stage2 = round(total_data1_portion * data1_frac_alloc_stage2 * num_data1_repetitions / duration_frac_stage2, 7)

    print('-' * 100)
    print(f"total_data1_portion: {total_data1_portion}")
    print(f"duration_frac_stage1: {duration_frac_stage1}, data1_frac_alloc_stage1: {data1_frac_alloc_stage1}, data1_weight_stage1: {data1_weight_stage1}")
    print(f"duration_frac_stage2: {duration_frac_stage2}, data1_frac_alloc_stage2: {data1_frac_alloc_stage2}, data1_weight_stage2: {data1_weight_stage2}")

    assert 0 <= data1_weight_stage1 <= 1, f"data1_weight_stage1: {data1_weight_stage1}"
    assert 0 <= data1_weight_stage2 <= 1, f"data1_weight_stage2: {data1_weight_stage2}"

    # Configure model and training
    model = model_dict[model_size]

    # Calculate transition point
    transition_seq_idx = int(duration_frac_stage1 * num_sequences)
    # Ensure sequence index is multiple of 2048
    assert transition_seq_idx % 2048 == 0, f"transition_seq_idx: {transition_seq_idx}"

    pretraining_data = get_pretraining_data(
        data1_name=data1_name,
        data2_name=data2_name,
        data1_weight_stage1=data1_weight_stage1,
        data1_weight_stage2=data1_weight_stage2,
        transition_seq_idx=transition_seq_idx,
        num_data1_sequences=num_data1_sequences,
        num_data1_repetitions=num_data1_repetitions,
    )

    weight_decay = 0.1
    steps_per_eval = num_train_steps // num_eval
    steps_per_eval_task = None if num_lm_eval_harness is None else num_train_steps // num_lm_eval_harness
    epochs_tag = f"-r{num_data1_repetitions}" if num_data1_repetitions is not None and num_data1_repetitions > 1 else ""
    
    if data1_frac_alloc_stage2 == 1.0:
        name_prefix = f"{data1_name}{epochs_tag}-{data2_name}-dur{duration_frac_stage2}-{model_size}"
    else:
        alloc_tag = f"-a{data1_frac_alloc_stage2}" if data1_frac_alloc_stage2 != 1.0 else ""
        name_prefix = f"{data1_name}{epochs_tag}-{data2_name}-dur{duration_frac_stage2}{alloc_tag}-{model_size}"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
            min_lr_ratio=min_lr_ratio,
        )
        name_prefix += f"-{schedule_type}-{cooldown_frac}"
    elif schedule_type == "cosine":
        optimizer_config = None
    elif schedule_type == "sft":
        assert sft_learning_rate is not None, "sft_learning_rate must be provided for sft schedule"
        assert sft_steps is not None, "sft_steps must be provided for sft schedule"
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            min_lr_ratio=0.0,
            weight_decay=weight_decay,
            lr_schedule="linear",
            decay=cooldown_frac,
            sft_learning_rate=sft_learning_rate,
            sft_steps=sft_steps,
        )
        name_prefix += f"-{schedule_type}-{sft_learning_rate},{sft_steps}"
    else:
        raise ValueError(f"Invalid schedule type: {schedule_type}")
    
    if num_train_steps != 3000:
        billion_tokens = round(num_train_steps / 1000.0, 7)
        name_prefix += f"-{int(billion_tokens)}B" if billion_tokens.is_integer() else f"-{billion_tokens}B"

    if total_data1_portion != 0.005:
        name_prefix += f"-ra{total_data1_portion}"

    assert len(f"{name_prefix}{version_tag}") <= 64, f"{name_prefix}{version_tag} is too long"

    train_step = train_executor_step(
        name=f"{name_prefix}{version_tag}",
        pretraining_data=pretraining_data,
        model=model,
        model_checkpoint=None,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export_list=[],
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags + [region_suffix],
        steps_per_eval_task=steps_per_eval_task,
    )

    return [train_step]
