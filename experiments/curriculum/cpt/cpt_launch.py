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
from experiments.pretraining_datasets import dclm_baseline, slimpajama
from experiments.midtraining_datasets import finemath_3_plus_tokenized, pubmed_abstracts_tokenized, open_web_math_tokenized

from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_mixture_data_config, lm_varying_mixture_data_config
from experiments.defaults import default_tokenize

from experiments.curriculum.cpt.cpt_train_config import cpt_train_executor_step
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
    tpu_type = "v5litepod-128"
    region_suffix = "euw4"
else:
    raise ValueError("Unknown prefix")

# finemath_3_plus_tokenized
dclm_tokenized = dataclasses.replace(
    default_tokenize(
        name="dclm_baseline",
        dataset=dclm_baseline,
        tokenizer=llama3_tokenizer,
    ),
    # make consistent with path in eu (they differ b/c of the cd)
    override_output_path="tokenized/dclm_baseline-0206f1/",
)

data_dict = {
    "finemath": finemath_3_plus_tokenized,
    "dclm": dclm_tokenized,
    "pubmed": pubmed_abstracts_tokenized,
    "open-web-math": open_web_math_tokenized,
    "slimpajama": slimpajama,
}

def get_cpt_data(
    data1_name: str,
    data2_name: str,
    data1_weight_stage1: float,
    data1_weight_stage2: float,
    transition_seq_idx: int,
    num_data1_sequences: int,
    num_data1_repetitions: Optional[int] = None,
    num_validation_sequences: Optional[int] = 10 * 1024,
):
    
    assert np.isclose(data1_weight_stage1, 0.0), "data1_weight_stage1 must be 0.0 for now"
    # Create data config with varying mixture
    components = {
        data1_name: data_dict[data1_name],
        data2_name: data_dict[data2_name]
    }

    if transition_seq_idx == 0.0:
        weights_list = [
            (0, {data1_name: data1_weight_stage2, data2_name: 1 - data1_weight_stage2}),  # Stage 2
        ]
    else:
        assert False
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

def full_cpt_varying_mixture(
    data1_name: str,
    data2_name: str,
    total_data1_portion: float,
    duration_frac_stage2: float,
    data1_frac_alloc_stage2: float,
    learning_rate: float = 3e-5,
    sft_learning_rate: float = None,
    cooldown_frac: Optional[float] = None,
    schedule_type: str = "cosine",
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
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
    warmup_steps: float = 0.01,
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
    num_data1_sequences = int(num_sequences * total_data1_portion / num_data1_repetitions)

    duration_frac_stage1 = round(1 - duration_frac_stage2, 7)
    data1_frac_alloc_stage1 = round(1 - data1_frac_alloc_stage2, 7)

    assert duration_frac_stage1 == 0.0, "duration_frac_stage1 must be 0.0 for now"
    assert data1_frac_alloc_stage2 == 1.0, "data1_frac_alloc_stage2 must be 1.0 for now"

    if duration_frac_stage1 == 0.0:
        data1_weight_stage1 = 0.0
    else:
        assert False
        data1_weight_stage1 = round(total_data1_portion * data1_frac_alloc_stage1 * num_data1_repetitions / duration_frac_stage1, 7)

    # data1_weight_stage2 = round(total_data1_portion * data1_frac_alloc_stage2 / duration_frac_stage2, 7)
    data1_weight_stage2 = round(total_data1_portion, 7)

    print('-' * 100)
    print(f"total_data1_portion: {total_data1_portion}")
    print(f"duration_frac_stage1: {duration_frac_stage1}, data1_frac_alloc_stage1: {data1_frac_alloc_stage1}, data1_weight_stage1: {data1_weight_stage1}")
    print(f"duration_frac_stage2: {duration_frac_stage2}, data1_frac_alloc_stage2: {data1_frac_alloc_stage2}, data1_weight_stage2: {data1_weight_stage2}")

    assert 0 <= data1_weight_stage1 <= 1, f"data1_weight_stage1: {data1_weight_stage1}"
    assert 0 <= data1_weight_stage2 <= 1, f"data1_weight_stage2: {data1_weight_stage2}"

    # Calculate transition point
    transition_seq_idx = int(duration_frac_stage1 * num_sequences)
    # Ensure sequence index is multiple of 2048
    assert transition_seq_idx == 0, f"transition_seq_idx: {transition_seq_idx}"
    assert transition_seq_idx % 2048 == 0, f"transition_seq_idx: {transition_seq_idx}"

    pretraining_data = get_cpt_data(
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
    
    # if data1_frac_alloc_stage2 == 1.0:
    #     name_prefix = f"{data1_name}{epochs_tag}-{data2_name}-dur{duration_frac_stage2}-{model_name}"
    # else:
    #     alloc_tag = f"-a{data1_frac_alloc_stage2}" if data1_frac_alloc_stage2 != 1.0 else ""
    #     name_prefix = f"{data1_name}{epochs_tag}-{data2_name}-dur{duration_frac_stage2}{alloc_tag}-{model_name}"
    name_prefix = f"{data1_name}-{epochs_tag}-{data2_name}-llama3.1-8b"

    if schedule_type == "linear":
        optimizer_config = AdamConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lr_schedule=schedule_type,
            decay=cooldown_frac,
            min_lr_ratio=min_lr_ratio,
            warmup=warmup_steps,
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

    name_prefix += f"-ra{total_data1_portion}"

    assert len(f"{name_prefix}{version_tag}") <= 64, f"{name_prefix}{version_tag} is too long"

    train_step = cpt_train_executor_step(
        name=f"{name_prefix}{version_tag}",
        pretraining_data=pretraining_data,
        model_name=model_name,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        steps_per_eval=steps_per_eval,
        steps_per_export_list=[num_train_steps - 2],
        tpu_type=tpu_type,
        optimizer_config=optimizer_config,
        additional_tags=additional_tags + [region_suffix],
        steps_per_eval_task=steps_per_eval_task,
        warmup_steps=warmup_steps,
    )

    return [train_step]
