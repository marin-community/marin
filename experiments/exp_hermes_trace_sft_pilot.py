# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace-focused Hermes SFT pilot built on the SmolTalk2 + Nemotron recipe."""

import dataclasses
import math
import re

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.evals import default_sft_eval
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.instruction_datasets import INSTRUCTION_DATASET_NAME_TO_CONFIG, get_instruction_dataset
from experiments.simple_sft_config import SimpleSFTConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize import lm_mixture_data_config

SLUGIFY_PATTERN = re.compile(r"[^a-z0-9]+")
TARGET_EPOCHS = 3
TRAIN_BATCH_SIZE = 2048

# Row counts captured on 2026-04-16 from the Hugging Face dataset page / datasets-server.
TRACE_PILOT_DATASETS = {
    "smoltalk2_smolagents_toolcalling_traces_think": (
        "HuggingFaceTB/smoltalk2/smolagents_toolcalling_traces_think",
        9079,
    ),
    "smoltalk2_hermes_function_calling_v1_no_think": (
        "HuggingFaceTB/smoltalk2/hermes_function_calling_v1_no_think",
        8961,
    ),
    "smoltalk2_xlam_traces_no_think": (
        "HuggingFaceTB/smoltalk2/xlam_traces_no_think",
        59962,
    ),
    "nemotron_v2_chat": ("nvidia/Nemotron-Post-Training-Dataset-v2/chat", 627720),
    "nemotron_v2_code": ("nvidia/Nemotron-Post-Training-Dataset-v2/code", 175000),
    "hermes_glm_5_1": ("lambda/hermes-agent-reasoning-traces/glm-5.1", 7055),
    "hermes_kimi": ("lambda/hermes-agent-reasoning-traces/kimi", 7646),
}


def _slugify(value: str) -> str:
    slug = SLUGIFY_PATTERN.sub("_", value.lower()).strip("_")
    return slug or "dataset"


def create_tokenization_step(dataset_identifier: str, short_name: str) -> ExecutorStep:
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name}_marin_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )


dataset_ids = {
    _slugify(short_name): dataset_identifier for short_name, (dataset_identifier, _count) in TRACE_PILOT_DATASETS.items()
}
mixture_weights = {
    _slugify(short_name): row_count for short_name, (_dataset_identifier, row_count) in TRACE_PILOT_DATASETS.items()
}
tokenized_datasets = {
    short_name: create_tokenization_step(dataset_identifier, short_name)
    for short_name, dataset_identifier in dataset_ids.items()
}

assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

total_examples = sum(mixture_weights.values())
num_train_steps = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

pilot_sft_config = SimpleSFTConfig(
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=num_train_steps,
    learning_rate=1e-5,
    resources=ResourceConfig.with_tpu("v4-128"),
    tokenizer=marin_tokenizer,
    initialize_from_hf="marin-community/marin-8b-base",
    max_seq_len=8192,
    seed=0,
)

pilot_mixture = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    shuffle=True,
    missing_weights_are_validation=True,
)

llama_8b_8k = dataclasses.replace(llama_8b, max_seq_len=8192)

marin_8b_sft_hermes_trace_pilot = default_sft(
    name="marin_8b_sft_hermes_trace_pilot",
    tokenized=pilot_mixture,
    model_config=llama_8b_8k,
    sft_config=pilot_sft_config,
    tags=["llama", "smoltalk2", "nemotron_v2", "hermes_trace", "sft"],
)

marin_8b_sft_hermes_trace_pilot_evals = default_sft_eval(
    marin_8b_sft_hermes_trace_pilot,
    use_levanter_inference=True,
    resource_config=ResourceConfig.with_tpu("v4-8"),
)


if __name__ == "__main__":
    executor_main(steps=[marin_8b_sft_hermes_trace_pilot, *marin_8b_sft_hermes_trace_pilot_evals])
