# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SmolTalk2 + Nemotron v2 instruction-tuning mixture.

This experiment mirrors ``exp808_sft_mixture`` but restricts the mixture to the
SmolTalk2 SFT release and NVIDIA's Nemotron Post Training Dataset v2. Mixture
weights were looked up dynamically from the Hugging Face datasets-server ``/size``
endpoint so that each split is weighted by its document count. (2025-11-07)

TODO: this should probably be tokens instead of doc counts.
"""
import math
import re

from experiments.evals.resource_configs import TPU_V4_8
from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.evals.evals import default_sft_eval
from experiments.llama import llama_8b
from experiments.marin_models import marin_tokenizer
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    NEMOTRON_V2_SPLITS,
    SMOLTALK2_SPLITS,
    get_instruction_dataset,
)
from experiments.simple_sft_config import SimpleSFTConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize import lm_mixture_data_config
from marin.resources import TpuPodConfig

SLUGIFY_PATTERN = re.compile(r"[^a-z0-9]+")

TARGET_EPOCHS = 3
TRAIN_BATCH_SIZE = 2048

# Row counts captured on 2025-11-07 via `uv run python lib/marin/tools/get_hf_dataset_schema.py ...`.
SMOLTALK2_ROW_COUNTS = {
    "LongAlign_64k_Qwen3_32B_yarn_131k_think": 7526,
    "OpenThoughts3_1.2M_think": 1133524,
    "aya_dataset_Qwen3_32B_think": 15222,
    "multi_turn_reasoning_if_think": 28217,
    "s1k_1.1_think": 835,
    "smolagents_toolcalling_traces_think": 9079,
    "smoltalk_everyday_convs_reasoning_Qwen3_32B_think": 2057,
    "smoltalk_multilingual8_Qwen3_32B_think": 244736,
    "smoltalk_systemchats_Qwen3_32B_think": 27436,
    "table_gpt_Qwen3_32B_think": 13201,
    "LongAlign_64k_context_lang_annotated_lang_6_no_think": 6249,
    "Mixture_of_Thoughts_science_no_think": 86110,
    "OpenHermes_2.5_no_think": 384900,
    "OpenThoughts3_1.2M_no_think_no_think": 435193,
    "hermes_function_calling_v1_no_think": 8961,
    "smoltalk_multilingual_8languages_lang_5_no_think": 254047,
    "smoltalk_smollm3_everyday_conversations_no_think": 2260,
    "smoltalk_smollm3_explore_instruct_rewriting_no_think": 30391,
    "smoltalk_smollm3_smol_magpie_ultra_no_think": 406843,
    "smoltalk_smollm3_smol_rewrite_no_think": 53262,
    "smoltalk_smollm3_smol_summarize_no_think": 96061,
    "smoltalk_smollm3_systemchats_30k_no_think": 33997,
    "table_gpt_no_think": 13203,
    "tulu_3_sft_personas_instruction_following_no_think": 29970,
    "xlam_traces_no_think": 59962,
}

NEMOTRON_V2_ROW_COUNTS = {
    "stem": 355000,
    "chat": 627720,
    "math": 239467,
    "code": 175000,
    "multilingual_ja": 975202,
    "multilingual_de": 1015314,
    "multilingual_it": 1016503,
    "multilingual_es": 935704,
    "multilingual_fr": 1001504,
}

assert set(SMOLTALK2_SPLITS) == set(SMOLTALK2_ROW_COUNTS), "Update SMOLTALK2_ROW_COUNTS when SMOLTALK2_SPLITS changes"
assert set(NEMOTRON_V2_SPLITS) == set(
    NEMOTRON_V2_ROW_COUNTS
), "Update NEMOTRON_V2_ROW_COUNTS when NEMOTRON_V2_SPLITS changes"


def _slugify(value: str) -> str:
    slug = SLUGIFY_PATTERN.sub("_", value.lower()).strip("_")
    return slug or "dataset"


def build_dataset_specs() -> tuple[dict[str, str], dict[str, int]]:
    datasets: dict[str, str] = {}
    weights: dict[str, int] = {}

    for split in SMOLTALK2_SPLITS:
        key = _slugify(f"smoltalk2_{split}")
        datasets[key] = f"HuggingFaceTB/smoltalk2/{split}"
        weights[key] = SMOLTALK2_ROW_COUNTS[split]

    for split in NEMOTRON_V2_SPLITS:
        key = _slugify(f"nemotron_v2_{split}")
        datasets[key] = f"nvidia/Nemotron-Post-Training-Dataset-v2/{split}"
        weights[key] = NEMOTRON_V2_ROW_COUNTS[split]

    return datasets, weights


def create_tokenization_step(dataset_identifier: str, short_name: str) -> ExecutorStep:
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name}_marin_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=marin_tokenizer,
        format=ChatLmDatasetFormat(),
    )


DATASETS, mixture_weights = build_dataset_specs()
tokenized_datasets = {
    short_name: create_tokenization_step(dataset_identifier, short_name)
    for short_name, dataset_identifier in DATASETS.items()
}

assert set(tokenized_datasets.keys()) == set(mixture_weights.keys())

total_examples = sum(mixture_weights.values())
NUM_TRAIN_STEPS = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

mixture_sft_config = SimpleSFTConfig(
    train_batch_size=TRAIN_BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=1e-5,
    resources=TpuPodConfig(tpu_type="v4-2048"),
    tokenizer=marin_tokenizer,
    model_name_or_path="marin-community/marin-8b-base",
    max_seq_len=8192,
    seed=0,
)

mixture_config = lm_mixture_data_config(
    tokenized_datasets,
    mixture_weights,
    permutation_type="feistel",
    shuffle=True,
    missing_weights_are_validation=True,
    mixture_block_size=12288,  # large block size to include the tiny datasets (namely s1k_1.1)
)

marin_8b_sft_smoltalk2_nemotron_v2 = default_sft(
    name="marin_8b_sft_smoltalk2_nemotron_v2_big",
    tokenized=mixture_config,
    model_config=llama_8b,
    sft_config=mixture_sft_config,
    tags=["llama", "smoltalk2", "nemotron_v2", "sft"],
)

marin_8b_sft_smoltalk2_nemotron_v2_evals = default_sft_eval(
    marin_8b_sft_smoltalk2_nemotron_v2,
    use_levanter_inference=True,
    resource_config=TPU_V4_8,
)

if __name__ == "__main__":
    executor_main(steps=[marin_8b_sft_smoltalk2_nemotron_v2, *marin_8b_sft_smoltalk2_nemotron_v2_evals])
