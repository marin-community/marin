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

"""
Fine-tunes Qwen/Qwen3-8B on SWE-smith agentic trajectories (Rust/ripgrep tasks).

Supports multiple teacher models:
  - gpt-5-mini:     AlienKevin/SWE-smith-rs-gpt-5-mini-trajectories     (3953 samples)
  - minimax-m2.5:   AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories   (5251 samples)
  - gemini-3-flash: AlienKevin/SWE-smith-rs-gemini-3-flash-trajectories (1449 samples)

Each dataset has a JSON-serialized "messages" column in OpenAI chat format
plus metadata like instance_id, resolved, model, traj_id, and patch.

Usage:
    python -m experiments.exp2956_sft_swe_smith_qwen3_8b --teacher gpt-5-mini
    python -m experiments.exp2956_sft_swe_smith_qwen3_8b --teacher minimax-m2.5
    python -m experiments.exp2956_sft_swe_smith_qwen3_8b --teacher gemini-3-flash

Full command for minimax-m2.5:
RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py \
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \
    --env_vars WANDB_ENTITY marin-community \
    --env_vars WANDB_PROJECT marin \
    --env_vars TPU_CI true \
    --env_vars HF_TOKEN ${HF_TOKEN} \
    --cluster us-east5-a \
    -- python -m experiments.exp2956_sft_swe_smith_qwen3_8b --teacher minimax-m2.5 --force_run_failed true
"""

import argparse
import dataclasses
import math
from dataclasses import dataclass

from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig

from experiments.defaults import default_sft, default_tokenize
from experiments.posttrain.instruction_datasets import (
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    get_instruction_dataset,
)
from experiments.qwen3 import qwen3_8b, qwen3_8b_tokenizer
from experiments.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize import TokenizerStep, lm_mixture_data_config


@dataclass(frozen=True)
class TeacherConfig:
    """Configuration for a teacher model's SWE-smith trajectory dataset."""

    hf_dataset_id: str
    num_samples: int


TEACHER_CONFIGS: dict[str, TeacherConfig] = {
    "gpt-5-mini": TeacherConfig(
        hf_dataset_id="AlienKevin/SWE-smith-rs-gpt-5-mini-trajectories",
        num_samples=3953,
    ),
    "minimax-m2.5": TeacherConfig(
        hf_dataset_id="AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories",
        num_samples=5251,
    ),
    "gemini-3-flash": TeacherConfig(
        hf_dataset_id="AlienKevin/SWE-smith-rs-gemini-3-flash-trajectories",
        num_samples=1449,
    ),
}

TARGET_EPOCHS = 7
TRAIN_BATCH_SIZE = 16
MICROBATCH_SIZE = 16
RESOURCES = ResourceConfig.with_tpu("v5p-32")
RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "") if RESOURCES.device.kind == "tpu" else "gpu"

qwen3_8b_32768_seq_len = dataclasses.replace(
    qwen3_8b,
    max_seq_len=32768,
    rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
    cross_entropy_block_size=32000,  # Process vocab in chunks to reduce memory during loss computation
)


def _create_tokenization_step(dataset_identifier: str, short_name: str) -> TokenizerStep:
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name.split('/')[-1]}_qwen3_8b_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=qwen3_8b_tokenizer,
        format=ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE),
    )


def build_swe_smith_sft(teacher: str) -> tuple[ExecutorStep, ExecutorStep]:
    """Build SFT experiment and checkpoint steps for a given teacher model.

    Args:
        teacher: One of "gpt-5-mini", "minimax-m2.5", "gemini-3-flash".

    Returns:
        (sft_step, checkpoint_step) tuple.
    """
    config = TEACHER_CONFIGS[teacher]

    datasets = {config.hf_dataset_id: config.hf_dataset_id}
    weights = {config.hf_dataset_id: float(config.num_samples)}

    tokenized_datasets = {name: _create_tokenization_step(dataset_id, name) for name, dataset_id in datasets.items()}

    total_examples = config.num_samples
    num_train_steps = math.ceil(TARGET_EPOCHS * total_examples / TRAIN_BATCH_SIZE)

    sft_config = SimpleSFTConfig(
        resources=RESOURCES,
        tokenizer=qwen3_8b_tokenizer,
        model_name_or_path="Qwen/Qwen3-8B",
        train_batch_size=TRAIN_BATCH_SIZE,
        per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
        per_device_eval_parallelism=8,
        num_train_steps=num_train_steps,
        learning_rate=4e-5,
        max_seq_len=32768,
        seed=42,
        steps_per_checkpoint=(total_examples // TRAIN_BATCH_SIZE) // 4,  # Every quarter epoch
        lr_schedule="cosine",
        warmup=0.1,
        decay=0.9,
        weight_decay=0.0,
        max_grad_norm=1e-4,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-8,
        pad_tokenizer_to_match_model=True,  # Model and tokenizer vocab sizes differ
    )

    mixture_config = lm_mixture_data_config(
        tokenized_datasets,
        weights,
        permutation_type="feistel",
        # IMPORTANT: Era shuffling (shuffle after every epoch). `shuffle=True` repeats the same shuffle each epoch.
        shuffle=total_examples,
        missing_weights_are_validation=True,
        mixture_block_size=12288,  # Doesn't matter for mixtures with 1 dataset
    )

    # Sanitize teacher name for use in experiment name (e.g., "minimax-m2.5" -> "minimax_m2_5")
    teacher_slug = teacher.replace("-", "_").replace(".", "_")

    sft_step = default_sft(
        name=f"exp2956_sft_swe_smith_{teacher_slug}_qwen3_8b_32768tokens_{RESOURCE_SUFFIX}",
        tokenized=mixture_config,
        model_config=qwen3_8b_32768_seq_len,
        sft_config=sft_config,
        tags=["qwen", "swe-smith", "sft", teacher, RESOURCE_SUFFIX],
    )

    checkpoint = sft_step.cd(f"hf/step-{num_train_steps - 1}").nonblocking()

    return sft_step, checkpoint


# Build all teacher experiments at module level for importability
exp2956_sft_swe_smith_qwen3_8b, exp2956_checkpoint = build_swe_smith_sft("gpt-5-mini")
exp2956_sft_swe_smith_minimax_qwen3_8b, exp2956_minimax_checkpoint = build_swe_smith_sft("minimax-m2.5")
exp2956_sft_swe_smith_gemini_qwen3_8b, exp2956_gemini_checkpoint = build_swe_smith_sft("gemini-3-flash")


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="SFT on SWE-smith trajectories")
    parser.add_argument(
        "--teacher",
        choices=list(TEACHER_CONFIGS.keys()),
        default="gpt-5-mini",
        help="Teacher model whose trajectories to train on (default: gpt-5-mini)",
    )
    args, remaining = parser.parse_known_args()
    # Strip --teacher from sys.argv so draccus in executor_main only sees executor args
    sys.argv = [sys.argv[0], *remaining]
    sft_step, _ = build_swe_smith_sft(args.teacher)
    executor_main(steps=[sft_step])
