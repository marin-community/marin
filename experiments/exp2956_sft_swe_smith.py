# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
Fine-tunes a student model on SWE-smith agentic trajectories (Rust/ripgrep tasks).

Supports multiple teacher models:
  - gpt-5-mini:          AlienKevin/SWE-smith-rs-gpt-5-mini-trajectories                (3953 samples)
  - minimax-m2.5:        AlienKevin/SWE-smith-rs-minimax-m2.5-trajectories              (5251 samples)
  - gemini-3-flash:      AlienKevin/SWE-smith-rs-gemini-3-flash-trajectories            (1449 samples)
  - glm-4.6:             AlienKevin/SWE-smith-rs-glm-4.6-trajectories                    (369 samples)
  - swebm-minimax-m2.5:  AlienKevin/SWE-bench-multilingual-minimax-m2.5-trajectories     (299 samples)

Supports multiple student models:
  - qwen3-8b:                    Qwen/Qwen3-8B                       (v5p-8)
  - qwen25-coder-7b-instruct:    Qwen/Qwen2.5-Coder-7B-Instruct     (v5p-8)
  - qwen25-coder-32b-instruct:   Qwen/Qwen2.5-Coder-32B-Instruct    (v5p-64)

Each dataset has a JSON-serialized "messages" column in OpenAI chat format
plus metadata like instance_id, resolved, model, traj_id, and patch.

Usage:
    python -m experiments.exp2956_sft_swe_smith --teacher gpt-5-mini
    python -m experiments.exp2956_sft_swe_smith --teacher minimax-m2.5 --student qwen25-coder-32b-instruct
    python -m experiments.exp2956_sft_swe_smith --teacher gemini-3-flash
    python -m experiments.exp2956_sft_swe_smith --teacher glm-4.6
    python -m experiments.exp2956_sft_swe_smith --teacher swebm-minimax-m2.5

Full command for minimax-m2.5:
RAY_AUTH_MODE=token uv run lib/marin/src/marin/run/ray_run.py \\
    --env_vars WANDB_API_KEY ${WANDB_API_KEY} \\
    --env_vars WANDB_ENTITY marin-community \\
    --env_vars WANDB_PROJECT marin \\
    --env_vars TPU_CI true \\
    --env_vars HF_TOKEN ${HF_TOKEN} \\
    --env_vars RUN_ID "exp2956_minimax_$(date +%s)" \\
    --cluster us-east5-a \\
    -- python -m experiments.exp2956_sft_swe_smith --teacher minimax-m2.5 --force_run_failed true

Note: Pass a unique RUN_ID env var to avoid wandb init hangs when resubmitting
(wandb hangs when resuming a deleted run with the same ID).
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
from experiments.qwen2pt5_instruct_chat_template import QWEN_2_5_INSTRUCT_CHAT_TEMPLATE
from experiments.qwen3 import (
    qwen2_5_coder_7b_instruct,
    qwen2_5_coder_7b_instruct_tokenizer,
    qwen2_5_coder_32b_instruct,
    qwen2_5_coder_32b_instruct_tokenizer,
    qwen3_8b,
    qwen3_8b_tokenizer,
)
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


@dataclass(frozen=True)
class StudentConfig:
    """Configuration for a student model to fine-tune."""

    model_config: object
    tokenizer: str
    model_name_or_path: str
    chat_template: str
    resources: ResourceConfig
    train_batch_size: int
    microbatch_size: int


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
    "glm-4.6": TeacherConfig(
        hf_dataset_id="AlienKevin/SWE-smith-rs-glm-4.6-trajectories",
        num_samples=369,
    ),
    "swebm-minimax-m2.5": TeacherConfig(
        hf_dataset_id="AlienKevin/SWE-bench-multilingual-minimax-m2.5-trajectories",
        num_samples=299,
    ),
}

STUDENT_CONFIGS: dict[str, StudentConfig] = {
    "qwen3-8b": StudentConfig(
        model_config=qwen3_8b,
        tokenizer=qwen3_8b_tokenizer,
        model_name_or_path="Qwen/Qwen3-8B",
        chat_template=QWEN_3_CHAT_TEMPLATE,
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=16,
        microbatch_size=4,
    ),
    "qwen25-coder-7b-instruct": StudentConfig(
        model_config=qwen2_5_coder_7b_instruct,
        tokenizer=qwen2_5_coder_7b_instruct_tokenizer,
        model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        chat_template=QWEN_2_5_INSTRUCT_CHAT_TEMPLATE,
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=16,
        microbatch_size=4,
    ),
    "qwen25-coder-32b-instruct": StudentConfig(
        model_config=qwen2_5_coder_32b_instruct,
        tokenizer=qwen2_5_coder_32b_instruct_tokenizer,
        model_name_or_path="Qwen/Qwen2.5-Coder-32B-Instruct",
        chat_template=QWEN_2_5_INSTRUCT_CHAT_TEMPLATE,
        resources=ResourceConfig.with_tpu("v5p-64"),
        train_batch_size=32,
        microbatch_size=32,
    ),
}

TARGET_EPOCHS = 3


def _create_tokenization_step(
    dataset_identifier: str, short_name: str, tokenizer: str, chat_template: str
) -> TokenizerStep:
    dataset_config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_identifier]
    dataset = get_instruction_dataset(dataset_identifier, splits=dataset_config.splits)
    return default_tokenize(
        name=f"{short_name.split('/')[-1]}_qwen3_8b_tokenizer",
        dataset=dataset / "**/*.jsonl.gz",
        tokenizer=tokenizer,
        format=ChatLmDatasetFormat(chat_template=chat_template),
    )


def build_swe_smith_sft(teacher: str, student: str = "qwen3-8b") -> tuple[ExecutorStep, ExecutorStep]:
    """Build SFT experiment and checkpoint steps for a given teacher and student model.

    Args:
        teacher: One of the keys in TEACHER_CONFIGS.
        student: One of the keys in STUDENT_CONFIGS (default: "qwen3-8b").

    Returns:
        (sft_step, checkpoint_step) tuple.
    """
    teacher_config = TEACHER_CONFIGS[teacher]
    student_config = STUDENT_CONFIGS[student]

    resources = student_config.resources
    train_batch_size = student_config.train_batch_size
    microbatch_size = student_config.microbatch_size
    resource_suffix = resources.device.variant.replace("-", "") if resources.device.kind == "tpu" else "gpu"

    model_config = dataclasses.replace(
        student_config.model_config,
        max_seq_len=32768,
        rope=DefaultRotaryEmbeddingsConfig(theta=1_000_000.0),
        cross_entropy_block_size=32000,  # Process vocab in chunks to reduce memory during loss computation
    )

    datasets = {teacher_config.hf_dataset_id: teacher_config.hf_dataset_id}
    weights = {teacher_config.hf_dataset_id: float(teacher_config.num_samples)}

    tokenized_datasets = {
        name: _create_tokenization_step(dataset_id, name, student_config.tokenizer, student_config.chat_template)
        for name, dataset_id in datasets.items()
    }

    total_examples = teacher_config.num_samples
    num_train_steps = math.ceil(TARGET_EPOCHS * total_examples / train_batch_size)

    warmup_steps = 5
    warmup_fraction = warmup_steps / num_train_steps if num_train_steps > 0 else 0.0
    decay_fraction = 1.0 - warmup_fraction  # All post-warmup steps are cosine decay (no stable phase)

    sft_config = SimpleSFTConfig(
        resources=resources,
        tokenizer=student_config.tokenizer,
        model_name_or_path=student_config.model_name_or_path,
        train_batch_size=train_batch_size,
        per_device_parallelism=compute_per_device_parallelism(train_batch_size, microbatch_size, resources),
        per_device_eval_parallelism=8,
        num_train_steps=num_train_steps,
        learning_rate=1e-4,
        max_seq_len=32768,
        seed=42,
        steps_per_checkpoint=(total_examples // train_batch_size) // 4,  # Every quarter epoch
        lr_schedule="cosine",
        warmup=warmup_fraction,
        decay=decay_fraction,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
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

    # Sanitize names for use in experiment name (e.g., "minimax-m2.5" -> "minimax_m2_5")
    teacher_slug = teacher.replace("-", "_").replace(".", "_")
    student_slug = student.replace("-", "_").replace(".", "_")

    sft_step = default_sft(
        name=f"exp2956_sft_swe_smith_{teacher_slug}_{student_slug}_32768tokens_{resource_suffix}",
        tokenized=mixture_config,
        model_config=model_config,
        sft_config=sft_config,
        tags=["qwen", "swe-smith", "sft", teacher, student, resource_suffix],
    )

    checkpoint = sft_step.cd(f"hf/step-{num_train_steps - 1}").nonblocking()

    return sft_step, checkpoint


# Build all teacher experiments at module level for importability (default student: qwen3-8b)
exp2956_sft_swe_smith_qwen3_8b, exp2956_checkpoint = build_swe_smith_sft("gpt-5-mini")
exp2956_sft_swe_smith_minimax_qwen3_8b, exp2956_minimax_checkpoint = build_swe_smith_sft("minimax-m2.5")
exp2956_sft_swe_smith_gemini_qwen3_8b, exp2956_gemini_checkpoint = build_swe_smith_sft("gemini-3-flash")
exp2956_sft_swe_smith_glm_qwen3_8b, exp2956_glm_checkpoint = build_swe_smith_sft("glm-4.6")
exp2956_sft_swe_smith_swebm_minimax_qwen3_8b, exp2956_swebm_minimax_checkpoint = build_swe_smith_sft(
    "swebm-minimax-m2.5"
)


if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="SFT on SWE-smith trajectories")
    parser.add_argument(
        "--teacher",
        choices=list(TEACHER_CONFIGS.keys()),
        default="gpt-5-mini",
        help="Teacher model whose trajectories to train on (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--student",
        choices=list(STUDENT_CONFIGS.keys()),
        default="qwen3-8b",
        help="Student model to fine-tune (default: qwen3-8b)",
    )
    args, remaining = parser.parse_known_args()
    # Strip --teacher and --student from sys.argv so draccus in executor_main only sees executor args
    sys.argv = [sys.argv[0], *remaining]
    sft_step, _ = build_swe_smith_sft(args.teacher, args.student)
    executor_main(steps=[sft_step])
