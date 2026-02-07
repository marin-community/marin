#!/usr/bin/env python
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
Convert selected checkpoints from exp1699 (marin_yodas2-b5edae) into Hugging Face format.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

from experiments.audio.exp1699_marin_yodas2 import yodas_1b_model, yodas_qwen
from levanter.models.qwen import QwenConfig
from levanter.trainer import TrainerConfig
from marin.export import convert_checkpoint_to_hf_step
from marin.execution.executor import ExecutorStep, executor_main
from fray.cluster import ResourceConfig

CHECKPOINT_ROOT = "gs://marin-us-central1/checkpoints/exp1699_marin_yodas2-b5edae/checkpoints"
EXECUTOR_DESCRIPTION = "Convert exp1699 (Marin Yodas2) checkpoints to Hugging Face."
TOKENIZER = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"
CONVERSION_RESOURCES = ResourceConfig.with_tpu("v5p-8")


@dataclass(frozen=True)
class CheckpointExportSpec:
    """Metadata describing a checkpoint to convert."""

    step: int
    slug: str | None = None

    @property
    def checkpoint_path(self) -> str:
        return f"{CHECKPOINT_ROOT}/step-{self.step}"

    @property
    def name(self) -> str:
        if self.slug is not None:
            return self.slug
        return f"step-{self.step}"


CHECKPOINT_EXPORTS: tuple[CheckpointExportSpec, ...] = tuple(
    CheckpointExportSpec(step=step) for step in (10_000, 210_000, 220_000, 230_000)
)


def _trainer_config() -> TrainerConfig:
    """Extract the TrainerConfig from the original training experiment."""
    config = getattr(yodas_1b_model, "config", None)
    train_config = getattr(config, "train_config", None)
    trainer = getattr(train_config, "trainer", None)
    if not isinstance(trainer, TrainerConfig):
        raise ValueError(
            "exp1699_marin_yodas2.yodas_1b_model does not expose a TrainerConfig; "
            "update convert_initial_checkpoints.py to reference the correct training step."
        )
    return deepcopy(trainer)


def _model_config() -> QwenConfig:
    """Return the model configuration that was used for the training run."""
    return deepcopy(yodas_qwen)


def create_conversion_steps() -> Sequence[ExecutorStep]:
    steps = [
        convert_checkpoint_to_hf_step(
            name=f"marin-yodas2-{spec.name}",
            checkpoint_path=spec.checkpoint_path,
            model=_model_config(),
            trainer=_trainer_config(),
            resources=CONVERSION_RESOURCES,
            tokenizer=TOKENIZER,
        )
        for spec in CHECKPOINT_EXPORTS
    ]
    return steps


if __name__ == "__main__":
    executor_main(
        steps=create_conversion_steps(),
        description=EXECUTOR_DESCRIPTION,
    )
