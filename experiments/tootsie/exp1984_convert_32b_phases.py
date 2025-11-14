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
Export checkpoints for every Marin 32B "Tootsie" phase described in
`docs/reports/marin-32b-retro.md` to Hugging Face format.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

from levanter.trainer import TrainerConfig

from experiments.tootsie.exp1295_32b import llama_32b_remat, llama_32b_tootsie
from experiments.tootsie.exp1380_muon32b import llama_32b_muon
from experiments.tootsie.exp1390_32b_necro import marin_32b_necro
from experiments.tootsie.exp1395_qwen3_32b import marin_32b_qwen, qwen3_32b_remat
from marin.export import convert_checkpoint_to_hf_step
from marin.execution.executor import ExecutorStep, executor_main
from marin.resources import TpuPodConfig

BASE_CHECKPOINT_PREFIX = "gs://marin-us-central2/checkpoints"
CONVERSION_RESOURCES = TpuPodConfig(tpu_type="v4-8")


def _trainer_from_training_step(step: ExecutorStep) -> TrainerConfig:
    config = getattr(step, "config", None)
    train_config = getattr(config, "train_config", None)
    trainer = getattr(train_config, "trainer", None)
    if not isinstance(trainer, TrainerConfig):
        raise ValueError(
            f"Training step '{step.name}' does not expose a TrainLmOnPodConfig with a TrainerConfig; "
            "update exp1803 to reference the correct training experiment."
        )
    return deepcopy(trainer)


DEFAULT_TRAINER_CONFIGS: dict[str, TrainerConfig] = {
    # Phase 1 baseline (llama-32b-tootsie-2)
    "llama-32b-tootsie-2": _trainer_from_training_step(llama_32b_tootsie),
    # Phase 2a necromancy restart
    "marin-32b-necro-2": _trainer_from_training_step(marin_32b_necro),
    # Phase 2b optimizer swap (Muon)
    "marin-32b-muon-4": _trainer_from_training_step(llama_32b_muon),
    # Phase 3 Qwen switch
    "marin-32b-qwen": _trainer_from_training_step(marin_32b_qwen),
}


def _default_trainer_for_run(run_name: str) -> TrainerConfig:
    try:
        trainer = DEFAULT_TRAINER_CONFIGS[run_name]
    except KeyError as exc:
        raise ValueError(
            f"No TrainerConfig known for run '{run_name}'. "
            "Update DEFAULT_TRAINER_CONFIGS in exp1803_convert_32b_phases.py with the training step for this run."
        ) from exc
    return deepcopy(trainer)


@dataclass(frozen=True)
class PhaseExportSpec:
    """Metadata describing how to convert a single training phase."""

    slug: str
    label: str
    run_name: str
    model_name: str
    checkpoint_step: int | None
    trainer_run_name: str | None = None

    @property
    def model_config(self):
        if self.model_name == "llama32":
            return llama_32b_remat
        if self.model_name == "qwen32":
            return qwen3_32b_remat
        raise ValueError(f"Unknown model alias '{self.model_name}'")

    @property
    def checkpoint_path(self) -> str:
        base = f"{BASE_CHECKPOINT_PREFIX}/{self.run_name}/checkpoints"
        if self.checkpoint_step is None:
            return base
        return f"{base}/step-{self.checkpoint_step}"

    @property
    def discover_latest(self) -> bool:
        return self.checkpoint_step is None

    def resolve_trainer_config(self, override: TrainerConfig | None = None) -> TrainerConfig:
        if override is not None:
            return override
        trainer_run = self.trainer_run_name or self.run_name
        return _default_trainer_for_run(trainer_run)


PHASE_EXPORT_SPECS: tuple[PhaseExportSpec, ...] = (
    PhaseExportSpec(
        slug="phase1-baseline",
        label="Phase 1 - Baseline (Llama 32B)",
        run_name="llama-32b-tootsie-2",
        model_name="llama32",
        checkpoint_step=80000,
    ),
    PhaseExportSpec(
        slug="phase2a-necro",
        label="Phase 2a - Necromancy Restart",
        run_name="marin-32b-necro-2",
        model_name="llama32",
        checkpoint_step=None,  # discover the latest post-restart checkpoint
    ),
    PhaseExportSpec(
        slug="phase2b-muon",
        label="Phase 2b - Optimizer Swap (Muon)",
        run_name="marin-32b-muon-4",
        model_name="llama32",
        checkpoint_step=None,
    ),
    PhaseExportSpec(
        slug="phase3-qwen",
        label="Phase 3 - QK-Norm Switch",
        run_name="marin-32b-qwen",
        model_name="qwen32",
        checkpoint_step=160_000,
    ),
)


def create_phase_export_steps(trainer: TrainerConfig | None = None) -> Sequence[ExecutorStep]:
    """Instantiate conversion steps for each retrospective phase."""
    steps: list[ExecutorStep] = []
    for spec in PHASE_EXPORT_SPECS:
        trainer_config = spec.resolve_trainer_config(trainer)
        step = convert_checkpoint_to_hf_step(
            name=f"tootsie-32b-{spec.slug}",
            checkpoint_path=spec.checkpoint_path,
            model=spec.model_config,
            trainer=trainer_config,
            resources=CONVERSION_RESOURCES,
            tokenizer="marin-community/marin-tokenizer",
            discover_latest=spec.discover_latest,
        )
        steps.append(step)
    return steps


if __name__ == "__main__":
    executor_main(
        steps=create_phase_export_steps(),
        description="Convert every Marin 32B Tootsie phase into Hugging Face checkpoints.",
    )
