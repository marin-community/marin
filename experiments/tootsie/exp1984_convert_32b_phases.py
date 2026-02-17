#!/usr/bin/env python
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Export checkpoints for every Marin 32B "Tootsie" phase described in
`docs/reports/marin-32b-retro.md` to Hugging Face format.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import llama_32b_remat, llama_32b_train_config
from experiments.tootsie.exp1380_muon32b import llama_32b_warmstart_train as muon_train_config
from experiments.tootsie.exp1390_32b_necro import llama_32b_warmstart_train as necro_train_config
from experiments.tootsie.exp1395_qwen3_32b import qwen3_32b_remat, qwen_32b_warmstart_train
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner
from marin.export import convert_checkpoint_to_hf, ConvertCheckpointStepConfig

BASE_CHECKPOINT_PREFIX = "gs://marin-us-central2/checkpoints"
CONVERSION_RESOURCES = ResourceConfig.with_tpu("v4-8")


def _trainer_from_simple_config(config: SimpleTrainConfig) -> TrainerConfig:
    """Build a minimal TrainerConfig from a SimpleTrainConfig for checkpoint export."""
    return TrainerConfig(
        train_batch_size=config.train_batch_size,
        per_device_parallelism=config.per_device_parallelism,
        num_train_steps=config.num_train_steps,
        mesh=MeshConfig(
            compute_mapping={
                "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            }
        ),
    )


DEFAULT_TRAINER_CONFIGS: dict[str, TrainerConfig] = {
    # Phase 1 baseline (llama-32b-tootsie-2)
    "llama-32b-tootsie-2": _trainer_from_simple_config(llama_32b_train_config),
    # Phase 2a necromancy restart
    "marin-32b-necro-2": _trainer_from_simple_config(necro_train_config),
    # Phase 2b optimizer swap (Muon)
    "marin-32b-muon-4": _trainer_from_simple_config(muon_train_config),
    # Phase 3 Qwen switch
    "marin-32b-qwen": _trainer_from_simple_config(qwen_32b_warmstart_train),
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


def create_phase_export_steps(trainer: TrainerConfig | None = None) -> Sequence[StepSpec]:
    """Instantiate conversion steps for each retrospective phase."""
    steps: list[StepSpec] = []
    for spec in PHASE_EXPORT_SPECS:
        trainer_config = spec.resolve_trainer_config(trainer)
        _spec = spec
        _trainer_config = trainer_config
        step = StepSpec(
            name=f"tootsie-32b-{spec.slug}",
            hash_attrs={
                "checkpoint_path": spec.checkpoint_path,
                "model_name": spec.model_name,
                "discover_latest": spec.discover_latest,
            },
            fn=lambda output_path, _s=_spec, _t=_trainer_config: convert_checkpoint_to_hf(
                ConvertCheckpointStepConfig(
                    checkpoint_path=_s.checkpoint_path,
                    trainer=_t,
                    model=_s.model_config,
                    resources=CONVERSION_RESOURCES,
                    tokenizer="marin-community/marin-tokenizer",
                    discover_latest=_s.discover_latest,
                )
            ),
            resources=CONVERSION_RESOURCES,
        )
        steps.append(step)
    return steps


if __name__ == "__main__":
    StepRunner().run(create_phase_export_steps())
