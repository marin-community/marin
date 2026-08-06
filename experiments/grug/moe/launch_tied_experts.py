# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched d512 architecture gate for cross-layer routed-expert tying.

Set ``GRUG_TIED_PHASE=smoke`` (default) for the seven-run 500-step matrix or
``GRUG_TIED_PHASE=full`` for the contemporaneous full-schedule baseline,
pairwise, and middle-four runs.
"""

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    GrugMoeLaunchConfig,
    grug_moe_training_datasets,
    grug_moe_validation_datasets,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import TiedExpertLrScale
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

_EXPERIMENT_PREFIX = "GRUG-XEM"
_BUDGET = 3.82e17
_HIDDEN_DIM = 512
_MODEL_LABEL = f"d{_HIDDEN_DIM}"
_SEQUENCE_LENGTH = 4096
_TARGET_STEPS = 2**14
_SMOKE_STEPS = 500
_EXPERIMENT_REGION = "us-central1"
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v5p-8", regions=[_EXPERIMENT_REGION])
_TRAIN_RESOURCES_KEY = "train_resources"

_BASELINE_TOPOLOGY = (0, 1, 2, 3, 4, 5)
_PAIRWISE_TOPOLOGY = (0, 1, 1, 2, 2, 3)
_MIDDLE_FOUR_TOPOLOGY = (0, 1, 1, 1, 1, 2)


class TiedExpertPhase(StrEnum):
    SMOKE = "smoke"
    FULL = "full"


@dataclass(frozen=True)
class TiedExpertVariant:
    name: str
    topology: tuple[int, ...]
    lr_scale: TiedExpertLrScale


def _run_id(
    phase: TiedExpertPhase,
    variant: str,
    *,
    run_id_prefix: str | None,
    ferry_date: str | None,
) -> str:
    default = f"grug_xem_{_MODEL_LABEL}_{phase.value}_{variant}"
    run_id = default if run_id_prefix is None else f"{run_id_prefix}_{variant}"
    return run_id if ferry_date is None else f"{run_id}-{ferry_date}"


def _matrix(phase: TiedExpertPhase) -> Sequence[TiedExpertVariant]:
    if phase is TiedExpertPhase.SMOKE:
        variants = [TiedExpertVariant("baseline", _BASELINE_TOPOLOGY, TiedExpertLrScale.UNSCALED)]
        for topology_name, topology in (("pairwise", _PAIRWISE_TOPOLOGY), ("middle4", _MIDDLE_FOUR_TOPOLOGY)):
            for scale in (
                TiedExpertLrScale.UNSCALED,
                TiedExpertLrScale.SQRT,
                TiedExpertLrScale.LINEAR,
            ):
                variants.append(TiedExpertVariant(f"{topology_name}_{scale.value}", topology, scale))
        return variants
    if phase is TiedExpertPhase.FULL:
        return [
            TiedExpertVariant("baseline", _BASELINE_TOPOLOGY, TiedExpertLrScale.UNSCALED),
            TiedExpertVariant("pairwise_sqrt", _PAIRWISE_TOPOLOGY, TiedExpertLrScale.SQRT),
            TiedExpertVariant("middle4_sqrt", _MIDDLE_FOUR_TOPOLOGY, TiedExpertLrScale.SQRT),
        ]
    raise ValueError(f"unknown tied-expert phase: {phase}")


def tied_expert_runs(
    *,
    version: str | None = None,
    phase: TiedExpertPhase | None = None,
    run_id_prefix: str | None = None,
    ferry_date: str | None = None,
    variant_names: Sequence[str] | None = None,
) -> list[ArtifactStep[LevanterCheckpoint]]:
    if phase is None:
        phase = TiedExpertPhase(os.environ.get("GRUG_TIED_PHASE", TiedExpertPhase.SMOKE).lower())
    if run_id_prefix is None:
        run_id_prefix = os.environ.get("GRUG_RUN_ID")
    if ferry_date is None:
        ferry_date = os.environ.get("FERRY_DATE")
    if variant_names is None and (requested := os.environ.get("GRUG_TIED_VARIANTS")):
        variant_names = tuple(name.strip() for name in requested.split(",") if name.strip())
    base_model, base_optimizer, batch_size, full_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
        seq_len=_SEQUENCE_LENGTH,
    )
    if base_model.num_layers != 6:
        raise ValueError(
            f"{_MODEL_LABEL} tied-expert matrix requires 6 layers, heuristic produced {base_model.num_layers}"
        )
    steps = _SMOKE_STEPS if phase is TiedExpertPhase.SMOKE else full_steps

    train = grug_moe_training_datasets()
    validation = grug_moe_validation_datasets()

    variants = list(_matrix(phase))
    if variant_names is not None:
        if len(set(variant_names)) != len(variant_names):
            raise ValueError(f"GRUG_TIED_VARIANTS contains duplicates: {variant_names}")
        available = {variant.name for variant in variants}
        unknown = set(variant_names) - available
        if unknown:
            raise ValueError(f"unknown {phase.value} tied-expert variants: {sorted(unknown)}")
        requested_names = set(variant_names)
        variants = [variant for variant in variants if variant.name in requested_names]
        if not variants:
            raise ValueError("GRUG_TIED_VARIANTS selected no runs")

    runs: list[ArtifactStep[LevanterCheckpoint]] = []
    for variant in variants:
        name = f"grug/tied_experts/{_MODEL_LABEL}/{phase.value}/{variant.name}"
        resolved_version = resolve_version(name, version)
        model = dataclasses.replace(base_model, expert_bank_for_layer=variant.topology)
        optimizer = dataclasses.replace(
            base_optimizer,
            expert_bank_group_sizes=model.expert_bank_group_sizes,
            tied_expert_lr_scale=variant.lr_scale,
            schedule_horizon_steps=full_steps,
        )
        run_id = _run_id(
            phase,
            variant.name,
            run_id_prefix=run_id_prefix,
            ferry_date=ferry_date,
        )

        def build_config(
            ctx: StepContext,
            *,
            model=model,
            optimizer=optimizer,
            run_id=run_id,
        ) -> GrugMoeLaunchConfig:
            return GrugMoeLaunchConfig(
                model=model,
                data=mixture(ctx, train, validation=validation),
                output_path=ctx.output_path,
                run_id=run_id,
                resources=ctx.runtime_arg(_TRAIN_RESOURCES_KEY),
                steps=steps,
                batch_size=batch_size,
                seed=0,
                mp="params=float32,compute=bfloat16,output=bfloat16",
                tracker=WandbConfig(
                    project="marin_moe",
                    tags=[_EXPERIMENT_PREFIX, "tied-experts", _MODEL_LABEL, phase.value],
                    group="grug-xem-architecture",
                    name=None,
                ),
                optimizer=optimizer,
                profiler=ProfilerConfig(enabled=phase is TiedExpertPhase.SMOKE, start_step=5, num_steps=25),
                grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
                eval=GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                ),
            )

        runs.append(
            ArtifactStep(
                name=user_namespaced_name(name, resolved_version),
                version=resolved_version,
                artifact_type=LevanterCheckpoint,
                run=run_grug_moe_trial,
                build_config=build_config,
                deps=(*train, *validation),
                runtime_args={_TRAIN_RESOURCES_KEY: _TRAIN_RESOURCES},
            )
        )
    return runs


if __name__ == "__main__":
    experiment_main(tied_expert_runs)()
