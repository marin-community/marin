# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched architecture gates for cross-layer routed-expert tying.

Set ``GRUG_TIED_MODEL=d512`` (default) for the original LR/topology matrix or
``GRUG_TIED_MODEL=d768`` for the contemporaneous untied and two-anchor middle-four
comparison with unscaled and ``1/sqrt(g)`` expert learning rates. Larger d1024
and d1280 comparisons use two anchors at each end, core groups no larger than
four, and only the empirically selected unscaled expert learning rate.
``GRUG_TIED_PHASE=smoke`` (default) runs 500 steps; ``full`` uses the model's
compute-optimal schedule.
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
_SEQUENCE_LENGTH = 4096
_TARGET_STEPS = 2**14
_SMOKE_STEPS = 500
_EXPERIMENT_REGION = "us-central1"
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v5p-8", regions=[_EXPERIMENT_REGION])
_TRAIN_RESOURCES_KEY = "train_resources"

_D512_BASELINE_TOPOLOGY = (0, 1, 2, 3, 4, 5)
_D512_PAIRWISE_TOPOLOGY = (0, 1, 1, 2, 2, 3)
_D512_MIDDLE_FOUR_TOPOLOGY = (0, 1, 1, 1, 1, 2)
_D768_BASELINE_TOPOLOGY = (0, 1, 2, 3, 4, 5, 6, 7)
_D768_TWO_ANCHOR_MIDDLE_FOUR_TOPOLOGY = (0, 1, 2, 2, 2, 2, 3, 4)
_D1024_BASELINE_TOPOLOGY = tuple(range(11))
_D1024_TWO_ANCHOR_CORE_GROUPS_TOPOLOGY = (0, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5)
_D1280_BASELINE_TOPOLOGY = tuple(range(13))
_D1280_TWO_ANCHOR_CORE_GROUPS_TOPOLOGY = (0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6)


class TiedExpertPhase(StrEnum):
    SMOKE = "smoke"
    FULL = "full"


class TiedExpertModelSize(StrEnum):
    D512 = "d512"
    D768 = "d768"
    D1024 = "d1024"
    D1280 = "d1280"


@dataclass(frozen=True)
class TiedExpertVariant:
    name: str
    topology: tuple[int, ...]
    lr_scale: TiedExpertLrScale


@dataclass(frozen=True)
class TiedExpertModelSpec:
    budget: float
    hidden_dim: int
    num_layers: int


_MODEL_SPECS = {
    TiedExpertModelSize.D512: TiedExpertModelSpec(budget=3.82e17, hidden_dim=512, num_layers=6),
    TiedExpertModelSize.D768: TiedExpertModelSpec(budget=2.81e18, hidden_dim=768, num_layers=8),
    TiedExpertModelSize.D1024: TiedExpertModelSpec(budget=1.16e19, hidden_dim=1024, num_layers=11),
    TiedExpertModelSize.D1280: TiedExpertModelSpec(budget=3.46e19, hidden_dim=1280, num_layers=13),
}


def _run_id(
    model_size: TiedExpertModelSize,
    phase: TiedExpertPhase,
    variant: str,
    *,
    run_id_prefix: str | None,
    ferry_date: str | None,
) -> str:
    default = f"grug_xem_{model_size.value}_{phase.value}_{variant}"
    run_id = default if run_id_prefix is None else f"{run_id_prefix}_{variant}"
    return run_id if ferry_date is None else f"{run_id}-{ferry_date}"


def _matrix(model_size: TiedExpertModelSize, phase: TiedExpertPhase) -> Sequence[TiedExpertVariant]:
    if model_size is TiedExpertModelSize.D1024:
        return [
            TiedExpertVariant("baseline", _D1024_BASELINE_TOPOLOGY, TiedExpertLrScale.UNSCALED),
            TiedExpertVariant(
                "core_groups_two_anchor_unscaled",
                _D1024_TWO_ANCHOR_CORE_GROUPS_TOPOLOGY,
                TiedExpertLrScale.UNSCALED,
            ),
        ]
    if model_size is TiedExpertModelSize.D1280:
        return [
            TiedExpertVariant("baseline", _D1280_BASELINE_TOPOLOGY, TiedExpertLrScale.UNSCALED),
            TiedExpertVariant(
                "core_groups_two_anchor_unscaled",
                _D1280_TWO_ANCHOR_CORE_GROUPS_TOPOLOGY,
                TiedExpertLrScale.UNSCALED,
            ),
        ]
    if model_size is TiedExpertModelSize.D768:
        return [
            TiedExpertVariant("baseline", _D768_BASELINE_TOPOLOGY, TiedExpertLrScale.UNSCALED),
            TiedExpertVariant(
                "middle4_two_anchor_unscaled",
                _D768_TWO_ANCHOR_MIDDLE_FOUR_TOPOLOGY,
                TiedExpertLrScale.UNSCALED,
            ),
            TiedExpertVariant(
                "middle4_two_anchor_sqrt",
                _D768_TWO_ANCHOR_MIDDLE_FOUR_TOPOLOGY,
                TiedExpertLrScale.SQRT,
            ),
        ]
    if phase is TiedExpertPhase.SMOKE:
        variants = [TiedExpertVariant("baseline", _D512_BASELINE_TOPOLOGY, TiedExpertLrScale.UNSCALED)]
        for topology_name, topology in (
            ("pairwise", _D512_PAIRWISE_TOPOLOGY),
            ("middle4", _D512_MIDDLE_FOUR_TOPOLOGY),
        ):
            for scale in (
                TiedExpertLrScale.UNSCALED,
                TiedExpertLrScale.SQRT,
                TiedExpertLrScale.LINEAR,
            ):
                variants.append(TiedExpertVariant(f"{topology_name}_{scale.value}", topology, scale))
        return variants
    if phase is TiedExpertPhase.FULL:
        return [
            TiedExpertVariant("baseline", _D512_BASELINE_TOPOLOGY, TiedExpertLrScale.UNSCALED),
            TiedExpertVariant("pairwise_unscaled", _D512_PAIRWISE_TOPOLOGY, TiedExpertLrScale.UNSCALED),
            TiedExpertVariant("pairwise_sqrt", _D512_PAIRWISE_TOPOLOGY, TiedExpertLrScale.SQRT),
            TiedExpertVariant("middle4_unscaled", _D512_MIDDLE_FOUR_TOPOLOGY, TiedExpertLrScale.UNSCALED),
            TiedExpertVariant("middle4_sqrt", _D512_MIDDLE_FOUR_TOPOLOGY, TiedExpertLrScale.SQRT),
        ]
    raise ValueError(f"unknown tied-expert phase: {phase}")


def tied_expert_runs(
    *,
    version: str | None = None,
    model_size: TiedExpertModelSize | None = None,
    phase: TiedExpertPhase | None = None,
    run_id_prefix: str | None = None,
    ferry_date: str | None = None,
    variant_names: Sequence[str] | None = None,
) -> list[ArtifactStep[LevanterCheckpoint]]:
    if model_size is None:
        model_size = TiedExpertModelSize(os.environ.get("GRUG_TIED_MODEL", TiedExpertModelSize.D512).lower())
    if phase is None:
        phase = TiedExpertPhase(os.environ.get("GRUG_TIED_PHASE", TiedExpertPhase.SMOKE).lower())
    if run_id_prefix is None:
        run_id_prefix = os.environ.get("GRUG_RUN_ID")
    if ferry_date is None:
        ferry_date = os.environ.get("FERRY_DATE")
    if variant_names is None and (requested := os.environ.get("GRUG_TIED_VARIANTS")):
        variant_names = tuple(name.strip() for name in requested.split(",") if name.strip())
    model_spec = _MODEL_SPECS[model_size]
    base_model, base_optimizer, batch_size, full_steps = build_from_heuristic(
        budget=model_spec.budget,
        hidden_dim=model_spec.hidden_dim,
        target_steps=_TARGET_STEPS,
        seq_len=_SEQUENCE_LENGTH,
    )
    if base_model.num_layers != model_spec.num_layers:
        raise ValueError(
            f"{model_size.value} tied-expert matrix requires {model_spec.num_layers} layers, "
            f"heuristic produced {base_model.num_layers}"
        )
    steps = _SMOKE_STEPS if phase is TiedExpertPhase.SMOKE else full_steps

    train = grug_moe_training_datasets()
    validation = grug_moe_validation_datasets()

    variants = list(_matrix(model_size, phase))
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
        name = f"grug/tied_experts/{model_size.value}/{phase.value}/{variant.name}"
        resolved_version = resolve_version(name, version)
        model = dataclasses.replace(base_model, expert_bank_for_layer=variant.topology)
        optimizer = dataclasses.replace(
            base_optimizer,
            expert_bank_group_sizes=model.expert_bank_group_sizes,
            tied_expert_lr_scale=variant.lr_scale,
            schedule_horizon_steps=full_steps,
        )
        run_id = _run_id(
            model_size,
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
                    tags=[_EXPERIMENT_PREFIX, "tied-experts", model_size.value, phase.value],
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
