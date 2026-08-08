# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched June 67B-A2B tied-expert architecture comparison in ``us-central2``.

This is a from-initialization architecture test. It does not read or convert the
Snowball checkpoint. ``smoke`` runs 100 steps to gate compile, HBM, routing, and
early optimization; ``milestone`` runs 3,000 steps and includes matched Paloma
evaluations. Both phases use the full 10T schedule for optimizer timing.
"""

import dataclasses
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.launch_2x_bs import GrugMoeLaunchConfig2xBS, run_grug_moe_trial_2x_bs
from experiments.june_tpu_67b_a2b.moe.launch_datakit_moe_mix import (
    _MIXTURE_BLOCK_SIZE,
    _phase_weights,
    _validation_component,
    datakit_components_with_prefix,
)
from experiments.june_tpu_67b_a2b.moe.optimizer import TiedExpertLrScale
from experiments.june_tpu_67b_a2b.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

_EXPERIMENT_REGION = "us-central2"
_CENTRAL2_PREFIX = "gs://marin-us-central2"
_RESOURCES_KEY = "train_resources"
_RESOURCES = ResourceConfig.with_tpu("v4-2048", regions=[_EXPERIMENT_REGION], preemptible=False)

_DIM = 2_560
_SEQUENCE_LENGTH = 8_192
_BATCH_SIZE = 8_192
_FULL_SCHEDULE_STEPS = 150_000
_SMOKE_STEPS = 100
_MILESTONE_STEPS = 3_000
_EXPERT_PARALLEL = 1
_REPLICA_AXIS_SIZE = 8
_PER_DEVICE_PARALLELISM = 8

_UNTIED_TOPOLOGY = tuple(range(26))
_MIDDLE_GROUPS_TOPOLOGY = (
    0,
    1,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    7,
    7,
    8,
    9,
)


class JuneTiedPhase(StrEnum):
    SMOKE = "smoke"
    MILESTONE = "milestone"


@dataclass(frozen=True)
class JuneTiedVariant:
    name: str
    topology: tuple[int, ...]


_VARIANTS = (
    JuneTiedVariant("baseline", _UNTIED_TOPOLOGY),
    JuneTiedVariant("middle_groups_unscaled", _MIDDLE_GROUPS_TOPOLOGY),
)

_VALIDATION = {
    **{f"paloma/{name}": step for name, step in paloma_datasets(tokenizer=marin_tokenizer).items()},
    **{f"uncheatable_eval/{name}": step for name, step in uncheatable_datasets(tokenizer=marin_tokenizer).items()},
}


def _training_data(ctx: StepContext) -> LmDataConfig:
    if ctx.is_fingerprint:
        validation_components = {
            name: _validation_component(ctx.artifact_path(dep)) for name, dep in _VALIDATION.items()
        }
    else:
        validation_components = {name: ctx.resolved(dep).as_component() for name, dep in _VALIDATION.items()}

    validation_weights = {name: 0.0 for name in validation_components}
    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components={**datakit_components_with_prefix(_CENTRAL2_PREFIX), **validation_components},
        train_weights=[(0, {**_phase_weights(0), **validation_weights})],
        auto_build_caches=False,
        mixture_block_size=_MIXTURE_BLOCK_SIZE,
    )


def build_tied_expert_runs(
    *,
    version: str | None = None,
    phase: JuneTiedPhase | None = None,
    variant_names: Sequence[str] | None = None,
) -> list[ArtifactStep[LevanterCheckpoint]]:
    """Build the matched untied/tied June 67B architecture runs."""
    if phase is None:
        phase = JuneTiedPhase(os.environ.get("GRUG_JUNE67B_TIED_PHASE", JuneTiedPhase.SMOKE).lower())
    if variant_names is None and (requested := os.environ.get("GRUG_JUNE67B_TIED_VARIANTS")):
        variant_names = tuple(name.strip() for name in requested.split(",") if name.strip())

    variants = list(_VARIANTS)
    if variant_names is not None:
        if len(set(variant_names)) != len(variant_names):
            raise ValueError(f"GRUG_JUNE67B_TIED_VARIANTS contains duplicates: {variant_names}")
        available = {variant.name for variant in variants}
        unknown = set(variant_names) - available
        if unknown:
            raise ValueError(f"unknown June 67B tied-expert variants: {sorted(unknown)}")
        requested_names = set(variant_names)
        variants = [variant for variant in variants if variant.name in requested_names]
        if not variants:
            raise ValueError("GRUG_JUNE67B_TIED_VARIANTS selected no runs")

    steps = _SMOKE_STEPS if phase is JuneTiedPhase.SMOKE else _MILESTONE_STEPS
    heuristic = MoeMuonHHeuristic(min_lr_ratio=0.05)
    base_model = dataclasses.replace(
        heuristic.build_model_config(_DIM, seq_len=_SEQUENCE_LENGTH),
        disable_pko=True,
        disable_long_rope=True,
        sliding_window=2_048,
        use_array_stacked_blocks=True,
    )
    if base_model.num_layers != len(_UNTIED_TOPOLOGY):
        raise ValueError(f"June 67B topology requires 26 layers, got {base_model.num_layers}")
    schedule_tokens = float(_FULL_SCHEDULE_STEPS * _BATCH_SIZE * _SEQUENCE_LENGTH)
    base_optimizer = dataclasses.replace(
        heuristic.build_muonh_config(_BATCH_SIZE, schedule_tokens, _DIM, seq_len=_SEQUENCE_LENGTH),
        rmsnorm_to_adam=True,
        schedule_num_train_steps_override=_FULL_SCHEDULE_STEPS,
        tied_expert_lr_scale=TiedExpertLrScale.UNSCALED,
    )

    runs: list[ArtifactStep[LevanterCheckpoint]] = []
    for variant in variants:
        name = f"grug/tied_experts/june67b/{phase.value}/{variant.name}"
        resolved_version = resolve_version(name, version)
        model = dataclasses.replace(base_model, expert_bank_for_layer=variant.topology)
        optimizer = dataclasses.replace(base_optimizer, expert_bank_for_layer=variant.topology)
        run_id = f"grug_xem_june67b_{phase.value}_{variant.name}"

        def build_config(
            ctx: StepContext,
            *,
            model=model,
            optimizer=optimizer,
            run_id=run_id,
            variant_name=variant.name,
        ) -> GrugMoeLaunchConfig2xBS:
            return GrugMoeLaunchConfig2xBS(
                model=model,
                data=_training_data(ctx),
                output_path=ctx.output_path,
                run_id=run_id,
                resources=ctx.runtime_arg(_RESOURCES_KEY),
                steps=steps,
                batch_size=_BATCH_SIZE,
                seed=0,
                mp="params=float32,compute=bfloat16,output=bfloat16",
                tracker=WandbConfig(
                    project="marin_moe",
                    tags=["GRUG-XEM", "tied-experts", "june67b", phase.value, variant_name],
                    group="grug-xem-june67b-architecture",
                    name=None,
                ),
                optimizer=optimizer,
                profiler=ProfilerConfig(
                    enabled=phase is JuneTiedPhase.SMOKE,
                    start_step=5,
                    num_steps=10,
                ),
                grug_trainer=GrugTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                    replica_axis_size=_REPLICA_AXIS_SIZE,
                ),
                eval=GrugEvalConfig(
                    eval_batch_size=1_024,
                    steps_per_eval=_SMOKE_STEPS if phase is JuneTiedPhase.SMOKE else 500,
                    max_eval_batches=1,
                    eval_current=True,
                    eval_ema=False,
                ),
                expert_parallel=_EXPERT_PARALLEL,
                checkpoint_keep=[{"every": _SMOKE_STEPS if phase is JuneTiedPhase.SMOKE else 500}],
                save_interval_minutes=60,
                source_batch_size=None,
                resume_step=0,
                per_device_parallelism=_PER_DEVICE_PARALLELISM,
            )

        runs.append(
            ArtifactStep(
                name=user_namespaced_name(name, resolved_version),
                version=resolved_version,
                artifact_type=LevanterCheckpoint,
                run=run_grug_moe_trial_2x_bs,
                build_config=build_config,
                deps=tuple(_VALIDATION.values()),
                runtime_args={_RESOURCES_KEY: _RESOURCES},
            )
        )
    return runs


if __name__ == "__main__":
    experiment_main(build_tied_expert_runs)()
