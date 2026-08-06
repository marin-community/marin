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

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import experiment_main
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.proofpile import proofpile_dataset
from experiments.datasets.starcoder import starcoder_dataset
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.optimizer import TiedExpertLrScale
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.llama import llama3_tokenizer

_EXPERIMENT_PREFIX = "GRUG-XEM"
_BUDGET = 3.82e17
_HIDDEN_DIM = 512
_SEQUENCE_LENGTH = 4096
_TARGET_STEPS = 2**14
_SMOKE_STEPS = 500
_TRAIN_RESOURCES = ResourceConfig.with_tpu("v5p-8")

_NEMOTRON_WEIGHTS = {
    "hq_actual": 0.91351,
    "hq_synth": 2.72,
    "medium_high": 0.82471,
    "medium": 3.38,
    "medium_low": 1.54,
    "low_actual": 0.70123,
    "low_synth": 0.62771,
}
_STARCODER_WEIGHT = 0.25
_PROOFPILE_WEIGHT = 0.055

_TOPOLOGIES = {
    "baseline": (0, 1, 2, 3, 4, 5),
    "pairwise": (0, 1, 1, 2, 2, 3),
    "middle4": (0, 1, 1, 1, 1, 2),
}


def _run_id(phase: str, variant: str) -> str:
    default = f"grug_xem_d512_{phase}_{variant}"
    prefix = os.environ.get("GRUG_RUN_ID")
    run_id = default if prefix is None else f"{prefix}_{variant}"
    ferry_date = os.environ.get("FERRY_DATE")
    return run_id if ferry_date is None else f"{run_id}-{ferry_date}"


def _matrix(phase: str) -> Sequence[tuple[str, tuple[int, ...], TiedExpertLrScale]]:
    if phase == "smoke":
        variants: list[tuple[str, tuple[int, ...], TiedExpertLrScale]] = [
            ("baseline", _TOPOLOGIES["baseline"], TiedExpertLrScale.UNSCALED)
        ]
        for topology_name in ("pairwise", "middle4"):
            for scale in (
                TiedExpertLrScale.UNSCALED,
                TiedExpertLrScale.SQRT,
                TiedExpertLrScale.LINEAR,
            ):
                variants.append((f"{topology_name}_{scale.value}", _TOPOLOGIES[topology_name], scale))
        return variants
    if phase == "full":
        return [
            ("baseline", _TOPOLOGIES["baseline"], TiedExpertLrScale.UNSCALED),
            ("pairwise_sqrt", _TOPOLOGIES["pairwise"], TiedExpertLrScale.SQRT),
            ("middle4_sqrt", _TOPOLOGIES["middle4"], TiedExpertLrScale.SQRT),
        ]
    raise ValueError(f"GRUG_TIED_PHASE must be 'smoke' or 'full', got {phase!r}")


def tied_expert_runs(*, version: str | None = None) -> list[ArtifactStep[LevanterCheckpoint]]:
    phase = os.environ.get("GRUG_TIED_PHASE", "smoke").lower()
    base_model, base_optimizer, batch_size, full_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
        seq_len=_SEQUENCE_LENGTH,
    )
    if base_model.num_layers != 6:
        raise ValueError(f"d512 tied-expert matrix requires 6 layers, heuristic produced {base_model.num_layers}")
    steps = _SMOKE_STEPS if phase == "smoke" else full_steps

    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    train = {nemotron[split]: weight for split, weight in _NEMOTRON_WEIGHTS.items()}
    train[starcoder_dataset(tokenizer=llama3_tokenizer)] = _STARCODER_WEIGHT
    train[proofpile_dataset(tokenizer=llama3_tokenizer)] = _PROOFPILE_WEIGHT
    validation = [
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    ]

    runs: list[ArtifactStep[LevanterCheckpoint]] = []
    for variant, mapping, lr_scale in _matrix(phase):
        name = f"grug/tied_experts/d512/{phase}/{variant}"
        resolved_version = resolve_version(name, version)
        model = dataclasses.replace(base_model, expert_bank_for_layer=mapping)
        optimizer = dataclasses.replace(
            base_optimizer,
            expert_bank_group_sizes=model.expert_bank_group_sizes,
            tied_expert_lr_scale=lr_scale,
            schedule_horizon_steps=full_steps,
        )
        run_id = _run_id(phase, variant)

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
                resources=ctx.runtime_arg("train_resources"),
                steps=steps,
                batch_size=batch_size,
                seed=0,
                mp="params=float32,compute=bfloat16,output=bfloat16",
                tracker=WandbConfig(
                    project="marin_moe",
                    tags=[_EXPERIMENT_PREFIX, "tied-experts", "d512", phase],
                    group="grug-xem-architecture",
                    name=None,
                ),
                optimizer=optimizer,
                profiler=ProfilerConfig(enabled=phase == "smoke", start_step=5, num_steps=25),
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
                runtime_args={"train_resources": _TRAIN_RESOURCES},
            )
        )
    return runs


if __name__ == "__main__":
    experiment_main(tied_expert_runs)()
