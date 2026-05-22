# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch Grug-MoE proportional-to-v4 interpolation path runs.

This is the one-dimensional path test

    w_t = (1 - t) * w_proportional + t * w_v4

for t in {0.25, 0.50, 0.75}. The endpoint weights are recovered from the
completed Grug-MoE scaling tracks on GCS, while the training/data wiring uses
the canonical Dolma 3 + Dolmino top-level runtime-cache builder in this
checkout.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from datetime import timedelta
from functools import cache
from pathlib import Path
from typing import Any

import fsspec
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture
from marin.processing.tokenize.data_configs import TokenizedMixtureGroup, lm_varying_mixture_data_config
from marin.training.training import temporary_checkpoint_base_path

from experiments.defaults import default_validation_sets
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DOMAIN_NAMES,
    PHASE_BOUNDARIES,
    PHASE_NAMES,
    TARGET_BUDGET,
    build_top_level_domains,
)
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.marin_models import marin_tokenizer

GCS_GRUG_PREFIX = "gs://marin-us-east5/grug"
ENDPOINT_EXECUTOR_ROOTS = {
    "grug_moe_mix_d512-2.19e+17": f"{GCS_GRUG_PREFIX}/grug_moe_mix_d512-2.19e+17-e6a48f",
    "grug_moe_mix_v4_d512-2.19e+17": f"{GCS_GRUG_PREFIX}/grug_moe_mix_v4_d512-2.19e+17-6aa7c8",
}
OUTPUT_DIR = Path(
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/" "grug_moe_v4_path_test_20260520"
)
TARGET_STEPS = 2**14
RUN_ID_PREFIX = "grug_moe_mix_v4_path_r1"
PATH_T_VALUES: tuple[tuple[str, float], ...] = (
    ("t025", 0.25),
    ("t050", 0.50),
    ("t075", 0.75),
)
SCALES: tuple[tuple[float, int], ...] = (
    (2.19e17, 512),
    (1.70e18, 768),
    (9.00e18, 1024),
    (2.83e19, 1280),
    (9.00e19, 1536),
)


def _read_text(path: str, *, allow_failure: bool = False) -> str:
    try:
        with fsspec.open(path, "rt") as handle:
            return handle.read()
    except OSError:
        if allow_failure:
            return ""
        raise


def _run_id(track: str, budget: float, hidden_dim: int) -> str:
    return f"{track}_d{hidden_dim}-{budget:.2e}"


def _successful_executor_info(logical_run_id: str) -> dict[str, Any]:
    root = ENDPOINT_EXECUTOR_ROOTS[logical_run_id]
    status = _read_text(f"{root}/.executor_status", allow_failure=True).strip()
    if status != "SUCCESS":
        raise FileNotFoundError(f"Endpoint executor root is not successful: {root} status={status!r}")
    return json.loads(_read_text(f"{root}/.executor_info"))


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    missing = set(DOMAIN_NAMES) - set(weights)
    extra = set(weights) - set(DOMAIN_NAMES)
    if missing:
        raise ValueError(f"Endpoint weights are missing domains: {sorted(missing)}")
    if extra:
        weights = {name: value for name, value in weights.items() if name in DOMAIN_NAMES}
    total = sum(float(weights[name]) for name in DOMAIN_NAMES)
    if total <= 0:
        raise ValueError("Endpoint weight sum must be positive")
    return {name: float(weights[name]) / total for name in DOMAIN_NAMES}


@cache
def _endpoint_weights() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return normalized proportional weights and normalized v4 phase weights."""
    budget, hidden_dim = SCALES[0]
    prop_info = _successful_executor_info(_run_id("grug_moe_mix", budget, hidden_dim))
    v4_info = _successful_executor_info(_run_id("grug_moe_mix_v4", budget, hidden_dim))

    prop_weights = prop_info["config"]["data"]["train_weights"]
    if not isinstance(prop_weights, dict):
        raise ValueError("Expected proportional endpoint to use constant train_weights")

    v4_schedule = v4_info["config"]["data"]["train_weights"]
    if not isinstance(v4_schedule, list) or len(v4_schedule) != len(PHASE_NAMES):
        raise ValueError("Expected v4 endpoint to use a two-phase train_weights schedule")

    v4_by_phase: dict[str, dict[str, float]] = {}
    for phase_name, item in zip(PHASE_NAMES, v4_schedule, strict=True):
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise ValueError(f"Unexpected v4 schedule item: {item!r}")
        _, phase_weights = item
        v4_by_phase[phase_name] = _normalize(phase_weights)

    return _normalize(prop_weights), v4_by_phase


def _interpolate_weights(
    proportional: dict[str, float],
    v4_by_phase: dict[str, dict[str, float]],
    t_value: float,
) -> dict[str, dict[str, float]]:
    interpolated: dict[str, dict[str, float]] = {}
    for phase_name in PHASE_NAMES:
        weights = {
            domain: (1.0 - t_value) * proportional[domain] + t_value * v4_by_phase[phase_name][domain]
            for domain in DOMAIN_NAMES
        }
        interpolated[phase_name] = _normalize(weights)
    return interpolated


@cache
def _runtime_components() -> dict[str, object]:
    components: dict[str, object] = {}
    for domain in build_top_level_domains(runtime_cache_region="us-east5"):
        if len(domain.components) == 1:
            components[domain.name] = domain.components[0].get_step()
            continue
        components[domain.name] = TokenizedMixtureGroup(
            components={component.name: component.get_step() for component in domain.components},
            weights=domain.get_component_weights(),
            token_counts={component.name: int(component.weight) for component in domain.components},
        )
    return components


def _phase_boundary_step(*, total_steps: int, batch_size: int) -> int:
    boundary_fraction = PHASE_BOUNDARIES[0]
    raw_step = int(total_steps * boundary_fraction)
    step_alignment = 2048 // math.gcd(batch_size, 2048)
    return (raw_step // step_alignment) * step_alignment


def _data_config(
    *,
    phase_weights: dict[str, dict[str, float]],
    steps: int,
    batch_size: int,
    max_seq_len: int,
):
    if tuple(phase_weights) != PHASE_NAMES:
        raise ValueError(f"Expected phase weights in order {PHASE_NAMES}, got {tuple(phase_weights)}")
    boundary = _phase_boundary_step(total_steps=steps, batch_size=batch_size)
    train_weights = [
        (0, phase_weights["phase_0"]),
        (boundary, phase_weights["phase_1"]),
    ]
    experiment_budget = batch_size * steps * max_seq_len
    if experiment_budget > TARGET_BUDGET:
        raise ValueError(f"experiment_budget={experiment_budget} exceeds target_budget={TARGET_BUDGET}")

    base = lm_varying_mixture_data_config(
        components=_runtime_components(),
        weights_list=train_weights,
        shuffle=True,
        mixture_block_size=2048,
    )
    base = dataclasses.replace(base, target_budget=TARGET_BUDGET, experiment_budget=experiment_budget)
    return add_validation_sets_to_mixture(base, default_validation_sets(tokenizer=marin_tokenizer))


def _resolve_tracker(tracker, run_id: str):
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_path(config: GrugMoeLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )
    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


def _build_scale_step(t_slug: str, t_value: float, budget: float, hidden_dim: int) -> ExecutorStep:
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=budget,
        hidden_dim=hidden_dim,
        target_steps=TARGET_STEPS,
    )
    proportional, v4_by_phase = _endpoint_weights()
    phase_weights = _interpolate_weights(proportional, v4_by_phase, t_value)
    slug = f"{t_slug}_d{hidden_dim}-{budget:.2e}"
    run_id = f"{RUN_ID_PREFIX}_{slug}"

    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_grug_moe_path,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=_data_config(
                phase_weights=phase_weights,
                steps=steps,
                batch_size=batch_size,
                max_seq_len=model.max_seq_len,
            ),
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8", zone="us-east5-a")),
            steps=versioned(steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin_moe",
                tags=["moe", "dolma3_dolmino_mix", "v4_path_test", t_slug, f"d{hidden_dim}", f"{budget:.2e}"],
                group="moe-v4-path-test",
                name=None,
            ),
            optimizer=versioned(optimizer),
            profiler=versioned(ProfilerConfig()),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=512,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


def _write_local_artifacts() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    proportional, v4_by_phase = _endpoint_weights()

    with (OUTPUT_DIR / "candidate_weights_long.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "t", "phase", "domain", "weight"])
        writer.writeheader()
        for t_slug, t_value in PATH_T_VALUES:
            weights = _interpolate_weights(proportional, v4_by_phase, t_value)
            for phase_name, phase_weights in weights.items():
                for domain, weight in phase_weights.items():
                    writer.writerow(
                        {
                            "candidate_id": f"v4_path_{t_slug}",
                            "t": t_value,
                            "phase": phase_name,
                            "domain": domain,
                            "weight": weight,
                        }
                    )

    with (OUTPUT_DIR / "training_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "candidate_id",
                "t",
                "hidden_dim",
                "budget",
                "target_steps",
                "batch_size",
                "phase_1_boundary_step",
            ],
        )
        writer.writeheader()
        for t_slug, t_value in PATH_T_VALUES:
            for budget, hidden_dim in SCALES:
                _, _, batch_size, steps = build_from_heuristic(
                    budget=budget,
                    hidden_dim=hidden_dim,
                    target_steps=TARGET_STEPS,
                )
                writer.writerow(
                    {
                        "run_id": f"{RUN_ID_PREFIX}_{t_slug}_d{hidden_dim}-{budget:.2e}",
                        "candidate_id": f"v4_path_{t_slug}",
                        "t": t_value,
                        "hidden_dim": hidden_dim,
                        "budget": f"{budget:.2e}",
                        "target_steps": steps,
                        "batch_size": batch_size,
                        "phase_1_boundary_step": _phase_boundary_step(total_steps=steps, batch_size=batch_size),
                    }
                )

    summary = {
        "description": "Grug-MoE proportional-to-v4 interpolation path test.",
        "run_id_prefix": RUN_ID_PREFIX,
        "retry_reason": "r1 avoids stale executor statuses from the initial missing-asyncio failed launch.",
        "t_values": [t for _, t in PATH_T_VALUES],
        "num_training_specs": len(PATH_T_VALUES) * len(SCALES),
        "scales": [{"budget": f"{budget:.2e}", "hidden_dim": hidden_dim} for budget, hidden_dim in SCALES],
        "target_budget": TARGET_BUDGET,
        "target_steps_heuristic": TARGET_STEPS,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


grug_moe_v4_path_steps: list[ExecutorStep] = [
    _build_scale_step(t_slug, t_value, budget, hidden_dim)
    for t_slug, t_value in PATH_T_VALUES
    for budget, hidden_dim in SCALES
]


if __name__ == "__main__":
    _write_local_artifacts()
    executor_main(
        steps=grug_moe_v4_path_steps,
        description="Grug-MoE proportional-to-v4 interpolation path test at t=0.25/0.50/0.75.",
        max_concurrent=8,
    )
