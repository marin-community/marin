# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE hyperparameter sweep at 3e18 FLOPs.

Explores core MoE architecture knobs (expert count, shared expert, routing
density, aux loss coefficients) at the 3e18 FLOP budget to establish a
reference recipe for the 10T scaling path.

Each config targets 3e18 total training FLOPs (including 3x fwd+bwd multiplier).
All runs use the grug MoE template on Nemotron mix with seq_len=4096 and
vocab=128256.

See https://github.com/marin-community/marin/issues/4018.
"""

import dataclasses
import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix_block_shuffle

logger = logging.getLogger(__name__)

TARGET_FLOPS = 3e18
SEQ_LEN = 4096
VOCAB_SIZE = 128_256
BATCH_SIZE = 256

NEMOTRON_MIX_WITH_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _steps_for_budget(model: GrugModelConfig, batch_size: int) -> int:
    """Compute training steps to hit TARGET_FLOPS for a given model config."""
    fpt = lm_flops_per_token(
        hidden_dim=model.hidden_dim,
        intermediate_dim=model.intermediate_dim,
        shared_intermediate_dim=model.shared_expert_intermediate_dim,
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        num_heads=model.num_heads,
        seq_len=model.max_seq_len,
        vocab_size=model.vocab_size,
        glu=True,
        num_experts=model.num_experts,
        num_shared_experts=1 if model.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model.num_experts_per_token,
    )
    flops_per_step = 3 * fpt * model.max_seq_len * batch_size
    return int(TARGET_FLOPS / flops_per_step)


# ---------------------------------------------------------------------------
# Baseline model: d=768, L=12, E=8, K=2, shared expert
# ~1300 steps at bs=256 → ~1.4B tokens
# ---------------------------------------------------------------------------
BASELINE = GrugModelConfig(
    vocab_size=VOCAB_SIZE,
    hidden_dim=768,
    intermediate_dim=2048,
    shared_expert_intermediate_dim=2048,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=12,
    num_heads=12,
    num_kv_heads=4,
    max_seq_len=SEQ_LEN,
    head_dim=None,
    load_balancing_loss_coef=0.01,
    router_z_loss_coef=0.001,
)


@dataclass(frozen=True)
class SweepPoint:
    """One arm of the sweep."""

    name: str
    model: GrugModelConfig
    batch_size: int = BATCH_SIZE


def _expert_count_variants() -> list[SweepPoint]:
    """Axis 1: E in {8, 16, 32} with K=2 (FLOPs ~constant)."""
    return [
        SweepPoint("e8-k2", BASELINE),
        SweepPoint("e16-k2", dataclasses.replace(BASELINE, num_experts=16)),
        SweepPoint("e32-k2", dataclasses.replace(BASELINE, num_experts=32)),
    ]


def _shared_expert_variants() -> list[SweepPoint]:
    """Axis 2: shared expert on vs off."""
    no_shared = dataclasses.replace(BASELINE, shared_expert_intermediate_dim=0)
    return [
        SweepPoint("shared-on", BASELINE),
        SweepPoint("shared-off", no_shared),
    ]


def _routing_density_variants() -> list[SweepPoint]:
    """Axis 3: K=2 vs K=4. K=4 doubles routed MLP FLOPs so we halve intermediate_dim."""
    k4 = dataclasses.replace(
        BASELINE,
        num_experts_per_token=4,
        intermediate_dim=1024,
        shared_expert_intermediate_dim=1024,
    )
    return [
        SweepPoint("k2-i2048", BASELINE),
        SweepPoint("k4-i1024", k4),
    ]


def _aux_loss_variants() -> list[SweepPoint]:
    """Axis 4: aux loss coefficient grid."""
    points = []
    for lbl in [0.001, 0.01, 0.1]:
        for rzl in [0.0, 0.001]:
            name = f"lbl{lbl}-rzl{rzl}"
            model = dataclasses.replace(
                BASELINE,
                load_balancing_loss_coef=lbl,
                router_z_loss_coef=rzl if rzl > 0 else None,
            )
            points.append(SweepPoint(name, model))
    return points


def all_sweep_points() -> list[SweepPoint]:
    """Deduplicated union of all sweep axes."""
    seen: set[str] = set()
    points: list[SweepPoint] = []
    for axis_fn in [_expert_count_variants, _shared_expert_variants, _routing_density_variants, _aux_loss_variants]:
        for pt in axis_fn():
            if pt.name not in seen:
                seen.add(pt.name)
                points.append(pt)
    return points


def _resolve_run_id(sweep_name: str) -> str:
    run_id = f"moe-3e18-{sweep_name}"
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _build_step(point: SweepPoint) -> ExecutorStep:
    """Build an ExecutorStep for one sweep arm."""
    steps = _steps_for_budget(point.model, point.batch_size)
    run_id = _resolve_run_id(point.name)
    logger.info("Sweep point %s: %d steps at bs=%d", point.name, steps, point.batch_size)

    config = GrugMoeLaunchConfig(
        model=versioned(point.model),
        data=NEMOTRON_MIX_WITH_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(steps),
        batch_size=versioned(point.batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["moe", "3e18", "hparam-sweep"],
            group="moe-3e18-sweep",
            name=None,
        ),
        optimizer=versioned(
            AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=200,
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=256,
                steps_per_eval=200,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    )

    return ExecutorStep(
        name=f"moe-3e18/{point.name}",
        fn=run_grug_moe,
        config=config,
    )


def build_sweep_steps() -> list[ExecutorStep]:
    return [_build_step(pt) for pt in all_sweep_points()]


if __name__ == "__main__":
    executor_main(
        steps=build_sweep_steps(),
        description="MoE hparam sweep at 3e18 FLOPs (issue #4018).",
    )
