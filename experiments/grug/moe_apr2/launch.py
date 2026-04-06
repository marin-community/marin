# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug MoE experiment launcher (mar26).

Pulls optimizer hyperparameters from CompletedAdamHHeuristic, runs through
the grug copy-first training pipeline with QB routing, GatedNorm, XSA.
All layers are MoE (E=64, K=4), no dense layers.
"""

import dataclasses
import math
import os

from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe_apr2.heuristic import CompletedAdamHHeuristic
from experiments.grug.moe_apr2.model import GrugModelConfig
from experiments.grug.moe_apr2.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template."""

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
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


# ============================================================
# Config
# ============================================================

SEQ_LEN: int = 4096
MIN_BATCH_SIZE: int = 32

HEURISTIC = CompletedAdamHHeuristic()

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _compute_flops_per_token(cfg: GrugModelConfig) -> float:
    """Non-embedding FLOPs per token (excludes lm_head)."""
    fpt_with_lm_head = lm_flops_per_token(
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        num_heads=cfg.num_heads,
        seq_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        glu=True,
        num_experts=cfg.num_experts,
        num_shared_experts=1 if cfg.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=cfg.num_experts_per_token,
        shared_intermediate_dim=cfg.shared_expert_intermediate_dim,
    )
    return fpt_with_lm_head - 2 * cfg.hidden_dim * cfg.vocab_size


MAX_BATCH_SIZE: int = 256


def _compute_tokens_and_batch(
    budget: float, flops_per_token: float, target_steps: int = 2**14, min_batch_size: int = MIN_BATCH_SIZE
) -> tuple[float, int, int]:
    tokens = budget / (3 * flops_per_token)
    batch_exact = tokens / (target_steps * SEQ_LEN)
    batch_size = max(min_batch_size, _round_to_power_of_two(batch_exact))
    batch_size = min(batch_size, MAX_BATCH_SIZE)
    train_steps = max(1, round(tokens / (batch_size * SEQ_LEN)))
    return tokens, batch_size, train_steps


# ============================================================
# ISOFlop sweep grid
# ============================================================

BUDGETS: tuple[float, ...] = (1e18, 3e18, 1e19, 3e19, 1e20)
HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024, 1280, 1536, 1792, 2048)

# Budget -> hidden dims to exclude
EXCLUDED_DIMS: dict[float, set[int]] = {
    1e18: {2048},
    3e19: {512},
    1e20: {512},
}

# Budget -> (target_steps multiplier, min_batch_size)
BUDGET_CONFIG: dict[float, tuple[int, int]] = {
    1e18: (1, MIN_BATCH_SIZE),
    3e18: (2, MIN_BATCH_SIZE),
    1e19: (4, MIN_BATCH_SIZE),
    3e19: (4, MIN_BATCH_SIZE),
    1e20: (8, MIN_BATCH_SIZE),
}


def create_moe_isoflop_steps() -> list[ExecutorStep]:
    """Create ExecutorSteps for the MoE isoflop grid."""
    steps: list[ExecutorStep] = []

    for budget in BUDGETS:
        excluded = EXCLUDED_DIMS.get(budget, set())
        step_mult, min_bs = BUDGET_CONFIG.get(budget, (1, MIN_BATCH_SIZE))
        target_steps = 2**14 * step_mult
        for hidden_dim in HIDDEN_DIMS:
            if hidden_dim in excluded:
                continue

            model_cfg = HEURISTIC.build_model_config(hidden_dim)
            fpt = _compute_flops_per_token(model_cfg)
            tokens, batch_size, train_steps = _compute_tokens_and_batch(
                budget, fpt, target_steps=target_steps, min_batch_size=min_bs
            )

            optimizer = HEURISTIC.build_optimizer_config(batch_size, tokens, hidden_dim)

            run_id = f"isoflop-moe-v16-{budget:.0e}-d{hidden_dim}"

            config = GrugMoeLaunchConfig(
                model=versioned(model_cfg),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=run_id,
                resources=versioned(ResourceConfig.with_tpu("v4-32")),
                steps=versioned(train_steps),
                batch_size=versioned(batch_size),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="dial_moe",
                    tags=["grug", "moe-core", "isoflop", "v16", "gqa4", f"budget={budget:.0e}", f"d={hidden_dim}"],
                    group="isoflop-moe-v16",
                    name=run_id,
                ),
                optimizer=versioned(optimizer),
                grug_trainer=versioned(
                    GrugTrainerConfig(
                        z_loss_weight=HEURISTIC.z_loss_weight,
                        ema_beta=None,
                        log_every=1,
                    )
                ),
                eval=versioned(
                    GrugEvalConfig(
                        eval_batch_size=64 if hidden_dim >= 2048 else 512,
                        steps_per_eval=1000,
                        max_eval_batches=64 if hidden_dim >= 2048 else 8,
                        eval_current=True,
                        eval_ema=False,
                    )
                ),
            )

            step = ExecutorStep(
                name=f"grug/{run_id}",
                fn=run_grug_moe_trial,
                config=config,
            )
            steps.append(step)

    return steps


moe_isoflop_steps = create_moe_isoflop_steps()


# ============================================================
# 75-run sweep: token horizon x LR grid
# ============================================================


def create_lr_grid_sweep() -> list[ExecutorStep]:
    """75-run sweep: d512/d768/d1024 x 5 token ratios x 5 Adam LRs, warmup=0.1, decay=None."""
    steps = []
    ratio = 13 / 3
    adam_lrs_base = [0.0008, 0.0016, 0.0028, 0.0036, 0.0044]
    token_ratios = [2, 4.5, 10, 22, 50]

    configs = [
        (1280, 441e6, 128),
    ]

    for hidden_dim, active_params, batch_size in configs:
        model_cfg = HEURISTIC.build_model_config(hidden_dim)
        fpt = _compute_flops_per_token(model_cfg)
        lr_scale = math.sqrt(batch_size / 32)
        adam_lrs = [round(lr * lr_scale, 5) for lr in adam_lrs_base]

        eval_bs = 64 if hidden_dim >= 2048 else 512
        eval_batches = 64 if hidden_dim >= 2048 else 8

        for tok_ratio in token_ratios:
            tokens = tok_ratio * active_params
            train_steps = max(1, round(tokens / (batch_size * SEQ_LEN)))

            for adam_lr in adam_lrs:
                adamh_lr = round(adam_lr * ratio, 5)
                optimizer = HEURISTIC.build_optimizer_config(batch_size, tokens, hidden_dim)
                optimizer = dataclasses.replace(optimizer, learning_rate=adamh_lr, adam_lr=adam_lr)

                run_id = f"isoflop-moe-v11-d{hidden_dim}-t{tok_ratio}x-adam{adam_lr}"
                config = GrugMoeLaunchConfig(
                    model=versioned(model_cfg),
                    data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                    output_path=this_output_path(),
                    run_id=run_id,
                    resources=versioned(ResourceConfig.with_tpu("v4-32")),
                    steps=versioned(train_steps),
                    batch_size=versioned(batch_size),
                    seed=versioned(0),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    tracker=WandbConfig(
                        project="dial_moe",
                        tags=[
                            "grug",
                            "moe-core",
                            "isoflop",
                            "v11",
                            f"d={hidden_dim}",
                            f"tok_ratio={tok_ratio}",
                            f"adam_lr={adam_lr}",
                            f"adamh_lr={adamh_lr}",
                            f"bs={batch_size}",
                            "decayNone",
                            "warmup=0.1",
                        ],
                        group="isoflop-moe-v11",
                        name=run_id,
                    ),
                    optimizer=versioned(optimizer),
                    grug_trainer=versioned(
                        GrugTrainerConfig(
                            z_loss_weight=HEURISTIC.z_loss_weight,
                            ema_beta=None,
                            log_every=1,
                        )
                    ),
                    eval=versioned(
                        GrugEvalConfig(
                            eval_batch_size=eval_bs,
                            steps_per_eval=1000,
                            max_eval_batches=eval_batches,
                            eval_current=True,
                            eval_ema=False,
                        )
                    ),
                )
                steps.append(ExecutorStep(name=f"grug/{run_id}", fn=run_grug_moe_trial, config=config))
    return steps


lr_grid_steps = create_lr_grid_sweep()


# ============================================================
# v12 fresh reruns (no checkpoint resume)
# (dim, tok_ratio, adam_lr, batch_size, total_steps)
# ============================================================

V12_RUNS = [
    # From v10
    (1024, 22.0, 0.0024, 128, 17624),
    (1024, 22.0, 0.008, 128, 17624),
    (1024, 50.0, 0.0008, 128, 40054),
    (1024, 50.0, 0.0024, 128, 40054),
    (1024, 50.0, 0.0048, 128, 40054),
    (1024, 50.0, 0.0064, 128, 40054),
    (1024, 50.0, 0.008, 128, 40054),
    (512, 50.0, 0.0004, 32, 58746),
    (512, 50.0, 0.0012, 32, 58746),
    (512, 50.0, 0.0024, 32, 58746),
    (512, 50.0, 0.0032, 32, 58746),
    (512, 50.0, 0.004, 32, 58746),
    (768, 22.0, 0.00339, 64, 21988),
    (768, 4.5, 0.00057, 64, 4498),
    (768, 4.5, 0.0017, 64, 4498),
    (768, 50.0, 0.0017, 64, 49973),
    # From v11
    (1024, 10.0, 0.0056, 128, 8011),
    (1024, 10.0, 0.0072, 128, 8011),
    (1024, 22.0, 0.0016, 128, 17624),
    (1024, 22.0, 0.0032, 128, 17624),
    (1024, 22.0, 0.0056, 128, 17624),
    (1024, 22.0, 0.0072, 128, 17624),
    (1024, 22.0, 0.0088, 128, 17624),
    (1024, 4.5, 0.0056, 128, 3605),
    (1024, 50.0, 0.0016, 128, 40054),
    (1024, 50.0, 0.0032, 128, 40054),
    (1024, 50.0, 0.0056, 128, 40054),
    (1024, 50.0, 0.0072, 128, 40054),
    (1024, 50.0, 0.0088, 128, 40054),
    (1280, 10.0, 0.0016, 128, 8411),
    (1280, 10.0, 0.0032, 128, 8411),
    (1280, 2.0, 0.0088, 128, 1682),
    (1280, 50.0, 0.0016, 128, 42057),
    (1280, 50.0, 0.0032, 128, 42057),
    (1280, 50.0, 0.0056, 128, 42057),
    (1280, 50.0, 0.0072, 128, 42057),
    (1280, 50.0, 0.0088, 128, 42057),
    (512, 22.0, 0.0016, 32, 25848),
    (512, 50.0, 0.0028, 32, 58746),
    (768, 50.0, 0.00509, 64, 49973),
]


def create_v12_rerun_steps() -> list[ExecutorStep]:
    """Fresh v12 reruns — no checkpoint resume."""
    ratio = 13 / 3
    steps = []

    for dim, tok_ratio, adam_lr, batch_size, train_steps in V12_RUNS:
        model_cfg = HEURISTIC.build_model_config(dim)
        fpt = _compute_flops_per_token(model_cfg)
        tokens = tok_ratio * {512: 154e6, 768: 262e6, 1024: 420e6, 1280: 441e6}[dim]
        adamh_lr = round(adam_lr * ratio, 5)
        optimizer = HEURISTIC.build_optimizer_config(batch_size, tokens, dim)
        optimizer = dataclasses.replace(optimizer, learning_rate=adamh_lr, adam_lr=adam_lr)

        run_id = f"isoflop-moe-v12-retry-d{dim}-t{tok_ratio}x-adam{adam_lr}"
        eval_bs = 64 if dim >= 2048 else 512
        eval_batches = 64 if dim >= 2048 else 8

        config = GrugMoeLaunchConfig(
            model=versioned(model_cfg),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v4-32")),
            steps=versioned(train_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="dial_moe",
                tags=[
                    "grug",
                    "moe-core",
                    "isoflop",
                    "v12",
                    f"d={dim}",
                    f"tok_ratio={tok_ratio}",
                    f"adam_lr={adam_lr}",
                    f"adamh_lr={adamh_lr}",
                    f"bs={batch_size}",
                    "decayNone",
                    "warmup=0.1",
                ],
                group="isoflop-moe-v12-retry",
                name=run_id,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=HEURISTIC.z_loss_weight,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=eval_bs,
                    steps_per_eval=1000,
                    max_eval_batches=eval_batches,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        )

        step = ExecutorStep(
            name=f"grug/{run_id}",
            fn=run_grug_moe_trial,
            config=config,
        )
        steps.append(step)

    return steps


v12_rerun_steps = create_v12_rerun_steps()


# Almost-done runs that crashed 1 step before final eval
V12_ALMOST_DONE = [
    (768, 2.0, 0.00566, 64, 1999),
    (512, 2.0, 0.0036, 32, 2350),
    (512, 2.0, 0.0044, 32, 2350),
    (512, 4.5, 0.0008, 32, 5287),
    (512, 4.5, 0.0016, 32, 5287),
    (512, 4.5, 0.0036, 32, 5287),
    (512, 50.0, 0.0036, 32, 58746),
    (512, 50.0, 0.0044, 32, 58746),
    (768, 2.0, 0.00226, 64, 1999),
    (768, 2.0, 0.00622, 64, 1999),
]

v12_almost_done_steps = (
    create_v12_rerun_steps.__wrapped__(V12_ALMOST_DONE) if hasattr(create_v12_rerun_steps, "__wrapped__") else []
)


def _create_v12_steps_from_list(run_list):
    """Create v12 steps from a list of (dim, tok_ratio, adam_lr, batch_size, total_steps)."""
    ratio = 13 / 3
    steps = []

    for dim, tok_ratio, adam_lr, batch_size, train_steps in run_list:
        model_cfg = HEURISTIC.build_model_config(dim)
        fpt = _compute_flops_per_token(model_cfg)
        tokens = tok_ratio * {512: 154e6, 768: 262e6, 1024: 420e6, 1280: 441e6}[dim]
        adamh_lr = round(adam_lr * ratio, 5)
        optimizer = HEURISTIC.build_optimizer_config(batch_size, tokens, dim)
        optimizer = dataclasses.replace(optimizer, learning_rate=adamh_lr, adam_lr=adam_lr)

        run_id = f"isoflop-moe-v12-retry-d{dim}-t{tok_ratio}x-adam{adam_lr}"
        eval_bs = 64 if dim >= 2048 else 512
        eval_batches = 64 if dim >= 2048 else 8

        config = GrugMoeLaunchConfig(
            model=versioned(model_cfg),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v4-32")),
            steps=versioned(train_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="dial_moe",
                tags=[
                    "grug",
                    "moe-core",
                    "isoflop",
                    "v12",
                    f"d={dim}",
                    f"tok_ratio={tok_ratio}",
                    f"adam_lr={adam_lr}",
                    f"adamh_lr={adamh_lr}",
                    f"bs={batch_size}",
                    "decayNone",
                    "warmup=0.1",
                ],
                group="isoflop-moe-v12-retry",
                name=run_id,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=HEURISTIC.z_loss_weight,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=eval_bs,
                    steps_per_eval=1000,
                    max_eval_batches=eval_batches,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        )
        steps.append(ExecutorStep(name=f"grug/{run_id}", fn=run_grug_moe_trial, config=config))

    return steps


v12_almost_done_steps = _create_v12_steps_from_list(V12_ALMOST_DONE)

# d1280 shifted LRs (v10-equivalent) to match d512/d768/d1024 granularity
V12_D1280_SHIFTED = [
    (1280, 2.0, 0.0008, 128, 1682),
    (1280, 2.0, 0.0024, 128, 1682),
    (1280, 2.0, 0.0048, 128, 1682),
    (1280, 2.0, 0.0064, 128, 1682),
    (1280, 2.0, 0.008, 128, 1682),
    (1280, 4.5, 0.0008, 128, 3784),
    (1280, 4.5, 0.0024, 128, 3784),
    (1280, 4.5, 0.0048, 128, 3784),
    (1280, 4.5, 0.0064, 128, 3784),
    (1280, 4.5, 0.008, 128, 3784),
    (1280, 10.0, 0.0008, 128, 8411),
    (1280, 10.0, 0.0024, 128, 8411),
    (1280, 10.0, 0.0048, 128, 8411),
    (1280, 10.0, 0.0064, 128, 8411),
    (1280, 10.0, 0.008, 128, 8411),
    (1280, 22.0, 0.0008, 128, 18504),
    (1280, 22.0, 0.0024, 128, 18504),
    (1280, 22.0, 0.0048, 128, 18504),
    (1280, 22.0, 0.0064, 128, 18504),
    (1280, 22.0, 0.008, 128, 18504),
    (1280, 50.0, 0.0008, 128, 42057),
    (1280, 50.0, 0.0024, 128, 42057),
    (1280, 50.0, 0.0048, 128, 42057),
    (1280, 50.0, 0.0064, 128, 42057),
    (1280, 50.0, 0.008, 128, 42057),
]

v12_d1280_shifted_steps = _create_v12_steps_from_list(V12_D1280_SHIFTED)


def _create_v13_retry_step(dim, tok_ratio, adam_lr, batch_size, train_steps):
    """Single v13-retry run with new name to avoid executor lock conflicts."""
    ratio = 13 / 3
    model_cfg = HEURISTIC.build_model_config(dim)
    fpt = _compute_flops_per_token(model_cfg)
    tokens = tok_ratio * {512: 154e6, 768: 262e6, 1024: 420e6, 1280: 441e6}[dim]
    adamh_lr = round(adam_lr * ratio, 5)
    optimizer = HEURISTIC.build_optimizer_config(batch_size, tokens, dim)
    optimizer = dataclasses.replace(optimizer, learning_rate=adamh_lr, adam_lr=adam_lr)

    run_id = f"isoflop-moe-v13-retry-d{dim}-t{tok_ratio}x-adam{adam_lr}"
    eval_bs = 64 if dim >= 2048 else 512
    eval_batches = 64 if dim >= 2048 else 8

    config = GrugMoeLaunchConfig(
        model=versioned(model_cfg),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_tpu("v4-32")),
        steps=versioned(train_steps),
        batch_size=versioned(batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=[
                "grug",
                "moe-core",
                "isoflop",
                "v13",
                f"d={dim}",
                f"tok_ratio={tok_ratio}",
                f"adam_lr={adam_lr}",
                f"adamh_lr={adamh_lr}",
                f"bs={batch_size}",
                "decayNone",
                "warmup=0.1",
            ],
            group="isoflop-moe-v13-retry",
            name=run_id,
        ),
        optimizer=versioned(optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=HEURISTIC.z_loss_weight,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=eval_bs,
                steps_per_eval=1000,
                max_eval_batches=eval_batches,
                eval_current=True,
                eval_ema=False,
            )
        ),
    )
    return [ExecutorStep(name=f"grug/{run_id}", fn=run_grug_moe_trial, config=config)]


# ============================================================
# GQA LR sweep: d512 t4.5x, 4:1 GQA (num_kv_heads=1), 10 LRs
# ============================================================

GQA_LR_SWEEP_LRS = [0.0004, 0.0008, 0.0012, 0.0016, 0.0024, 0.0028, 0.0032, 0.0036, 0.004, 0.0044]


def create_gqa_lr_sweep() -> list[ExecutorStep]:
    """d512 t4.5x with 4:1 GQA, sweeping 10 LRs."""
    ratio = 13 / 3
    steps = []
    dim = 512
    tok_ratio = 4.5
    batch_size = 32
    train_steps = 5286
    tokens = tok_ratio * 154e6

    model_cfg = HEURISTIC.build_model_config(dim)
    fpt = _compute_flops_per_token(model_cfg)

    for adam_lr in GQA_LR_SWEEP_LRS:
        adamh_lr = round(adam_lr * ratio, 5)
        optimizer = HEURISTIC.build_optimizer_config(batch_size, tokens, dim)
        optimizer = dataclasses.replace(optimizer, learning_rate=adamh_lr, adam_lr=adam_lr)

        run_id = f"gqa-lr-d512-t4.5x-kv1-adam{adam_lr}"
        config = GrugMoeLaunchConfig(
            model=versioned(model_cfg),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v4-32")),
            steps=versioned(train_steps),
            batch_size=versioned(batch_size),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="dial_moe",
                tags=[
                    "grug",
                    "moe-core",
                    "gqa-lr-sweep",
                    "d=512",
                    "t4.5x",
                    "kv=1",
                    "gqa4to1",
                    f"adam_lr={adam_lr}",
                    f"adamh_lr={adamh_lr}",
                ],
                group="gqa-lr-sweep",
                name=run_id,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=HEURISTIC.z_loss_weight,
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
        )
        steps.append(ExecutorStep(name=f"grug/{run_id}", fn=run_grug_moe_trial, config=config))

    return steps


gqa_lr_sweep_steps = create_gqa_lr_sweep()


def create_d640_d896_sweep() -> list[ExecutorStep]:
    """d640 and d896 across all compute budgets (intermediate_dim now rounded to 128)."""
    steps = []
    extra_dims = [640, 896]
    extra_excluded = {1e20: {640}}  # skip d640 at 1e20

    for budget in BUDGETS:
        excluded = extra_excluded.get(budget, set())
        step_mult, min_bs = BUDGET_CONFIG.get(budget, (1, MIN_BATCH_SIZE))
        target_steps = 2**14 * step_mult
        for hidden_dim in extra_dims:
            if hidden_dim in excluded:
                continue

            model_cfg = HEURISTIC.build_model_config(hidden_dim)
            fpt = _compute_flops_per_token(model_cfg)
            tokens, batch_size, train_steps = _compute_tokens_and_batch(
                budget, fpt, target_steps=target_steps, min_batch_size=min_bs
            )
            optimizer = HEURISTIC.build_optimizer_config(batch_size, tokens, hidden_dim)

            run_id = f"isoflop-moe-v16-{budget:.0e}-d{hidden_dim}"
            eval_bs = 64 if hidden_dim >= 2048 else 512
            eval_batches = 64 if hidden_dim >= 2048 else 8

            config = GrugMoeLaunchConfig(
                model=versioned(model_cfg),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=run_id,
                resources=versioned(ResourceConfig.with_tpu("v4-32")),
                steps=versioned(train_steps),
                batch_size=versioned(batch_size),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="dial_moe",
                    tags=["grug", "moe-core", "isoflop", "v16", "gqa4", f"budget={budget:.0e}", f"d={hidden_dim}"],
                    group="isoflop-moe-v16",
                    name=run_id,
                ),
                optimizer=versioned(optimizer),
                grug_trainer=versioned(
                    GrugTrainerConfig(
                        z_loss_weight=HEURISTIC.z_loss_weight,
                        ema_beta=None,
                        log_every=1,
                    )
                ),
                eval=versioned(
                    GrugEvalConfig(
                        eval_batch_size=eval_bs,
                        steps_per_eval=1000,
                        max_eval_batches=eval_batches,
                        eval_current=True,
                        eval_ema=False,
                    )
                ),
            )
            steps.append(ExecutorStep(name=f"grug/{run_id}", fn=run_grug_moe_trial, config=config))
    return steps


d640_d896_steps = create_d640_d896_sweep()


if __name__ == "__main__":
    crashed = [
        s
        for s in moe_isoflop_steps
        if s.config.run_id
        in [
            "isoflop-moe-v16-3e+19-d1536",
            "isoflop-moe-v16-3e+19-d1792",
            "isoflop-moe-v16-1e+20-d896",
            "isoflop-moe-v16-1e+20-d1280",
            "isoflop-moe-v16-1e+20-d1792",
            "isoflop-moe-v16-1e+20-d2048",
        ]
    ]
    executor_main(
        steps=crashed,
        description="v16: resubmit 6 crashed runs (checkpointing disabled)",
    )
