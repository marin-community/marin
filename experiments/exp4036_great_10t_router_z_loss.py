# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Great 10T ablation: router z-loss (issue #4036).

Runs two MoE training configurations at 10T token budget to determine whether
router z-loss (logsumexp^2 penalty on router logits) improves quality at scale.

Baseline: router_z_loss_coef=0.001 (current default).
Ablation: router_z_loss_coef disabled (None).

Both arms share the same optimizer, data, and architecture.  Comparison metric
is validation perplexity on c4en and the default validation suite.
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
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix_block_shuffle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_LEN: int = 4096
VOCAB_SIZE: int = 128_256
MIN_BATCH_SIZE: int = 32

# 10 trillion tokens.
TOKEN_BUDGET: float = 10e12

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)

# ---------------------------------------------------------------------------
# Launch config (mirrors grug/moe/launch.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GreatMoeLaunchConfig:
    """Launch config for a single arm of the router z-loss ablation."""

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
    profiler: ProfilerConfig = field(default_factory=lambda: ProfilerConfig(enabled=False))


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str, output_path: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id, replicate_path=output_path)
    return tracker


def run_great_moe(config: GreatMoeLaunchConfig) -> None:
    """Map GreatMoeLaunchConfig onto TrainerConfig and run training."""
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=config.profiler,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id, config.output_path),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 5000}],
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


# ---------------------------------------------------------------------------
# Model and training arithmetic
# ---------------------------------------------------------------------------

HIDDEN_DIM = 2048
NUM_HEADS = HIDDEN_DIM // 128  # 16
NUM_KV_HEADS = NUM_HEADS


def _compute_num_layers(hidden_dim: int) -> int:
    """Depth-width formula from Marin2025Recipe."""
    hs_pow = math.log2(hidden_dim)
    return round(hidden_dim / (64 + (hs_pow * 4.0) - 9))


NUM_LAYERS = _compute_num_layers(HIDDEN_DIM)


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _build_model(*, router_z_loss_coef: float | None) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=HIDDEN_DIM // 2,
        shared_expert_intermediate_dim=HIDDEN_DIM,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        max_seq_len=SEQ_LEN,
        head_dim=None,
        load_balancing_loss_coef=0.01,
        router_z_loss_coef=router_z_loss_coef,
    )


def _compute_flops_per_token(cfg: GrugModelConfig) -> float:
    return lm_flops_per_token(
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
    )


def _compute_training_params(token_budget: float, flops_per_token: float) -> tuple[int, int]:
    """Compute batch_size and train_steps for a given token budget.

    Targets ~2^16 steps; minimum batch size 32.
    """
    target_steps = 2**16
    batch_exact = token_budget / (target_steps * SEQ_LEN)
    batch_size = max(MIN_BATCH_SIZE, _round_to_power_of_two(batch_exact))
    train_steps = max(1, round(token_budget / (batch_size * SEQ_LEN)))
    return batch_size, train_steps


# Precompute training params from the baseline model (both arms use same architecture).
_BASELINE_MODEL = _build_model(router_z_loss_coef=0.001)
_FPT = _compute_flops_per_token(_BASELINE_MODEL)
BATCH_SIZE, TRAIN_STEPS = _compute_training_params(TOKEN_BUDGET, _FPT)

# Learning rate scaled with sqrt(batch).
_EFFECTIVE_BS = BATCH_SIZE * SEQ_LEN / 4096
LR = min(0.01, (0.33 * math.sqrt(_EFFECTIVE_BS)) / HIDDEN_DIM)
BETA2 = max(0.95, 0.98 ** (_EFFECTIVE_BS / 128))

OPTIMIZER = AdamConfig(
    learning_rate=LR,
    weight_decay=0.1,
    lr_schedule="cosine",
    decay=0.2,
    min_lr_ratio=0.1,
    warmup=1000,
    beta2=BETA2,
)

TRAINER_CONFIG = GrugTrainerConfig(
    z_loss_weight=1e-4,
    ema_beta=None,
    log_every=1,
)

EVAL_CONFIG = GrugEvalConfig(
    eval_batch_size=512,
    steps_per_eval=5000,
    max_eval_batches=8,
    eval_current=True,
    eval_ema=False,
)


# ---------------------------------------------------------------------------
# Ablation arms
# ---------------------------------------------------------------------------


def _make_arm(label: str, router_z_loss_coef: float | None) -> ExecutorStep:
    model = _build_model(router_z_loss_coef=router_z_loss_coef)
    run_id = _resolve_run_id(f"great-10t-rzl-{label}")
    rzl_tag = f"rzl={router_z_loss_coef}" if router_z_loss_coef is not None else "rzl=off"
    return ExecutorStep(
        name=f"grug/great-10t-rzl-{label}",
        fn=run_great_moe,
        config=GreatMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-128")),
            steps=versioned(TRAIN_STEPS),
            batch_size=versioned(BATCH_SIZE),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "moe", "great-10t", "rzl-ablation", rzl_tag],
                group="great-10t-rzl-ablation",
                name=None,
            ),
            optimizer=versioned(OPTIMIZER),
            grug_trainer=versioned(TRAINER_CONFIG),
            eval=versioned(EVAL_CONFIG),
        ),
    )


# Baseline: router z-loss enabled (default coefficient).
great_10t_rzl_baseline = _make_arm("baseline", router_z_loss_coef=0.001)

# Ablation: router z-loss disabled.
great_10t_rzl_off = _make_arm("off", router_z_loss_coef=None)


if __name__ == "__main__":
    executor_main(
        steps=[great_10t_rzl_baseline, great_10t_rzl_off],
        description="Great 10T ablation: router z-loss (issue #4036). Two arms at 10T tokens.",
    )
