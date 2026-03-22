# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the iteration_02 Grug MoE experiments."""

import dataclasses
import math
import os

from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig, TpuConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe_scaling_iteration_02.model import GrugModelConfig
from experiments.grug.moe_scaling_iteration_02.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
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


def _mesh_expert_axis(resources: ResourceConfig) -> int:
    device = resources.device
    if isinstance(device, TpuConfig) and device.variant.startswith("v5p-"):
        return 1
    return 4


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    trainer = dataclasses.replace(
        config.grug_trainer.trainer,
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": _mesh_expert_axis(config.resources)}),
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


SEQ_LEN: int = 4096
VOCAB_SIZE: int = 128_256
MIN_BATCH_SIZE: int = 32

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _compute_num_layers(hidden_dim: int) -> int:
    hs_pow = math.log2(hidden_dim)
    return round(hidden_dim / (64 + (hs_pow * 4.0) - 9))


def _build_model_config(hidden_dim: int) -> GrugModelConfig:
    num_heads = hidden_dim // 128
    num_layers = _compute_num_layers(hidden_dim)
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim // 2,
        shared_expert_intermediate_dim=hidden_dim,
        num_experts=64,
        num_experts_per_token=4,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        max_seq_len=SEQ_LEN,
        num_dense_layers=2,
        dense_intermediate_dim=3 * hidden_dim,
        bias_update_rate=0.01,
        load_balancing_loss_coef=0.001,
        sliding_window=4096,
        initializer_std=0.5 / math.sqrt(hidden_dim),
        qk_mult=1.3,
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


def _compute_tokens_and_batch(budget: float, flops_per_token: float) -> tuple[float, int, int]:
    tokens = budget / (3 * flops_per_token)
    target_steps = 2**14
    batch_exact = tokens / (target_steps * SEQ_LEN)
    batch_size = max(MIN_BATCH_SIZE, _round_to_power_of_two(batch_exact))
    train_steps = max(1, round(tokens / (batch_size * SEQ_LEN)))
    return tokens, batch_size, train_steps
