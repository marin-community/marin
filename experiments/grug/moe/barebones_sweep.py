# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Barebones transformer: no XSA, no GatedNorm, no attn gate, no MoE, MHA.

Dense MLP (3x hidden_dim), QK norm, RoPE, sliding window every 4th + last.
Tests 4 configs: {AdamH, Muon} x {no PKO, PKO on long+last layers}.

GitHub issue: https://github.com/marin-community/marin/issues/TBD
"""

import dataclasses
import math

from fray.cluster import ResourceConfig
from levanter.optim.grugmuon import GrugMuonConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import MoeAdamHHeuristic, compute_flops_per_token, compute_tokens_and_batch
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

GATE1_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
]

SEQ_LEN = 4096


def _build_barebones_model(hidden_dim: int) -> GrugModelConfig:
    """Build a barebones model config: MHA, dense MLP at 3x width, no MoE."""
    heuristic = MoeAdamHHeuristic()
    num_layers = heuristic._compute_num_layers(hidden_dim)
    num_heads = max(1, hidden_dim // 128)
    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim * 3,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=SEQ_LEN,
        sliding_window=SEQ_LEN,
        initializer_std=0.5 / math.sqrt(hidden_dim),
        qk_mult=1.3,
    )


def _build_adamh_optimizer(hidden_dim: int, batch_size: int, tokens: float) -> GrugMoeAdamHConfig:
    heuristic = MoeAdamHHeuristic()
    return heuristic.build_optimizer_config(batch_size, tokens, hidden_dim)


def _build_muon_optimizer(warmup: float = 0.1) -> GrugMuonConfig:
    return GrugMuonConfig(
        learning_rate=0.02,
        adam_lr=6e-4,
        warmup=warmup,
        lr_schedule="linear",
        min_lr_ratio=0.0,
    )


CONFIGS: list[tuple[str, bool]] = [
    # ("adamh", False),  # already submitted
    # ("adamh", True),  # already submitted
    # ("muon", False),  # already submitted
    # ("muon", True),  # already submitted
    ("muon-nowarmup", False),
    ("muon-nowarmup", True),
]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for opt_name, use_pko in CONFIGS:
        for dim, budget in GATE1_CONFIGS:
            model = _build_barebones_model(dim)
            if use_pko:
                model = dataclasses.replace(model, partial_key_offset="every_4th", last_layer_pko=True)

            fpt = compute_flops_per_token(model)
            tokens, batch, num_steps = compute_tokens_and_batch(budget, fpt)

            if opt_name == "adamh":
                optimizer = _build_adamh_optimizer(dim, batch, tokens)
            elif opt_name == "muon-nowarmup":
                optimizer = _build_muon_optimizer(warmup=0.0)
            else:
                optimizer = _build_muon_optimizer()

            pko_label = "pko" if use_pko else "nopko"
            run_id = f"barebones-{opt_name}-{pko_label}-d{dim}-{budget:.2e}"

            steps.append(
                ExecutorStep(
                    name=f"grug/{run_id}",
                    fn=run_grug_moe_trial,
                    config=GrugMoeLaunchConfig(
                        model=versioned(model),
                        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                        output_path=this_output_path(),
                        run_id=run_id,
                        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                        enable_cross_region_ckpt_read=True,
                        steps=versioned(num_steps),
                        batch_size=versioned(batch),
                        seed=versioned(0),
                        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                        tracker=WandbConfig(
                            project="dial_moe",
                            tags=["barebones", opt_name, pko_label, f"d={dim}", f"budget={budget:.2e}"],
                            group="barebones",
                            name=run_id,
                        ),
                        optimizer=versioned(optimizer),
                        grug_trainer=versioned(
                            GrugTrainerConfig(
                                z_loss_weight=0.0,
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
            )
    return steps


all_steps = _make_steps()

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Barebones transformer: {AdamH, Muon} x {no PKO, PKO} at gate 1.",
    )
