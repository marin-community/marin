# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dedicated H100x8 Grug MoE reproduction harness for PR 4297.

This launch file exists only on the research branch used to validate the
ragged-dot Triton kernel. It fixes a single approximately 256M-parameter, 8
expert workload so we can A/B `RAGGED_DOT_IMPL=xla` versus `triton` on the same
branch, hardware, and training path.

The executor step name is salted with the run id so repeated A/B pairs on the
research branch do not reuse the same executor output path or checkpoint state.
"""

import dataclasses
import os

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.data.text import DatasetComponent, LmDataConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe_trial
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.moe.train import GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


MARIN_CACHE_PREFIX = "s3://marin-na/marin/"


def _cache_only_data(data: LmDataConfig, broken_components: set[str]) -> LmDataConfig:
    """Freeze the harness to the subset of prebuilt caches that are actually loadable on CoreWeave CI."""
    assert isinstance(data.train_weights, dict)
    components = {}
    for name, component in data.components.items():
        if name in broken_components:
            continue
        if not isinstance(component, DatasetComponent):
            raise TypeError(f"Expected DatasetComponent for {name}, got {type(component)}")
        step = component.cache_dir.step if component.cache_dir is not None else None
        if step is None or step.override_output_path is None:
            raise ValueError(f"Expected overridden cache path for {name}, got {component.cache_dir}")
        components[name] = dataclasses.replace(
            component,
            source=None,
            cache_dir=os.path.join(MARIN_CACHE_PREFIX, step.override_output_path),
        )
    return dataclasses.replace(
        data,
        components=components,
        train_weights={name: weight for name, weight in data.train_weights.items() if name not in broken_components},
        auto_build_caches=False,
    )


PR4297_H100_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=1024,
    shared_expert_intermediate_dim=512,
    dense_intermediate_dim=1536,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=10,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=4096,
    head_dim=None,
    initializer_std=0.5 / 512**0.5,
    qk_mult=1.3,
)

PR4297_H100_VARIANT = os.environ.get("RAGGED_DOT_IMPL", "xla")
PR4297_H100_RUN_ID = _resolve_run_id("pr4297-grug-moe-h100")
BROKEN_PR4297_COMPONENTS = {
    "nemotron_cc/hq_synth",
    "nemotron_cc/medium",
    "nemotron_cc/medium_low",
    "proofpile_2",
}
PR4297_H100_DATA = _cache_only_data(nemotron_mix_block_shuffle, BROKEN_PR4297_COMPONENTS)

pr4297_h100_repro = ExecutorStep(
    name=f"grug/pr4297_h100_repro_{PR4297_H100_VARIANT}_{PR4297_H100_RUN_ID}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(PR4297_H100_MODEL),
        data=PR4297_H100_DATA,
        output_path=this_output_path(),
        run_id=PR4297_H100_RUN_ID,
        resources=versioned(ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="256g")),
        steps=versioned(100),
        batch_size=versioned(32),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "moe", "gpu", "h100x8", "pr4297", "ragged-dot"],
            group="pr4297-ragged-dot",
            name=None,
        ),
        optimizer=versioned(
            GrugMoeAdamHConfig(
                learning_rate=0.003,
                adam_lr=0.003,
                beta1=0.96,
                beta2=0.995,
                epsilon=1e-15,
                lr_schedule="linear",
                decay=0.2,
                min_lr_ratio=0.0,
                warmup=0.1,
                max_grad_norm=1,
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=None,
        profiler=ProfilerConfig(enabled=True),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[pr4297_h100_repro],
        description="PR 4297 H100x8 Grug MoE reproduction harness.",
    )
