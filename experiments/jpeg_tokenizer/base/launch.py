# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
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
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.jpeg_tokenizer.base.data import build_passthrough_lm_data_config_from_store
from experiments.jpeg_tokenizer.base.model import JPEG_TOKENIZER_V0_MODEL, JpegLmConfig
from experiments.jpeg_tokenizer.base.train import JpegEvalConfig, JpegRunConfig, JpegTrainerConfig, run_jpeg_tokenizer

DEFAULT_COEFF_K4_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0"


@dataclass(frozen=True)
class JpegTokenizerLaunchConfig:
    """Last-mile launch config for the first JPEG tokenizer training runs."""

    model: JpegLmConfig
    token_store_path: str
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    jpeg_trainer: JpegTrainerConfig = field(default_factory=JpegTrainerConfig)
    eval: JpegEvalConfig | None = field(default_factory=JpegEvalConfig)


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("JPEG_TOKENIZER_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def build_coeff_k4_data_config(store_path: str = DEFAULT_COEFF_K4_STORE_PATH) -> LmDataConfig:
    """Build the passthrough data config for the local K=4 coefficient token store."""

    return build_passthrough_lm_data_config_from_store(store_dir=store_path)


def run_jpeg_tokenizer_trial(config: JpegTokenizerLaunchConfig) -> None:
    """Run a JPEG tokenizer trial once a direct dataset has been prepared."""

    data_config = build_coeff_k4_data_config(config.token_store_path)
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=f"{config.output_path}/checkpoints",
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )

    run_config = JpegRunConfig(
        model=config.model,
        data=data_config,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=dataclasses.replace(config.jpeg_trainer, trainer=trainer),
        eval=config.eval,
    )
    run_jpeg_tokenizer(run_config)


DEFAULT_JPEG_TOKENIZER_TRACKER = WandbConfig(
    entity="marin-community",
    project="tokexplore",
    tags=["jpeg-tokenizer", "coeff-k4", "template"],
    group="tokexplore-jpeg-tokenizer-k4",
    name=None,
)

RESOLVED_RUN_ID = _resolve_run_id("jpeg-tokenizer-k4-trial")

coeff_k4_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k4-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(JPEG_TOKENIZER_V0_MODEL),
        token_store_path=str(DEFAULT_COEFF_K4_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(2_000),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=DEFAULT_JPEG_TOKENIZER_TRACKER,
        optimizer=versioned(
            AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=1_000,
            )
        ),
        jpeg_trainer=versioned(
            JpegTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=128,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[coeff_k4_trial],
        description="JPEG tokenizer K=4 coefficient baseline on the local Imagenette token store.",
    )


__all__ = [
    "DEFAULT_COEFF_K4_STORE_PATH",
    "DEFAULT_JPEG_TOKENIZER_TRACKER",
    "RESOLVED_RUN_ID",
    "JpegTokenizerLaunchConfig",
    "build_coeff_k4_data_config",
    "coeff_k4_trial",
    "run_jpeg_tokenizer_trial",
]
