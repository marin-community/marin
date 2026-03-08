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
DEFAULT_COEFF_K8_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_v0"
DEFAULT_COEFF_K16_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k16_v0"
DEFAULT_TPU_TYPE = "v6e-8"


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
    checkpoint_minutes: int = 10
    checkpoint_keep_every_steps: int = 1_000
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


def _build_wandb_tracker(*, group: str, tags: list[str]) -> WandbConfig:
    return WandbConfig(
        entity="marin-community",
        project="tokexplore",
        tags=tags,
        group=group,
        name=None,
    )


def build_coeff_k4_data_config(store_path: str = DEFAULT_COEFF_K4_STORE_PATH) -> LmDataConfig:
    """Build the passthrough data config for the local K=4 coefficient token store."""

    return build_passthrough_lm_data_config_from_store(store_dir=store_path)


def run_jpeg_tokenizer_trial(config: JpegTokenizerLaunchConfig) -> None:
    """Run a JPEG tokenizer trial once a direct dataset has been prepared."""

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
            save_interval=timedelta(minutes=config.checkpoint_minutes),
            keep=[{"every": config.checkpoint_keep_every_steps}],
        ),
    )

    run_config = JpegRunConfig(
        model=config.model,
        token_store_path=config.token_store_path,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=dataclasses.replace(config.jpeg_trainer, trainer=trainer),
        eval=config.eval,
    )
    run_jpeg_tokenizer(run_config)


DEFAULT_JPEG_TOKENIZER_SMOKE_TRACKER = _build_wandb_tracker(
    group="tokexplore-jpeg-tokenizer-k4-smoke",
    tags=["jpeg-tokenizer", "coeff-k4", "smoke"],
)
DEFAULT_JPEG_TOKENIZER_TRACKER = _build_wandb_tracker(
    group="tokexplore-jpeg-tokenizer-k4",
    tags=["jpeg-tokenizer", "coeff-k4", "baseline"],
)

RESOLVED_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-k4-smoke")
RESOLVED_RUN_ID = _resolve_run_id("jpeg-tokenizer-k4-trial")
RESOLVED_K8_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-smoke")
RESOLVED_K8_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-trial")
RESOLVED_K8_RETRY_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-trial-r2")
RESOLVED_K16_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-k16-smoke")

coeff_k4_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k4-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(JPEG_TOKENIZER_V0_MODEL),
        token_store_path=str(DEFAULT_COEFF_K4_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(128),
        batch_size=versioned(256),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=DEFAULT_JPEG_TOKENIZER_SMOKE_TRACKER,
        checkpoint_minutes=versioned(2),
        checkpoint_keep_every_steps=versioned(500),
        optimizer=versioned(
            AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=64,
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
                eval_batch_size=64,
                steps_per_eval=64,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k4_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k4-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(JPEG_TOKENIZER_V0_MODEL),
        token_store_path=str(DEFAULT_COEFF_K4_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=DEFAULT_JPEG_TOKENIZER_TRACKER,
        checkpoint_minutes=versioned(2),
        checkpoint_keep_every_steps=versioned(500),
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
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k8_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k8-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, max_seq_len=8_192)),
        token_store_path=str(DEFAULT_COEFF_K8_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K8_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(96),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k8-smoke",
            tags=["jpeg-tokenizer", "coeff-k8", "smoke"],
        ),
        checkpoint_minutes=versioned(2),
        checkpoint_keep_every_steps=versioned(500),
        optimizer=versioned(
            AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=64,
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
                eval_batch_size=32,
                steps_per_eval=48,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k8_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k8-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, max_seq_len=8_192)),
        token_store_path=str(DEFAULT_COEFF_K8_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K8_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k8",
            tags=["jpeg-tokenizer", "coeff-k8", "baseline"],
        ),
        checkpoint_minutes=versioned(2),
        checkpoint_keep_every_steps=versioned(500),
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
                eval_batch_size=32,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k8_trial_retry = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k8-trial-r2",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, max_seq_len=8_192)),
        token_store_path=str(DEFAULT_COEFF_K8_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K8_RETRY_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k8",
            tags=["jpeg-tokenizer", "coeff-k8", "baseline", "retry"],
        ),
        checkpoint_minutes=versioned(2),
        checkpoint_keep_every_steps=versioned(500),
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
                eval_batch_size=32,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k16_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k16-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, max_seq_len=16_384)),
        token_store_path=str(DEFAULT_COEFF_K16_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K16_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(64),
        batch_size=versioned(64),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k16-smoke",
            tags=["jpeg-tokenizer", "coeff-k16", "smoke"],
        ),
        checkpoint_minutes=versioned(2),
        checkpoint_keep_every_steps=versioned(500),
        optimizer=versioned(
            AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=32,
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
                eval_batch_size=16,
                steps_per_eval=32,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[coeff_k4_smoke, coeff_k4_trial, coeff_k8_smoke, coeff_k8_trial, coeff_k8_trial_retry, coeff_k16_smoke],
        description="JPEG tokenizer coefficient runs on Imagenette token stores.",
    )


__all__ = [
    "DEFAULT_COEFF_K4_STORE_PATH",
    "DEFAULT_COEFF_K8_STORE_PATH",
    "DEFAULT_COEFF_K16_STORE_PATH",
    "DEFAULT_JPEG_TOKENIZER_SMOKE_TRACKER",
    "DEFAULT_JPEG_TOKENIZER_TRACKER",
    "DEFAULT_TPU_TYPE",
    "RESOLVED_K8_RETRY_RUN_ID",
    "RESOLVED_K8_RUN_ID",
    "RESOLVED_K8_SMOKE_RUN_ID",
    "RESOLVED_K16_SMOKE_RUN_ID",
    "RESOLVED_RUN_ID",
    "RESOLVED_SMOKE_RUN_ID",
    "JpegTokenizerLaunchConfig",
    "build_coeff_k4_data_config",
    "coeff_k4_smoke",
    "coeff_k4_trial",
    "coeff_k8_smoke",
    "coeff_k8_trial",
    "coeff_k8_trial_retry",
    "coeff_k16_smoke",
    "run_jpeg_tokenizer_trial",
]
