# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from argparse import Namespace
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

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.jpeg_tokenizer.base.data import build_passthrough_lm_data_config_from_store
from experiments.jpeg_tokenizer.base.model import JPEG_TOKENIZER_V0_MODEL, JPEG_TOKENIZER_V1_LARGE_MODEL, JpegLmConfig
from experiments.jpeg_tokenizer.base.train import JpegEvalConfig, JpegRunConfig, JpegTrainerConfig, run_jpeg_tokenizer
from scripts.jpeg_tokenizer.evaluate_representation_head2head import main as evaluate_representation_head2head_main

DEFAULT_COEFF_K4_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k4_v0"
DEFAULT_COEFF_K8_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_v0"
DEFAULT_COEFF_K8_LIBJPEG_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k8_libjpeg_v0"
DEFAULT_COEFF_K16_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k16_v0"
DEFAULT_COEFF_K64_LIBJPEG_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k64_libjpeg_v0"
DEFAULT_BYTE_W8192_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_w8192_v0"
DEFAULT_BYTE_WHOLE_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_whole_v0"
DEFAULT_SCAN_BYTES_WHOLE_STORE_PATH = "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_scan_bytes_whole_v0"
DEFAULT_HUFFMAN_EVENTS_WHOLE_LIBJPEG_STORE_PATH = (
    "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_huffman_events_whole_libjpeg_v0"
)
DEFAULT_SYMBOL_WHOLE_LIBJPEG_STORE_PATH = (
    "gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0"
)
DEFAULT_TPU_TYPE = "v6e-8"
DEFAULT_BYTE_WHOLE_SEQ_LEN = 54_656
DEFAULT_SCAN_BYTES_WHOLE_SEQ_LEN = 53_760
DEFAULT_BYTE_WHOLE_SWA = 4_096
DEFAULT_HUFFMAN_EVENTS_WHOLE_SEQ_LEN = 115_840
DEFAULT_SYMBOL_WHOLE_SEQ_LEN = 58_240


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
    pip_packages: tuple[str, ...] = ()
    load_checkpoint_path: str | None = None
    checkpoint_minutes: int = 10
    checkpoint_keep_every_steps: int = 1_000
    jpeg_trainer: JpegTrainerConfig = field(default_factory=JpegTrainerConfig)
    eval: JpegEvalConfig | None = field(default_factory=JpegEvalConfig)


@dataclass(frozen=True)
class JpegRepresentationEvalLaunchConfig:
    """Launch config for whole-image representation evaluation jobs."""

    run_id: str
    resources: ResourceConfig
    output_dir: str
    run_specs: tuple[str, ...]
    split: str = "validation"
    batch_size: int = 8
    max_examples: int | None = None
    pixels_per_image: int = 256 * 256
    log_every: int = 10


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
        load_checkpoint_path=config.load_checkpoint_path,
    )

    run_config = JpegRunConfig(
        model=config.model,
        token_store_path=config.token_store_path,
        resources=config.resources,
        pip_packages=config.pip_packages,
        optimizer=config.optimizer,
        trainer=dataclasses.replace(config.jpeg_trainer, trainer=trainer),
        eval=config.eval,
    )
    run_jpeg_tokenizer(run_config)


def _run_representation_eval_local(config: JpegRepresentationEvalLaunchConfig) -> None:
    args = Namespace(
        run_spec=list(config.run_specs),
        split=config.split,
        batch_size=config.batch_size,
        max_examples=config.max_examples,
        pixels_per_image=config.pixels_per_image,
        output_dir=config.output_dir,
        log_every=config.log_every,
    )
    evaluate_representation_head2head_main(args)


def run_jpeg_representation_eval(config: JpegRepresentationEvalLaunchConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_representation_eval_local,
        resources=config.resources,
        max_retries_failure=1,
    )


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
RESOLVED_K4_MATCHED_RUN_ID = _resolve_run_id("jpeg-tokenizer-k4-trial-matched")
RESOLVED_K8_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-smoke")
RESOLVED_K8_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-trial")
RESOLVED_K8_RETRY_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-trial-r2")
RESOLVED_K8_LIBJPEG_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-libjpeg-smoke")
RESOLVED_K8_LIBJPEG_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-libjpeg-trial")
RESOLVED_K8_LIBJPEG_SWA4096_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-libjpeg-swa4096-smoke")
RESOLVED_K8_LIBJPEG_SWA4096_RUN_ID = _resolve_run_id("jpeg-tokenizer-k8-libjpeg-swa4096-trial")
RESOLVED_K16_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-k16-smoke")
RESOLVED_K16_RUN_ID = _resolve_run_id("jpeg-tokenizer-k16-trial")
RESOLVED_K64_LIBJPEG_SWA4096_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-k64-libjpeg-swa4096-smoke")
RESOLVED_K64_LIBJPEG_SWA4096_RUN_ID = _resolve_run_id("jpeg-tokenizer-k64-libjpeg-swa4096-trial")
RESOLVED_K64_LIBJPEG_SWA4096_LONG_RUN_ID = _resolve_run_id("jpeg-tokenizer-k64-libjpeg-swa4096-long")
RESOLVED_K64_LIBJPEG_LARGE_SWA4096_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-k64-libjpeg-large-swa4096-smoke")
RESOLVED_K64_LIBJPEG_LARGE_SWA4096_RUN_ID = _resolve_run_id("jpeg-tokenizer-k64-libjpeg-large-swa4096-trial")
RESOLVED_BYTES_W8192_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-bytes-w8192-smoke")
RESOLVED_BYTES_W8192_RUN_ID = _resolve_run_id("jpeg-tokenizer-bytes-w8192-trial")
RESOLVED_BYTES_WHOLE_SWA4096_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-bytes-whole-swa4096-smoke")
RESOLVED_BYTES_WHOLE_SWA4096_RUN_ID = _resolve_run_id("jpeg-tokenizer-bytes-whole-swa4096-trial")
RESOLVED_BYTES_WHOLE_SWA4096_LONG_RUN_ID = _resolve_run_id("jpeg-tokenizer-bytes-whole-swa4096-long")
RESOLVED_BYTES_WHOLE_LARGE_SWA4096_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-bytes-whole-large-swa4096-smoke")
RESOLVED_BYTES_WHOLE_LARGE_SWA4096_RUN_ID = _resolve_run_id("jpeg-tokenizer-bytes-whole-large-swa4096-trial")
RESOLVED_SCAN_BYTES_WHOLE_SWA4096_SMOKE_RUN_ID = _resolve_run_id("jpeg-tokenizer-scan-bytes-whole-swa4096-smoke")
RESOLVED_SCAN_BYTES_WHOLE_SWA4096_RUN_ID = _resolve_run_id("jpeg-tokenizer-scan-bytes-whole-swa4096-trial")
RESOLVED_HUFFMAN_EVENTS_WHOLE_LIBJPEG_SWA4096_SMOKE_RUN_ID = _resolve_run_id(
    "jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-smoke"
)
RESOLVED_HUFFMAN_EVENTS_WHOLE_LIBJPEG_SWA4096_RUN_ID = _resolve_run_id(
    "jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-trial"
)
RESOLVED_HUFFMAN_EVENTS_WHOLE_LIBJPEG_SWA4096_RETRY_RUN_ID = _resolve_run_id(
    "jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-trial-r2"
)
RESOLVED_SYMBOLS_WHOLE_LIBJPEG_SWA4096_SMOKE_RUN_ID = _resolve_run_id(
    "jpeg-tokenizer-symbols-whole-libjpeg-swa4096-smoke"
)
RESOLVED_SYMBOLS_WHOLE_LIBJPEG_SWA4096_RUN_ID = _resolve_run_id("jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial")
RESOLVED_SYMBOLS_WHOLE_LIBJPEG_SWA4096_LONG_RUN_ID = _resolve_run_id("jpeg-tokenizer-symbols-whole-libjpeg-swa4096-long")
RESOLVED_SYMBOLS_WHOLE_LIBJPEG_LARGE_SWA4096_SMOKE_RUN_ID = _resolve_run_id(
    "jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-smoke"
)
RESOLVED_SYMBOLS_WHOLE_LIBJPEG_LARGE_SWA4096_RUN_ID = _resolve_run_id(
    "jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-trial"
)
RESOLVED_REPRESENTATION_EVAL_LONG_R3_RUN_ID = _resolve_run_id("jpeg-tokenizer-representation-eval-long-r3")
RESOLVED_REPRESENTATION_EVAL_LARGE_R3_RUN_ID = _resolve_run_id("jpeg-tokenizer-representation-eval-large-r3")

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

coeff_k4_trial_matched = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k4-trial-matched",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(JPEG_TOKENIZER_V0_MODEL),
        token_store_path=str(DEFAULT_COEFF_K4_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K4_MATCHED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(256),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k4-matched",
            tags=["jpeg-tokenizer", "coeff-k4", "matched-budget"],
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

coeff_k8_libjpeg_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k8-libjpeg-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, max_seq_len=8_192)),
        token_store_path=str(DEFAULT_COEFF_K8_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K8_LIBJPEG_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        pip_packages=("jpeglib>=1.0.2",),
        steps=versioned(96),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k8-libjpeg-smoke",
            tags=["jpeg-tokenizer", "coeff-k8", "libjpeg", "smoke"],
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

coeff_k8_libjpeg_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k8-libjpeg-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, max_seq_len=8_192)),
        token_store_path=str(DEFAULT_COEFF_K8_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K8_LIBJPEG_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        pip_packages=("jpeglib>=1.0.2",),
        steps=versioned(2_000),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k8-libjpeg",
            tags=["jpeg-tokenizer", "coeff-k8", "libjpeg", "baseline"],
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

coeff_k8_libjpeg_swa4096_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k8-libjpeg-swa4096-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                max_seq_len=8_192,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_COEFF_K8_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K8_LIBJPEG_SWA4096_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        pip_packages=("jpeglib>=1.0.2",),
        steps=versioned(96),
        batch_size=versioned(56),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-swa-head2head-smoke",
            tags=["jpeg-tokenizer", "coeff-k8", "libjpeg", "swa4096", "smoke", "head2head"],
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
                eval_batch_size=16,
                steps_per_eval=48,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k8_libjpeg_swa4096_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k8-libjpeg-swa4096-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                max_seq_len=8_192,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_COEFF_K8_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K8_LIBJPEG_SWA4096_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        pip_packages=("jpeglib>=1.0.2",),
        steps=versioned(2_000),
        batch_size=versioned(56),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-swa-head2head",
            tags=["jpeg-tokenizer", "coeff-k8", "libjpeg", "swa4096", "baseline", "head2head"],
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
                eval_batch_size=16,
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

coeff_k16_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k16-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, max_seq_len=16_384)),
        token_store_path=str(DEFAULT_COEFF_K16_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K16_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(64),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k16",
            tags=["jpeg-tokenizer", "coeff-k16", "baseline"],
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
                eval_batch_size=16,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k64_libjpeg_swa4096_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                max_seq_len=65_536,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_COEFF_K64_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K64_LIBJPEG_SWA4096_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        pip_packages=("jpeglib>=1.0.2",),
        steps=versioned(64),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k64-libjpeg-swa4096-smoke",
            tags=["jpeg-tokenizer", "coeff-k64", "libjpeg", "swa4096", "smoke"],
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
                eval_batch_size=8,
                steps_per_eval=32,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k64_libjpeg_swa4096_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                max_seq_len=65_536,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_COEFF_K64_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K64_LIBJPEG_SWA4096_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        pip_packages=("jpeglib>=1.0.2",),
        steps=versioned(2_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-k64-libjpeg-swa4096",
            tags=["jpeg-tokenizer", "coeff-k64", "libjpeg", "swa4096", "baseline"],
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
                eval_batch_size=8,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k64_libjpeg_swa4096_long = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-long",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=4_095,
                max_seq_len=65_536,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_COEFF_K64_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K64_LIBJPEG_SWA4096_LONG_RUN_ID,
        load_checkpoint_path=versioned(
            "gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-trial-7e3e81/checkpoints"
        ),
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(8_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-long",
            tags=["jpeg-tokenizer", "coeff-k64", "whole-image", "libjpeg", "swa4096", "long"],
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
        jpeg_trainer=versioned(JpegTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=8,
                steps_per_eval=2_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k64_libjpeg_large_swa4096_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k64-libjpeg-large-swa4096-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V1_LARGE_MODEL,
                vocab_size=4_095,
                max_seq_len=65_536,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_COEFF_K64_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K64_LIBJPEG_LARGE_SWA4096_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(32),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-large-smoke",
            tags=["jpeg-tokenizer", "coeff-k64", "whole-image", "libjpeg", "swa4096", "large", "smoke"],
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
                warmup=16,
            )
        ),
        jpeg_trainer=versioned(JpegTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=8,
                steps_per_eval=16,
                max_eval_batches=2,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

coeff_k64_libjpeg_large_swa4096_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-k64-libjpeg-large-swa4096-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V1_LARGE_MODEL,
                vocab_size=4_095,
                max_seq_len=65_536,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_COEFF_K64_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_K64_LIBJPEG_LARGE_SWA4096_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-large",
            tags=["jpeg-tokenizer", "coeff-k64", "whole-image", "libjpeg", "swa4096", "large", "trial"],
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
        jpeg_trainer=versioned(JpegTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=8,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

bytes_w8192_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-bytes-w8192-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, vocab_size=257, max_seq_len=8_192)),
        token_store_path=str(DEFAULT_BYTE_W8192_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_BYTES_W8192_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(96),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-bytes-w8192-smoke",
            tags=["jpeg-tokenizer", "bytes", "window-8192", "smoke"],
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

bytes_w8192_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-bytes-w8192-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(dataclasses.replace(JPEG_TOKENIZER_V0_MODEL, vocab_size=257, max_seq_len=8_192)),
        token_store_path=str(DEFAULT_BYTE_W8192_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_BYTES_W8192_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(128),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-bytes-w8192",
            tags=["jpeg-tokenizer", "bytes", "window-8192", "baseline"],
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

bytes_whole_swa4096_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-bytes-whole-swa4096-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=258,
                max_seq_len=DEFAULT_BYTE_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_BYTE_WHOLE_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_BYTES_WHOLE_SWA4096_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(32),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-bytes-whole-swa4096-smoke",
            tags=["jpeg-tokenizer", "bytes", "whole-image", "swa4096", "smoke"],
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
                warmup=16,
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
                eval_batch_size=8,
                steps_per_eval=16,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

bytes_whole_swa4096_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-bytes-whole-swa4096-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=258,
                max_seq_len=DEFAULT_BYTE_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_BYTE_WHOLE_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_BYTES_WHOLE_SWA4096_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-swa-head2head",
            tags=["jpeg-tokenizer", "bytes", "whole-image", "swa4096", "baseline", "head2head"],
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
                eval_batch_size=8,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

bytes_whole_swa4096_long = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-bytes-whole-swa4096-long",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=258,
                max_seq_len=DEFAULT_BYTE_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_BYTE_WHOLE_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_BYTES_WHOLE_SWA4096_LONG_RUN_ID,
        load_checkpoint_path=versioned(
            "gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-swa4096-trial-7cc718/checkpoints"
        ),
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(8_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-long",
            tags=["jpeg-tokenizer", "bytes", "whole-image", "swa4096", "long"],
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
        jpeg_trainer=versioned(JpegTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=8,
                steps_per_eval=2_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

bytes_whole_large_swa4096_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-bytes-whole-large-swa4096-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V1_LARGE_MODEL,
                vocab_size=258,
                max_seq_len=DEFAULT_BYTE_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_BYTE_WHOLE_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_BYTES_WHOLE_LARGE_SWA4096_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(32),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-large-smoke",
            tags=["jpeg-tokenizer", "bytes", "whole-image", "swa4096", "large", "smoke"],
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
                warmup=16,
            )
        ),
        jpeg_trainer=versioned(JpegTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=8,
                steps_per_eval=16,
                max_eval_batches=2,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

bytes_whole_large_swa4096_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-bytes-whole-large-swa4096-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V1_LARGE_MODEL,
                vocab_size=258,
                max_seq_len=DEFAULT_BYTE_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_BYTE_WHOLE_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_BYTES_WHOLE_LARGE_SWA4096_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-large",
            tags=["jpeg-tokenizer", "bytes", "whole-image", "swa4096", "large", "trial"],
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
        jpeg_trainer=versioned(JpegTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=8,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

scan_bytes_whole_swa4096_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-scan-bytes-whole-swa4096-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=258,
                max_seq_len=DEFAULT_SCAN_BYTES_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_SCAN_BYTES_WHOLE_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_SCAN_BYTES_WHOLE_SWA4096_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(32),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-middle-ground-smoke",
            tags=["jpeg-tokenizer", "scan-bytes", "whole-image", "swa4096", "smoke", "middle-ground"],
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
                warmup=16,
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
                eval_batch_size=8,
                steps_per_eval=16,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

scan_bytes_whole_swa4096_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-scan-bytes-whole-swa4096-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=258,
                max_seq_len=DEFAULT_SCAN_BYTES_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_SCAN_BYTES_WHOLE_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_SCAN_BYTES_WHOLE_SWA4096_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-middle-ground",
            tags=["jpeg-tokenizer", "scan-bytes", "whole-image", "swa4096", "baseline", "middle-ground"],
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
                eval_batch_size=8,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

huffman_events_whole_libjpeg_swa4096_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=2_224,
                max_seq_len=DEFAULT_HUFFMAN_EVENTS_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_HUFFMAN_EVENTS_WHOLE_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_HUFFMAN_EVENTS_WHOLE_LIBJPEG_SWA4096_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(32),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-middle-ground-smoke",
            tags=["jpeg-tokenizer", "huffman-events", "whole-image", "libjpeg", "swa4096", "smoke", "middle-ground"],
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
                warmup=16,
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
                eval_batch_size=8,
                steps_per_eval=16,
                max_eval_batches=2,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

huffman_events_whole_libjpeg_swa4096_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=2_224,
                max_seq_len=DEFAULT_HUFFMAN_EVENTS_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_HUFFMAN_EVENTS_WHOLE_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_HUFFMAN_EVENTS_WHOLE_LIBJPEG_SWA4096_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-middle-ground",
            tags=["jpeg-tokenizer", "huffman-events", "whole-image", "libjpeg", "swa4096", "baseline", "middle-ground"],
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
                eval_batch_size=8,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

huffman_events_whole_libjpeg_swa4096_retry = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-trial-r2",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=2_224,
                max_seq_len=DEFAULT_HUFFMAN_EVENTS_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_HUFFMAN_EVENTS_WHOLE_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_HUFFMAN_EVENTS_WHOLE_LIBJPEG_SWA4096_RETRY_RUN_ID,
        load_checkpoint_path=versioned(
            "gs://marin-eu-west4/tokexplore/jpeg-tokenizer-huffman-events-whole-libjpeg-swa4096-trial-dcbebc/checkpoints"
        ),
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-middle-ground",
            tags=["jpeg-tokenizer", "huffman-events", "whole-image", "libjpeg", "swa4096", "baseline", "retry"],
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
                eval_batch_size=8,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

symbols_whole_libjpeg_swa4096_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=36_835,
                max_seq_len=DEFAULT_SYMBOL_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_SYMBOL_WHOLE_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_SYMBOLS_WHOLE_LIBJPEG_SWA4096_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(32),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-symbols-whole-libjpeg-swa4096-smoke",
            tags=["jpeg-tokenizer", "symbols", "whole-image", "libjpeg", "swa4096", "smoke"],
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
                warmup=16,
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
                eval_batch_size=8,
                steps_per_eval=16,
                max_eval_batches=4,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

symbols_whole_libjpeg_swa4096_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=36_835,
                max_seq_len=DEFAULT_SYMBOL_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_SYMBOL_WHOLE_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_SYMBOLS_WHOLE_LIBJPEG_SWA4096_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-swa-head2head",
            tags=["jpeg-tokenizer", "symbols", "whole-image", "libjpeg", "swa4096", "baseline", "head2head"],
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
                eval_batch_size=8,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

symbols_whole_libjpeg_swa4096_long = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-long",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V0_MODEL,
                vocab_size=36_835,
                max_seq_len=DEFAULT_SYMBOL_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_SYMBOL_WHOLE_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_SYMBOLS_WHOLE_LIBJPEG_SWA4096_LONG_RUN_ID,
        load_checkpoint_path=versioned(
            "gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-trial-a844e3/checkpoints"
        ),
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(8_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-long",
            tags=["jpeg-tokenizer", "symbols", "whole-image", "libjpeg", "swa4096", "long"],
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
        jpeg_trainer=versioned(JpegTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=8,
                steps_per_eval=2_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

symbols_whole_libjpeg_large_swa4096_smoke = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-smoke",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V1_LARGE_MODEL,
                vocab_size=36_835,
                max_seq_len=DEFAULT_SYMBOL_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_SYMBOL_WHOLE_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_SYMBOLS_WHOLE_LIBJPEG_LARGE_SWA4096_SMOKE_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(32),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-large-smoke",
            tags=["jpeg-tokenizer", "symbols", "whole-image", "libjpeg", "swa4096", "large", "smoke"],
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
                warmup=16,
            )
        ),
        jpeg_trainer=versioned(JpegTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=8,
                steps_per_eval=16,
                max_eval_batches=2,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

symbols_whole_libjpeg_large_swa4096_trial = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-trial",
    fn=run_jpeg_tokenizer_trial,
    config=JpegTokenizerLaunchConfig(
        model=versioned(
            dataclasses.replace(
                JPEG_TOKENIZER_V1_LARGE_MODEL,
                vocab_size=36_835,
                max_seq_len=DEFAULT_SYMBOL_WHOLE_SEQ_LEN,
                sliding_window=DEFAULT_BYTE_WHOLE_SWA,
            )
        ),
        token_store_path=str(DEFAULT_SYMBOL_WHOLE_LIBJPEG_STORE_PATH),
        output_path=this_output_path(),
        run_id=RESOLVED_SYMBOLS_WHOLE_LIBJPEG_LARGE_SWA4096_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        steps=versioned(2_000),
        batch_size=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_build_wandb_tracker(
            group="tokexplore-jpeg-tokenizer-large",
            tags=["jpeg-tokenizer", "symbols", "whole-image", "libjpeg", "swa4096", "large", "trial"],
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
        jpeg_trainer=versioned(JpegTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            JpegEvalConfig(
                eval_batch_size=8,
                steps_per_eval=1_000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
                compute_bpb=False,
            )
        ),
    ),
)

representation_eval_long_r3 = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-representation-eval-long-r3",
    fn=run_jpeg_representation_eval,
    config=JpegRepresentationEvalLaunchConfig(
        run_id=RESOLVED_REPRESENTATION_EVAL_LONG_R3_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        output_dir="gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-long-r3",
        run_specs=(
            "name=coeff_k64_long,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-swa4096-long-5272ec/checkpoints/step-8000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k64_libjpeg_v0,sliding_window=4096,unit_name=block,unit_count=1024",
            "name=symbols_long,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-swa4096-long-b4aa28/checkpoints/step-8000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0,sliding_window=4096",
            "name=bytes_long,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-swa4096-long-64db87/checkpoints/step-8000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_whole_v0,sliding_window=4096",
        ),
        batch_size=8,
    ),
)

representation_eval_large_r3 = ExecutorStep(
    name="tokexplore/jpeg-tokenizer-representation-eval-large-r3",
    fn=run_jpeg_representation_eval,
    config=JpegRepresentationEvalLaunchConfig(
        run_id=RESOLVED_REPRESENTATION_EVAL_LARGE_R3_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(DEFAULT_TPU_TYPE)),
        output_dir="gs://marin-eu-west4/tokexplore/jpeg-tokenizer-representation-eval-large-r3",
        run_specs=(
            "name=coeff_k64_large,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-k64-libjpeg-large-swa4096-trial-de16b2/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_coeff_k64_libjpeg_v0,sliding_window=4096,unit_name=block,unit_count=1024",
            "name=symbols_large,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-symbols-whole-libjpeg-large-swa4096-trial-4b09ce/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_symbols_whole_libjpeg_v0,sliding_window=4096",
            "name=bytes_large,checkpoint=gs://marin-eu-west4/tokexplore/jpeg-tokenizer-bytes-whole-large-swa4096-trial-f64948/checkpoints/step-2000,store=gs://marin-eu-west4/jpeg_tokenizer/token_store/imagenette_bytes_whole_v0,sliding_window=4096",
        ),
        batch_size=8,
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[
            coeff_k4_smoke,
            coeff_k4_trial,
            coeff_k4_trial_matched,
            coeff_k8_smoke,
            coeff_k8_trial,
            coeff_k8_trial_retry,
            coeff_k8_libjpeg_smoke,
            coeff_k8_libjpeg_trial,
            coeff_k8_libjpeg_swa4096_smoke,
            coeff_k8_libjpeg_swa4096_trial,
            coeff_k16_smoke,
            coeff_k16_trial,
            coeff_k64_libjpeg_swa4096_smoke,
            coeff_k64_libjpeg_swa4096_trial,
            coeff_k64_libjpeg_swa4096_long,
            coeff_k64_libjpeg_large_swa4096_smoke,
            coeff_k64_libjpeg_large_swa4096_trial,
            bytes_w8192_smoke,
            bytes_w8192_trial,
            bytes_whole_swa4096_smoke,
            bytes_whole_swa4096_trial,
            bytes_whole_swa4096_long,
            bytes_whole_large_swa4096_smoke,
            bytes_whole_large_swa4096_trial,
            scan_bytes_whole_swa4096_smoke,
            scan_bytes_whole_swa4096_trial,
            huffman_events_whole_libjpeg_swa4096_smoke,
            huffman_events_whole_libjpeg_swa4096_trial,
            huffman_events_whole_libjpeg_swa4096_retry,
            symbols_whole_libjpeg_swa4096_smoke,
            symbols_whole_libjpeg_swa4096_trial,
            symbols_whole_libjpeg_swa4096_long,
            symbols_whole_libjpeg_large_swa4096_smoke,
            symbols_whole_libjpeg_large_swa4096_trial,
            representation_eval_long_r3,
            representation_eval_large_r3,
        ],
        description=(
            "JPEG tokenizer coefficient, libjpeg-coefficient, byte-window, "
            "whole-image byte, middle-ground JPEG, and whole-image symbol SWA runs on Imagenette token stores, "
            "including SWA head-to-head, longer-run, larger-model comparisons, and sequence-level representation evals."
        ),
    )
