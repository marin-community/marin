# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a full-context Snowball SFT demo from pinned Hugging Face inputs.

The graph converts the base model, normalizes OpenThoughts Agent through the
canonical SFT source registry, builds its token store, and trains on one
8xH100 node. No prebuilt Marin checkpoint, tokenizer, token store, or Marin
bucket is required. All generated artifacts are written beneath the caller's
``MARIN_PREFIX``.

For example, with an S3 bucket available to every worker::

    MARIN_PREFIX=s3://my-bucket/snowball-demo uv run python -m \
      experiments.grug_sft.snowball_262k_h100 --version dev --run
"""

import dataclasses
from datetime import timedelta
from functools import partial

import click
import jmp
from fray.cluster import ANY_REGION, ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.mixture import StopStrategy
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import TokenizerBackend
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.datakit.sft_sources import all_sft_sources
from marin.execution.artifact import read_artifact, validate_version
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize.attributes import tokenize_attributes_step
from marin.processing.tokenize.store_builder import LevanterStoreData, build_levanter_store_step
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.train import (
    GrugRunConfig,
    GrugTrainerConfig,
    grug_trainer_mesh_config,
    run_grug,
)
from experiments.grug_sft.snowball_hf_import import snowball_hf_to_grug

HF_MODEL = "open-athena/snowball-67b-a2b-base-262k-qk175-skew8"
HF_REVISION = "058ecaf27b9e4f37219df221a51e7d490d58ec3d"
DATA_SOURCE = "openthoughts-agent-sft-100k"
CONVERSION_VERSION = "2026.09.22-native"
DEFAULT_RUN_ID = "snowball-67b-262k-h100-demo"
WANDB_PROJECT = "snowball_sft_demo"

CONTEXT_LENGTH = 262_144
CONTEXT_SHARDS = 8
BATCH_SIZE = 1
DEFAULT_STEPS = 10
DEFAULT_SAMPLE_COUNT = 256
TENSORSTORE_CACHE_BYTES = 2 * 1024**3
TOKENIZE_WORKERS = 32
STORE_WORKERS = 32


def model_config() -> GrugModelConfig:
    """Current GPU trainer architecture corresponding to the pinned HF model."""
    return GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=2_560,
        intermediate_dim=1_280,
        shared_expert_intermediate_dim=2_560,
        num_shared_experts=1,
        num_experts=256,
        num_experts_per_token=4,
        num_layers=26,
        num_heads=20,
        num_kv_heads=5,
        head_dim=128,
        max_seq_len=CONTEXT_LENGTH,
        sliding_window=2_048,
        global_every=4,
        layer_norm_eps=1e-5,
        initializer_std=0.009882117688026186,
        qk_mult=1.75,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="sonic",
        remat_mode="recompute_all",
    )


def optimizer_config(steps: int) -> GrugMoeMuonHConfig:
    return GrugMoeMuonHConfig(
        learning_rate=5e-5,
        adam_lr=5e-5,
        beta1=0.9062,
        beta2=0.95,
        epsilon=3.8339433005718795e-15,
        weight_decay=0.0,
        max_grad_norm=None,
        min_lr_ratio=0.1,
        warmup=0,
        decay=max(1, steps // 10),
        lr_schedule="linear",
    )


def run_config(
    *,
    run_id: str,
    steps: int,
    data: LmDataConfig,
    output_path: str,
    base_checkpoint: str,
    resources: ResourceConfig,
) -> GrugRunConfig:
    if not run_id.strip():
        raise ValueError("Run ID must not be empty")
    if steps <= 0:
        raise ValueError("Steps must be positive")

    permanent_checkpoints = prefix_join(output_path, "checkpoints")
    temporary_checkpoints = prefix_join(output_path, "temporary-checkpoints")
    trainer = TrainerConfig(
        id=run_id,
        seed=0,
        train_batch_size=BATCH_SIZE,
        num_train_steps=steps,
        profiler=ProfilerConfig(enabled=False),
        mp=jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project=WANDB_PROJECT,
            name=run_id,
            id=run_id,
            mode="disabled",
            resume="allow",
            tags=["sft", "snowball", "262k", "h100x8", "openthoughts-agent", f"hf-{HF_REVISION[:12]}"],
        ),
        watch=WatchConfig(interval=0),
        progress_watchdog=ProgressWatchdogConfig(
            startup_timeout=timedelta(hours=2),
            step_timeout=timedelta(hours=1),
            process_timeout=timedelta(hours=2),
        ),
        use_explicit_mesh_axes=True,
        mesh=grug_trainer_mesh_config(CONTEXT_SHARDS),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        initialize_from=base_checkpoint,
        load_checkpoint_path=[permanent_checkpoints, temporary_checkpoints],
        checkpointer=CheckpointerConfig(
            base_path=permanent_checkpoints,
            temporary_base_path=temporary_checkpoints,
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=30),
            keep=None,
            delete_old_temp_checkpoints=True,
            keep_last_temporary_checkpoints=1,
        ),
    )
    return GrugRunConfig(
        model=model_config(),
        data=data,
        resources=resources,
        tensorstore_cache_bytes=TENSORSTORE_CACHE_BYTES,
        optimizer=optimizer_config(steps),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            log_every=1,
            ema_beta=None,
            z_loss_weight=1e-4,
            offload_opt_state=True,
            save_checkpoints=True,
            expert_axis_size=1,
            replica_axis_size=1,
            context_axis_size=CONTEXT_SHARDS,
        ),
        eval=None,
        processes_per_task=8,
        max_retries_failure=3,
        max_task_failures=3,
    )


def _training_data(store_path: str, tokenizer: str) -> LmDataConfig:
    store = read_artifact(store_path, LevanterStoreData)
    train = store.splits.get("train")
    if train is None or train.total_tokens <= 0:
        raise ValueError(f"{DATA_SOURCE} produced no training tokens")
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={
            DATA_SOURCE: DatasetComponent(
                source=None,
                cache_dir=store.cache_path,
                format=TextLmDatasetFormat(),
                pack=CONTEXT_LENGTH,
            )
        },
        train_weights={DATA_SOURCE: 1.0},
        auto_build_caches=False,
        shuffle=True,
        block_cross_document_attention=True,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )


def _run_training(
    output_path: str,
    *,
    run_id: str,
    steps: int,
    store_path: str,
    tokenizer: str,
    base_checkpoint: str,
    resources: ResourceConfig,
) -> None:
    run_grug(
        run_config(
            run_id=run_id,
            steps=steps,
            data=_training_data(store_path, tokenizer),
            output_path=output_path,
            base_checkpoint=base_checkpoint,
            resources=resources,
        )
    )


def build_demo(
    *,
    run_id: str = DEFAULT_RUN_ID,
    steps: int = DEFAULT_STEPS,
    version: str,
    sample_count: int | None = DEFAULT_SAMPLE_COUNT,
) -> StepSpec:
    """Build the registered Datakit source, HF conversion, and SFT graph."""
    if steps <= 0:
        raise ValueError("Steps must be positive")
    if sample_count is not None and sample_count <= 0:
        raise ValueError("Sample count must be positive")
    validate_version(version)

    conversion = snowball_hf_to_grug(
        HF_MODEL,
        hf_revision=HF_REVISION,
        model=model_config(),
        version=CONVERSION_VERSION,
        resources=ResourceConfig.with_cpu(cpu=64, ram="768g", disk="384g"),
    )
    conversion_step = conversion.step.lower()
    tokenizer = conversion.step.path()
    source = all_sft_sources()[DATA_SOURCE]
    tokenized = tokenize_attributes_step(
        name=f"datakit/tokenize/sft-demo/{DATA_SOURCE}",
        train_normalize=source.normalized,
        tokenizer=tokenizer,
        tokenizer_backend=TokenizerBackend.HF,
        sample_count=sample_count,
        max_workers=TOKENIZE_WORKERS,
        worker_resources=ResourceConfig(cpu=2, ram="32g", disk="10g"),
    )
    tokenized = dataclasses.replace(tokenized, deps=[*tokenized.deps, conversion_step])
    store = build_levanter_store_step(
        name=f"datakit/store/sft-demo/{DATA_SOURCE}",
        tokenize_steps=[tokenized],
        max_workers=STORE_WORKERS,
        worker_resources=ResourceConfig(cpu=2, ram="32g", disk="10g"),
    )
    resources = ResourceConfig.with_gpu(
        "H100",
        count=8,
        cpu=64,
        ram="768g",
        disk="384g",
        preemptible=False,
        regions=[ANY_REGION],
    )
    return StepSpec(
        name=f"grug-sft/{run_id}",
        deps=[store, conversion_step],
        fn=partial(
            _run_training,
            run_id=run_id,
            steps=steps,
            store_path=store.output_path,
            tokenizer=tokenizer,
            base_checkpoint=prefix_join(tokenizer, "checkpoints"),
            resources=resources,
        ),
        hash_attrs={
            "version": version,
            "steps": steps,
            "sample_count": sample_count,
            "model": f"{HF_MODEL}@{HF_REVISION}",
            "source": DATA_SOURCE,
        },
        override_output_path=f"grug-sft/{run_id}/{version}",
    )


@click.command()
@click.option("--run-id", default=DEFAULT_RUN_ID, show_default=True)
@click.option("--steps", type=click.IntRange(min=1), default=DEFAULT_STEPS, show_default=True)
@click.option("--version", required=True)
@click.option("--sample-count", type=click.IntRange(min=1), default=DEFAULT_SAMPLE_COUNT, show_default=True)
@click.option("--run", "do_run", is_flag=True)
@click.option("--max-concurrent", type=click.IntRange(min=1), default=8, show_default=True)
def main(run_id: str, steps: int, version: str, sample_count: int, do_run: bool, max_concurrent: int) -> None:
    step = build_demo(run_id=run_id, steps=steps, version=version, sample_count=sample_count)
    if do_run:
        StepRunner().run([step], max_concurrent=max_concurrent)
    else:
        click.echo(step)


if __name__ == "__main__":
    main()
