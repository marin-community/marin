# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from dataclasses import dataclass, replace
from datetime import timedelta
from math import ceil
from typing import Any, Literal

import draccus
import fsspec
from iris.cluster.client import get_job_info
from iris.marin_fs import marin_temp_bucket
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.elastic import DiLoCoSyncConfig, ElasticTrainingConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig

from fray.v2 import ResourceConfig
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.training.validation_sets import config_with_default_validation_sets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElasticBudgetCompareConfig:
    """Launch a baseline and elastic pretraining comparison at a fixed FLOP budget."""

    base_config_path: str = "lib/levanter/config/llama3_small_4k.yaml"
    dataset_cache_dir: str = "gs://marin-us-central1/tokenized/subcache/fineweb-edu-10B-6fbcbb"
    tokenizer: str = "meta-llama/Meta-Llama-3.1-8B"
    region: str = "us-central1"
    mode: Literal["both", "baseline", "elastic"] = "both"
    benchmark_id: str | None = None
    output_root: str | None = None
    target_flops: float = 1e19
    baseline_tpu_type: str = "v5p-32"
    elastic_tpu_type: str = "v5p-8"
    elastic_slice_count: int = 4
    checkpoint_every: int = 500
    publish_every: int = 200
    sync_every: int = 200
    steps_per_eval: int = 500
    max_eval_batches: int = 1
    baseline_global_batch_size: int = 128
    elastic_local_batch_size: int = 32
    outer_learning_rate: float = 0.25
    outer_optimizer: Literal["adam", "sgd"] = "sgd"
    max_peers: int | None = None
    max_peer_staleness_steps: int | None = None
    outer_max_update_norm: float | None = None
    data_config: LmDataConfig | None = None
    """Optional pre-resolved data config, useful when dispatching via Executor."""


def _write_json(path: str, payload: dict[str, Any]) -> None:
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(plain_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(plain_path, "w") as f:
        json.dump(payload, f, sort_keys=True, indent=2)


def _default_output_root(benchmark_id: str) -> str:
    job_info = get_job_info()
    if job_info is not None and str(job_info.job_id).startswith("/"):
        user = str(job_info.job_id).split("/")[1]
    else:
        user = os.environ.get("USER", "unknown")
    return marin_temp_bucket(ttl_days=30, prefix=f"{user}/resilient-tpu-training/{benchmark_id}")


def _base_train_config(path: str) -> TrainLmConfig:
    return draccus.load(TrainLmConfig, path)


def _experiment_model_config() -> LlamaConfig:
    return LlamaConfig(
        max_seq_len=4096,
        hidden_dim=768,
        intermediate_dim=2688,
        num_heads=12,
        num_kv_heads=12,
        num_layers=12,
        gradient_checkpointing=True,
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        reference_checkpoint="meta-llama/Meta-Llama-3.1-8B",
    )


def _experiment_optimizer_config() -> AdamConfig:
    return AdamConfig(
        learning_rate=0.008,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        warmup=2000,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-10,
        max_grad_norm=1.0,
        nesterov=False,
    )


def _cached_data_config(base_data: LmDataConfig, *, dataset_cache_dir: str, tokenizer: str) -> LmDataConfig:
    train_only_data = replace(
        base_data,
        tokenizer=tokenizer,
        cache_dir=None,
        auto_build_caches=False,
        components={
            "fineweb-edu-10b": DatasetComponent(
                source=UrlDatasetSourceConfig(train_urls=[], validation_urls=[]),
                cache_dir=dataset_cache_dir,
            ),
        },
        train_weights={"fineweb-edu-10b": 1.0},
        max_train_batches=None,
    )
    return config_with_default_validation_sets(train_only_data)


def _num_train_steps_for_target_flops(
    *,
    model_config: LlamaConfig,
    vocab_size: int,
    train_batch_size: int,
    train_seq_len: int,
    target_flops: float,
) -> int:
    flops_per_token = model_config.flops_per_token(vocab_size, train_seq_len)
    if flops_per_token is None:
        raise ValueError("Model config must expose flops_per_token for budgeted runs")
    total_tokens = target_flops / (3.0 * flops_per_token)
    tokens_per_step = train_batch_size * train_seq_len
    return ceil(total_tokens / tokens_per_step)


def _trainer_config(
    base: TrainLmConfig,
    *,
    run_id: str,
    run_name: str,
    num_steps: int,
    train_batch_size: int,
    checkpoint_every: int,
    steps_per_eval: int,
    max_eval_batches: int,
    elastic: ElasticTrainingConfig,
    tags: list[str],
) -> Any:
    base_trainer = base.trainer
    return replace(
        base_trainer,
        id=run_id,
        train_batch_size=train_batch_size,
        num_train_steps=num_steps,
        steps_per_eval=min(num_steps, max(1, steps_per_eval)),
        max_eval_batches=max_eval_batches,
        log_jaxprs=False,
        log_xla_hlo=False,
        tracker=WandbConfig(
            entity="marin-community",
            project="marin",
            name=run_name,
            tags=tags,
        ),
        checkpointer=CheckpointerConfig(
            base_path=base_trainer.checkpointer.base_path,
            save_interval=timedelta(minutes=5),
            keep=[{"every": checkpoint_every}],
            append_run_id_to_base_path=False,
        ),
        elastic=elastic,
    )


def _pod_config(
    *,
    base: TrainLmConfig,
    data: LmDataConfig,
    model: LlamaConfig,
    optimizer: AdamConfig,
    resources: ResourceConfig,
    run_id: str,
    run_name: str,
    output_path: str,
    train_batch_size: int,
    num_steps: int,
    checkpoint_every: int,
    steps_per_eval: int,
    max_eval_batches: int,
    elastic: ElasticTrainingConfig,
    tags: list[str],
) -> TrainLmOnPodConfig:
    train_config = replace(
        base,
        data=data,
        model=model,
        optimizer=optimizer,
        trainer=_trainer_config(
            base,
            run_id=run_id,
            run_name=run_name,
            num_steps=num_steps,
            train_batch_size=train_batch_size,
            checkpoint_every=checkpoint_every,
            steps_per_eval=steps_per_eval,
            max_eval_batches=max_eval_batches,
            elastic=elastic,
            tags=tags,
        ),
        train_seq_len=model.max_seq_len,
        hf_save_steps=num_steps + 1,
    )

    return TrainLmOnPodConfig(
        train_config=train_config,
        resources=resources,
        output_path=output_path,
        env_vars={"JAX_TRACEBACK_FILTERING": "off"},
        auto_build_caches=False,
    )


def run_elastic_budget_compare(config: ElasticBudgetCompareConfig) -> dict[str, Any]:
    benchmark_id = config.benchmark_id or f"elastic-1e19-{uuid.uuid4().hex[:8]}"
    output_root = config.output_root or _default_output_root(benchmark_id)
    summary_path = f"{output_root.rstrip('/')}/summary.json"
    parent_job = get_job_info()

    base_config = _base_train_config(config.base_config_path)
    model_config = _experiment_model_config()
    optimizer_config = _experiment_optimizer_config()
    if config.data_config is not None:
        data_config = config.data_config
    else:
        data_config = _cached_data_config(
            base_config.data,
            dataset_cache_dir=config.dataset_cache_dir,
            tokenizer=config.tokenizer,
        )

    baseline_steps = _num_train_steps_for_target_flops(
        model_config=model_config,
        vocab_size=128_256,
        train_batch_size=config.baseline_global_batch_size,
        train_seq_len=model_config.max_seq_len,
        target_flops=config.target_flops,
    )
    elastic_max_global_batch = config.elastic_local_batch_size * config.elastic_slice_count
    elastic_steps = _num_train_steps_for_target_flops(
        model_config=model_config,
        vocab_size=128_256,
        train_batch_size=elastic_max_global_batch,
        train_seq_len=model_config.max_seq_len,
        target_flops=config.target_flops,
    )

    baseline_run_id = f"{benchmark_id}-baseline"
    elastic_run_id = f"{benchmark_id}-elastic"
    baseline_output = f"{output_root.rstrip('/')}/baseline"
    elastic_output = f"{output_root.rstrip('/')}/elastic"
    max_peers = config.max_peers if config.max_peers is not None else max(1, config.elastic_slice_count - 1)

    summary: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "mode": config.mode,
        "region": config.region,
        "output_root": output_root,
        "parent_job_id": str(parent_job.job_id) if parent_job is not None else None,
        "dataset_cache_dir": config.dataset_cache_dir,
        "tokenizer": config.tokenizer,
        "target_flops": config.target_flops,
        "model": {
            "hidden_dim": model_config.hidden_dim,
            "intermediate_dim": model_config.intermediate_dim,
            "num_layers": model_config.num_layers,
            "num_heads": model_config.num_heads,
            "num_kv_heads": model_config.num_kv_heads,
            "train_seq_len": model_config.max_seq_len,
            "vocab_size": 128_256,
        },
        "baseline": {
            "run_id": baseline_run_id,
            "wandb_url": f"https://wandb.ai/marin-community/marin/runs/{baseline_run_id}",
            "output_path": baseline_output,
            "train_batch_size": config.baseline_global_batch_size,
            "num_steps": baseline_steps,
            "resources": {"tpu_type": config.baseline_tpu_type},
        },
        "elastic": {
            "run_id": elastic_run_id,
            "group": elastic_run_id,
            "worker_run_ids": [f"{elastic_run_id}-w{i:03d}" for i in range(config.elastic_slice_count)],
            "worker_wandb_urls": [
                f"https://wandb.ai/marin-community/marin/runs/{elastic_run_id}-w{i:03d}"
                for i in range(config.elastic_slice_count)
            ],
            "output_path": elastic_output,
            "local_batch_size": config.elastic_local_batch_size,
            "max_global_batch_size": elastic_max_global_batch,
            "num_steps": elastic_steps,
            "resources": {
                "tpu_type": config.elastic_tpu_type,
                "slice_count": config.elastic_slice_count,
            },
            "sync_every": config.sync_every,
            "publish_every": config.publish_every,
            "steps_per_eval": config.steps_per_eval,
            "max_eval_batches": config.max_eval_batches,
            "outer_learning_rate": config.outer_learning_rate,
            "outer_optimizer": config.outer_optimizer,
            "max_peers": max_peers,
            "max_peer_staleness_steps": config.max_peer_staleness_steps,
            "outer_max_update_norm": config.outer_max_update_norm,
        },
    }
    _write_json(summary_path, summary)

    logger.info("Budget comparison root: %s", output_root)
    logger.info("Summary path: %s", summary_path)

    if config.mode in ("both", "baseline"):
        logger.info("Starting baseline run %s", baseline_run_id)
        run_levanter_train_lm(
            _pod_config(
                base=base_config,
                data=data_config,
                model=model_config,
                optimizer=optimizer_config,
                resources=ResourceConfig.with_tpu(config.baseline_tpu_type, regions=[config.region]),
                run_id=baseline_run_id,
                run_name=f"{benchmark_id}-baseline",
                output_path=baseline_output,
                train_batch_size=config.baseline_global_batch_size,
                num_steps=baseline_steps,
                checkpoint_every=config.checkpoint_every,
                steps_per_eval=config.steps_per_eval,
                max_eval_batches=config.max_eval_batches,
                elastic=ElasticTrainingConfig(enabled=False),
                tags=["resilient-tpu", "budget-1e19", "baseline", config.baseline_tpu_type, config.region],
            )
        )
        summary["baseline"]["status"] = "succeeded"
        _write_json(summary_path, summary)

    if config.mode in ("both", "elastic"):
        logger.info("Starting elastic run %s", elastic_run_id)
        run_levanter_train_lm(
            _pod_config(
                base=base_config,
                data=data_config,
                model=model_config,
                optimizer=optimizer_config,
                resources=ResourceConfig.with_tpu(
                    config.elastic_tpu_type,
                    slice_count=config.elastic_slice_count,
                    regions=[config.region],
                ),
                run_id=elastic_run_id,
                run_name=f"{benchmark_id}-elastic",
                output_path=elastic_output,
                train_batch_size=config.elastic_local_batch_size,
                num_steps=elastic_steps,
                checkpoint_every=config.checkpoint_every,
                steps_per_eval=config.steps_per_eval,
                max_eval_batches=config.max_eval_batches,
                elastic=ElasticTrainingConfig(
                    enabled=True,
                    worker_count=config.elastic_slice_count,
                    transport="jax_transfer",
                    sync_interval_steps=config.sync_every,
                    publish_interval_steps=config.publish_every,
                    sync=DiLoCoSyncConfig(
                        outer_learning_rate=config.outer_learning_rate,
                        outer_optimizer=config.outer_optimizer,
                        outer_max_update_norm=config.outer_max_update_norm,
                    ),
                    max_peers=max_peers,
                    max_peer_staleness_steps=config.max_peer_staleness_steps,
                    transfer_timeout=timedelta(minutes=10),
                    request_poll_interval_seconds=0.1,
                ),
                tags=["resilient-tpu", "budget-1e19", "elastic", config.elastic_tpu_type, config.region],
            )
        )
        summary["elastic"]["status"] = "succeeded"
        _write_json(summary_path, summary)

    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _parse_args() -> ElasticBudgetCompareConfig:
    parser = argparse.ArgumentParser(description="Run a 1e19 FLOP elastic-vs-baseline comparison on Iris.")
    parser.add_argument("--base-config-path", default="lib/levanter/config/llama3_small_4k.yaml")
    parser.add_argument(
        "--dataset-cache-dir", default="gs://marin-us-central1/tokenized/subcache/fineweb-edu-10B-6fbcbb"
    )
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--mode", choices=("both", "baseline", "elastic"), default="both")
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--target-flops", type=float, default=1e19)
    parser.add_argument("--baseline-tpu-type", default="v5p-32")
    parser.add_argument("--elastic-tpu-type", default="v5p-8")
    parser.add_argument("--elastic-slice-count", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--publish-every", type=int, default=200)
    parser.add_argument("--sync-every", type=int, default=200)
    parser.add_argument("--steps-per-eval", type=int, default=500)
    parser.add_argument("--max-eval-batches", type=int, default=1)
    parser.add_argument("--baseline-global-batch-size", type=int, default=128)
    parser.add_argument("--elastic-local-batch-size", type=int, default=32)
    parser.add_argument("--outer-learning-rate", type=float, default=0.25)
    parser.add_argument("--outer-optimizer", choices=("adam", "sgd"), default="sgd")
    parser.add_argument("--max-peers", type=int, default=None)
    parser.add_argument("--max-peer-staleness-steps", type=int, default=None)
    parser.add_argument("--outer-max-update-norm", type=float, default=None)
    args = parser.parse_args()

    return ElasticBudgetCompareConfig(
        base_config_path=args.base_config_path,
        dataset_cache_dir=args.dataset_cache_dir,
        tokenizer=args.tokenizer,
        region=args.region,
        mode=args.mode,
        benchmark_id=args.benchmark_id,
        output_root=args.output_root,
        target_flops=args.target_flops,
        baseline_tpu_type=args.baseline_tpu_type,
        elastic_tpu_type=args.elastic_tpu_type,
        elastic_slice_count=args.elastic_slice_count,
        checkpoint_every=args.checkpoint_every,
        publish_every=args.publish_every,
        sync_every=args.sync_every,
        steps_per_eval=args.steps_per_eval,
        max_eval_batches=args.max_eval_batches,
        baseline_global_batch_size=args.baseline_global_batch_size,
        elastic_local_batch_size=args.elastic_local_batch_size,
        outer_learning_rate=args.outer_learning_rate,
        outer_optimizer=args.outer_optimizer,
        max_peers=args.max_peers,
        max_peer_staleness_steps=args.max_peer_staleness_steps,
        outer_max_update_norm=args.outer_max_update_norm,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    run_elastic_budget_compare(_parse_args())


if __name__ == "__main__":
    main()
