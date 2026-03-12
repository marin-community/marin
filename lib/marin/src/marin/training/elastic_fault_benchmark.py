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
from typing import Any, Literal

import draccus
import fsspec
from iris.cluster.client import get_job_info
from iris.marin_fs import REGION_TO_DATA_BUCKET
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.elastic import DiLoCoSyncConfig, ElasticTrainingConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig

from fray.v2 import ResourceConfig
from marin.training.training import (
    MARIN_FAULT_INJECTION_BY_WORKER_ENV,
    TrainLmOnPodConfig,
    run_levanter_train_lm,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ElasticFaultBenchmarkConfig:
    """Launch a small GPT-2 benchmark for resilient TPU training."""

    base_config_path: str = "lib/levanter/config/gpt2_small.yaml"
    tpu_type: str = "v5p-8"
    baseline_tpu_type: str = "v5p-16"
    mode: Literal["both", "baseline", "elastic"] = "both"
    region: str = "us-east5"
    num_steps: int = 2000
    seq_len: int = 1024
    elastic_local_batch_size: int = 128
    baseline_global_batch_size: int = 256
    benchmark_id: str | None = None
    output_root: str | None = None
    train_docs: int = 8192
    validation_docs: int = 256
    checkpoint_every: int = 100
    publish_every: int = 100
    sync_every: int = 100
    steps_per_eval: int = 500
    max_eval_batches: int = 1
    outer_learning_rate: float = 0.25
    outer_optimizer: Literal["adam", "sgd"] = "sgd"
    fault_steps_w001: tuple[int, ...] = (400, 1200)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(plain_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(plain_path, "w") as f:
        json.dump(payload, f, sort_keys=True, indent=2)


def _dataset_paths(root: str) -> tuple[str, str]:
    return (
        f"{root.rstrip('/')}/train.jsonl",
        f"{root.rstrip('/')}/validation.jsonl",
    )


def _synthetic_document(index: int) -> str:
    topics = (
        "distributed systems",
        "compilers",
        "optimization",
        "datasets",
        "tokenization",
        "numerical stability",
        "throughput",
        "fault tolerance",
    )
    adjectives = (
        "careful",
        "repeatable",
        "deterministic",
        "measured",
        "robust",
        "incremental",
        "elastic",
        "practical",
    )
    topic = topics[index % len(topics)]
    adjective = adjectives[(index * 3) % len(adjectives)]
    lines = [
        f"Benchmark document {index}. This passage is for TPU training experiments about {topic}.",
        f"The objective is {adjective} optimization under preemption while preserving useful learning dynamics.",
        (
            "Repeated corpora are acceptable for this benchmark because the primary metric is "
            "recovery behavior and useful throughput."
        ),
        (
            f"Shard {index % 128} contains notes on gradients, checkpointing, transfer servers, "
            "and peer synchronization."
        ),
        (
            f"Sequence {index} revisits the same concepts with slightly different wording so the "
            "tokenizer sees a broad but stable distribution."
        ),
    ]
    return " ".join(lines * 12)


def _ensure_synthetic_dataset(root: str, *, train_docs: int, validation_docs: int) -> tuple[str, str]:
    train_path, validation_path = _dataset_paths(root)
    if _exists(train_path) and _exists(validation_path):
        return train_path, validation_path

    for path, count in ((train_path, train_docs), (validation_path, validation_docs)):
        logger.info("Writing synthetic dataset shard %s with %d documents", path, count)
        fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
        parent = os.path.dirname(plain_path)
        if parent:
            fs.makedirs(parent, exist_ok=True)
        with fs.open(plain_path, "w") as f:
            for i in range(count):
                f.write(json.dumps({"text": _synthetic_document(i)}) + "\n")

    return train_path, validation_path


def _exists(path: str) -> bool:
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    return fs.exists(plain_path)


def _default_output_root(region: str, benchmark_id: str) -> str:
    bucket = REGION_TO_DATA_BUCKET[region]
    job_info = get_job_info()
    if job_info is not None and str(job_info.job_id).startswith("/"):
        user = str(job_info.job_id).split("/")[1]
    else:
        user = os.environ.get("USER", "unknown")
    return f"gs://{bucket}/scratch/{user}/resilient-tpu-training/{benchmark_id}"


def _base_train_config(path: str) -> TrainLmConfig:
    return draccus.load(TrainLmConfig, path)


def _base_validation_components(base_data: LmDataConfig) -> dict[str, DatasetComponent]:
    components: dict[str, DatasetComponent] = {}
    for name, component in base_data.components.items():
        if not isinstance(component, DatasetComponent):
            continue
        source = component.source
        if source is None:
            continue
        if isinstance(source, UrlDatasetSourceConfig) and len(source.validation_urls) == 0:
            continue
        components[name] = replace(component, cache_dir=None)
    return components


def _shared_data_config(base_data: LmDataConfig, root: str, *, train_docs: int, validation_docs: int) -> LmDataConfig:
    dataset_root = f"{root.rstrip('/')}/synthetic-data"
    train_path, validation_path = _ensure_synthetic_dataset(
        dataset_root,
        train_docs=train_docs,
        validation_docs=validation_docs,
    )

    validation_components = _base_validation_components(base_data)
    if not validation_components:
        validation_components["synthetic-validation"] = DatasetComponent(
            source=UrlDatasetSourceConfig(
                train_urls=[],
                validation_urls=[validation_path],
            )
        )

    return replace(
        base_data,
        cache_dir=f"{dataset_root}/cache",
        auto_build_caches=True,
        shuffle=True,
        components={
            "synthetic": DatasetComponent(
                source=UrlDatasetSourceConfig(
                    train_urls=[train_path],
                    validation_urls=[],
                )
            ),
            **validation_components,
        },
        train_weights={"synthetic": 1.0},
    )


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
            save_interval=timedelta(seconds=30),
            keep=[{"every": checkpoint_every}],
            append_run_id_to_base_path=False,
        ),
        elastic=elastic,
    )


def _pod_config(
    config: ElasticFaultBenchmarkConfig,
    *,
    base: TrainLmConfig,
    data: LmDataConfig,
    resources: ResourceConfig,
    run_id: str,
    run_name: str,
    output_path: str,
    train_batch_size: int,
    elastic: ElasticTrainingConfig,
    tags: list[str],
    env_vars: dict[str, str] | None = None,
) -> TrainLmOnPodConfig:
    train_config = replace(
        base,
        data=data,
        trainer=_trainer_config(
            base,
            run_id=run_id,
            run_name=run_name,
            num_steps=config.num_steps,
            train_batch_size=train_batch_size,
            checkpoint_every=config.checkpoint_every,
            steps_per_eval=config.steps_per_eval,
            max_eval_batches=config.max_eval_batches,
            elastic=elastic,
            tags=tags,
        ),
        train_seq_len=config.seq_len,
        hf_save_steps=config.num_steps + 1,
    )

    return TrainLmOnPodConfig(
        train_config=train_config,
        resources=resources,
        output_path=output_path,
        env_vars=env_vars,
        auto_build_caches=True,
    )


def run_elastic_fault_benchmark(config: ElasticFaultBenchmarkConfig) -> dict[str, Any]:
    benchmark_id = config.benchmark_id or f"elastic-125m-{uuid.uuid4().hex[:8]}"
    output_root = config.output_root or _default_output_root(config.region, benchmark_id)
    summary_path = f"{output_root.rstrip('/')}/summary.json"
    parent_job = get_job_info()

    logger.info("Benchmark root: %s", output_root)
    logger.info("Parent job: %s", parent_job.job_id if parent_job is not None else "local")

    base_config = _base_train_config(config.base_config_path)
    data_config = _shared_data_config(
        base_config.data,
        output_root,
        train_docs=config.train_docs,
        validation_docs=config.validation_docs,
    )

    baseline_run_id = f"{benchmark_id}-baseline"
    elastic_run_id = f"{benchmark_id}-elastic"
    baseline_output = f"{output_root.rstrip('/')}/baseline"
    elastic_output = f"{output_root.rstrip('/')}/elastic"

    summary: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "mode": config.mode,
        "output_root": output_root,
        "parent_job_id": str(parent_job.job_id) if parent_job is not None else None,
        "baseline": {
            "run_id": baseline_run_id,
            "wandb_url": f"https://wandb.ai/marin-community/marin/runs/{baseline_run_id}",
            "output_path": baseline_output,
        },
        "elastic": {
            "run_id": elastic_run_id,
            "group": elastic_run_id,
            "worker_run_ids": [f"{elastic_run_id}-w000", f"{elastic_run_id}-w001"],
            "worker_wandb_urls": [
                f"https://wandb.ai/marin-community/marin/runs/{elastic_run_id}-w000",
                f"https://wandb.ai/marin-community/marin/runs/{elastic_run_id}-w001",
            ],
            "fault_steps_by_worker": {"w001": list(config.fault_steps_w001)},
            "output_path": elastic_output,
        },
    }
    _write_json(summary_path, summary)

    if config.mode in ("both", "baseline"):
        logger.info("Starting baseline run %s", baseline_run_id)
        run_levanter_train_lm(
            _pod_config(
                config,
                base=base_config,
                data=data_config,
                resources=ResourceConfig.with_tpu(config.baseline_tpu_type, regions=[config.region]),
                run_id=baseline_run_id,
                run_name=f"{benchmark_id}-baseline",
                output_path=baseline_output,
                train_batch_size=config.baseline_global_batch_size,
                elastic=ElasticTrainingConfig(enabled=False),
                tags=["resilient-tpu", "baseline", config.tpu_type, config.region],
                env_vars={"JAX_TRACEBACK_FILTERING": "off"},
            )
        )
        summary["baseline"]["status"] = "succeeded"
        _write_json(summary_path, summary)

    if config.mode in ("both", "elastic"):
        logger.info("Starting elastic run %s", elastic_run_id)
        run_levanter_train_lm(
            _pod_config(
                config,
                base=base_config,
                data=data_config,
                resources=ResourceConfig.with_tpu(config.tpu_type, slice_count=2, regions=[config.region]),
                run_id=elastic_run_id,
                run_name=f"{benchmark_id}-elastic",
                output_path=elastic_output,
                train_batch_size=config.elastic_local_batch_size,
                elastic=ElasticTrainingConfig(
                    enabled=True,
                    worker_count=2,
                    transport="jax_transfer",
                    sync_interval_steps=config.sync_every,
                    publish_interval_steps=config.publish_every,
                    sync=DiLoCoSyncConfig(
                        outer_learning_rate=config.outer_learning_rate,
                        outer_optimizer=config.outer_optimizer,
                    ),
                    transfer_timeout=timedelta(minutes=5),
                    request_poll_interval_seconds=0.1,
                ),
                tags=["resilient-tpu", "elastic", "fault-injected", config.tpu_type, config.region],
                env_vars={
                    "JAX_TRACEBACK_FILTERING": "off",
                    MARIN_FAULT_INJECTION_BY_WORKER_ENV: json.dumps({"w001": list(config.fault_steps_w001)}),
                },
            )
        )
        summary["elastic"]["status"] = "succeeded"
        _write_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def _parse_args() -> ElasticFaultBenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run a resilient TPU benchmark on Iris.")
    parser.add_argument("--base-config-path", default="lib/levanter/config/gpt2_small.yaml")
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--baseline-tpu-type", default="v5p-16")
    parser.add_argument("--mode", choices=("both", "baseline", "elastic"), default="both")
    parser.add_argument("--region", default="us-east5")
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--elastic-local-batch-size", type=int, default=128)
    parser.add_argument("--baseline-global-batch-size", type=int, default=256)
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--train-docs", type=int, default=8192)
    parser.add_argument("--validation-docs", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--publish-every", type=int, default=100)
    parser.add_argument("--sync-every", type=int, default=100)
    parser.add_argument("--steps-per-eval", type=int, default=500)
    parser.add_argument("--max-eval-batches", type=int, default=1)
    parser.add_argument("--outer-learning-rate", type=float, default=0.25)
    parser.add_argument("--outer-optimizer", choices=("adam", "sgd"), default="sgd")
    parser.add_argument("--fault-steps-w001", default="400,1200")
    args = parser.parse_args()

    return ElasticFaultBenchmarkConfig(
        base_config_path=args.base_config_path,
        tpu_type=args.tpu_type,
        baseline_tpu_type=args.baseline_tpu_type,
        mode=args.mode,
        region=args.region,
        num_steps=args.num_steps,
        seq_len=args.seq_len,
        elastic_local_batch_size=args.elastic_local_batch_size,
        baseline_global_batch_size=args.baseline_global_batch_size,
        benchmark_id=args.benchmark_id,
        output_root=args.output_root,
        train_docs=args.train_docs,
        validation_docs=args.validation_docs,
        checkpoint_every=args.checkpoint_every,
        publish_every=args.publish_every,
        sync_every=args.sync_every,
        steps_per_eval=args.steps_per_eval,
        max_eval_batches=args.max_eval_batches,
        outer_learning_rate=args.outer_learning_rate,
        outer_optimizer=args.outer_optimizer,
        fault_steps_w001=tuple(int(step.strip()) for step in args.fault_steps_w001.split(",") if step.strip()),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    run_elastic_fault_benchmark(_parse_args())


if __name__ == "__main__":
    main()
