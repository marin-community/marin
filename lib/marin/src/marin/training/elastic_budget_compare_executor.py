# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass
from typing import Literal

from fray.v2 import ResourceConfig

from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main
from marin.execution.remote import remote
from marin.training.elastic_budget_compare import (
    ElasticBudgetCompareConfig,
    _base_train_config,
    _cached_data_config,
    _default_output_root,
    run_elastic_budget_compare,
)


@dataclass(frozen=True)
class ElasticBudgetCompareExecutorConfig:
    """Launch the elastic budget comparison through the Executor framework."""

    base_config_path: str = "lib/levanter/config/llama3_small_4k.yaml"
    dataset_cache_dir: str = "gs://marin-us-central1/tokenized/subcache/fineweb-edu-10B-6fbcbb"
    tokenizer: str = "meta-llama/Meta-Llama-3.1-8B"
    region: str = "us-central1"
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
    outer_optimizer: Literal["adam", "sgd", "nesterov_sgd"] = "sgd"
    outer_momentum: float = 0.9
    max_peers: int | None = None
    max_peer_staleness_steps: int | None = None
    outer_max_update_norm: float | None = None


def _budget_compare_step(
    *,
    benchmark_id: str,
    output_root: str,
    mode: Literal["baseline", "elastic"],
    config: ElasticBudgetCompareExecutorConfig,
    data_config,
) -> ExecutorStep[ElasticBudgetCompareConfig]:
    return ExecutorStep(
        name=f"training/{benchmark_id}/{mode}-parent",
        fn=remote(
            run_elastic_budget_compare,
            resources=ResourceConfig.with_cpu(cpu=1, ram="4GiB", disk="10GiB"),
            name=f"{benchmark_id}-{mode}-launcher",
        ),
        config=ElasticBudgetCompareConfig(
            base_config_path=config.base_config_path,
            dataset_cache_dir=config.dataset_cache_dir,
            tokenizer=config.tokenizer,
            region=config.region,
            mode=mode,
            benchmark_id=benchmark_id,
            output_root=f"{output_root.rstrip('/')}/{mode}-parent",
            target_flops=config.target_flops,
            baseline_tpu_type=config.baseline_tpu_type,
            elastic_tpu_type=config.elastic_tpu_type,
            elastic_slice_count=config.elastic_slice_count,
            checkpoint_every=config.checkpoint_every,
            publish_every=config.publish_every,
            sync_every=config.sync_every,
            steps_per_eval=config.steps_per_eval,
            max_eval_batches=config.max_eval_batches,
            baseline_global_batch_size=config.baseline_global_batch_size,
            elastic_local_batch_size=config.elastic_local_batch_size,
            outer_learning_rate=config.outer_learning_rate,
            outer_optimizer=config.outer_optimizer,
            outer_momentum=config.outer_momentum,
            max_peers=config.max_peers,
            max_peer_staleness_steps=config.max_peer_staleness_steps,
            outer_max_update_norm=config.outer_max_update_norm,
            data_config=data_config,
        ),
    )


def run_elastic_budget_compare_executor(config: ElasticBudgetCompareExecutorConfig):
    benchmark_id = config.benchmark_id or f"elastic-1e19-exec-{uuid.uuid4().hex[:8]}"
    output_root = config.output_root or _default_output_root(benchmark_id)

    base_config = _base_train_config(config.base_config_path)
    data_config = _cached_data_config(
        base_config.data,
        dataset_cache_dir=config.dataset_cache_dir,
        tokenizer=config.tokenizer,
    )

    baseline_step = _budget_compare_step(
        benchmark_id=benchmark_id,
        output_root=output_root,
        mode="baseline",
        config=config,
        data_config=data_config,
    )
    elastic_step = _budget_compare_step(
        benchmark_id=benchmark_id,
        output_root=output_root,
        mode="elastic",
        config=config,
        data_config=data_config,
    )

    executor_main.__wrapped__(
        ExecutorMainConfig(),
        steps=[baseline_step, elastic_step],
        description=f"Elastic budget compare via executor for {benchmark_id}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch elastic budget compare through Executor.")
    parser.add_argument("--base-config-path", default=ElasticBudgetCompareExecutorConfig.base_config_path)
    parser.add_argument("--dataset-cache-dir", default=ElasticBudgetCompareExecutorConfig.dataset_cache_dir)
    parser.add_argument("--tokenizer", default=ElasticBudgetCompareExecutorConfig.tokenizer)
    parser.add_argument("--region", default=ElasticBudgetCompareExecutorConfig.region)
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--target-flops",
        type=float,
        default=ElasticBudgetCompareExecutorConfig.target_flops,
    )
    parser.add_argument(
        "--baseline-tpu-type",
        default=ElasticBudgetCompareExecutorConfig.baseline_tpu_type,
    )
    parser.add_argument(
        "--elastic-tpu-type",
        default=ElasticBudgetCompareExecutorConfig.elastic_tpu_type,
    )
    parser.add_argument(
        "--elastic-slice-count", type=int, default=ElasticBudgetCompareExecutorConfig.elastic_slice_count
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=ElasticBudgetCompareExecutorConfig.checkpoint_every,
    )
    parser.add_argument(
        "--publish-every",
        type=int,
        default=ElasticBudgetCompareExecutorConfig.publish_every,
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=ElasticBudgetCompareExecutorConfig.sync_every,
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=ElasticBudgetCompareExecutorConfig.steps_per_eval,
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=ElasticBudgetCompareExecutorConfig.max_eval_batches,
    )
    parser.add_argument(
        "--baseline-global-batch-size",
        type=int,
        default=ElasticBudgetCompareExecutorConfig.baseline_global_batch_size,
    )
    parser.add_argument(
        "--elastic-local-batch-size",
        type=int,
        default=ElasticBudgetCompareExecutorConfig.elastic_local_batch_size,
    )
    parser.add_argument(
        "--outer-learning-rate", type=float, default=ElasticBudgetCompareExecutorConfig.outer_learning_rate
    )
    parser.add_argument(
        "--outer-optimizer",
        choices=("adam", "sgd", "nesterov_sgd"),
        default=ElasticBudgetCompareExecutorConfig.outer_optimizer,
    )
    parser.add_argument("--outer-momentum", type=float, default=ElasticBudgetCompareExecutorConfig.outer_momentum)
    parser.add_argument("--max-peers", type=int, default=None)
    parser.add_argument("--max-peer-staleness-steps", type=int, default=None)
    parser.add_argument("--outer-max-update-norm", type=float, default=None)
    args = parser.parse_args()

    run_elastic_budget_compare_executor(
        ElasticBudgetCompareExecutorConfig(
            base_config_path=args.base_config_path,
            dataset_cache_dir=args.dataset_cache_dir,
            tokenizer=args.tokenizer,
            region=args.region,
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
            outer_momentum=args.outer_momentum,
            max_peers=args.max_peers,
            max_peer_staleness_steps=args.max_peer_staleness_steps,
            outer_max_update_norm=args.outer_max_update_norm,
        )
    )
