# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for joint-decode-avg cross-region Zephyr completion on GSM8K."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, replace

import fsspec
from fray.cluster import ResourceConfig, get_tpu_topology
from marin.execution.executor import ExecutorStep, executor_main, output_path_of
from marin.execution.remote import remote
from marin.execution.types import this_output_path, versioned
from rigging.filesystem import data_config
from rigging.log_setup import configure_logging

from experiments.downstream_scaling.evals.algorithms.joint_decode_avg_xregion import (
    JointDecodeCompletionAlgorithm,
    JointDecodeConfig,
    JointDecodeExecutionConfig,
    JointDecodeModelConfig,
    JointDecodeSamplingConfig,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import COMPLETIONS_FILENAME, read_completion_rows
from experiments.downstream_scaling.evals.framework.xregion.pool import WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.evals.utils import version_path
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS

logger = logging.getLogger(__name__)

MODEL_KEY = "3e18"
N_PROBLEMS = 64
N_SAMPLES = 4
NUM_WORKERS = 2
CHUNK_SIZE = 16
TPU_TYPES: tuple[str, ...] = ("v4-8", "v5p-8", "v6e-4", "v5litepod-4", "v6e-8", "v5litepod-8")

BARRIER_TIMEOUT_S = 1200.0
HEARTBEAT_TIMEOUT = 120.0
POLL_BACKOFF = 10.0

MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234

TEMPERATURE = 0.4
TOP_K_A = 16
TOP_K_B = 16
ADVISOR_WEIGHT = 0.5

DEFAULT_PRESEED_REGIONS: tuple[str, ...] = ("us-central1",)
WORKER_REGIONS_ANY = "any"


@dataclass(frozen=True)
class ValidateCompletionsConfig:
    output_path: str
    completions_path: str
    expected_rows: int
    expected_completions_per_row: int


def make_task(n_problems: int) -> GSM8KTask:
    return GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=n_problems,
        )
    )


def make_algorithm(
    *,
    worker_pools: tuple[WorkerPoolConfig, ...],
    chunk_size: int,
    n_samples: int,
    heartbeat_timeout: float,
    poll_backoff: float,
    advisor_model_path,
) -> JointDecodeCompletionAlgorithm:
    return JointDecodeCompletionAlgorithm(
        config=JointDecodeConfig(
            sampling=JointDecodeSamplingConfig(
                n_samples=n_samples,
                max_tokens=MAX_TOKENS,
                top_k_a=TOP_K_A,
                top_k_b=TOP_K_B,
                seed=SEED,
                temperature=TEMPERATURE,
                advisor_weight=ADVISOR_WEIGHT,
                stop=STOP_TOKENS,
            ),
            advisor_model_path=advisor_model_path,
            decoder_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            advisor_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            execution=JointDecodeExecutionConfig(
                worker_pools=worker_pools,
                chunk_size=chunk_size,
                microbatch_size=None,
                heartbeat_timeout=heartbeat_timeout,
                poll_backoff=poll_backoff,
                barrier_timeout_s=BARRIER_TIMEOUT_S,
            ),
        )
    )


def validate_completions(config: ValidateCompletionsConfig) -> None:
    rows = list(read_completion_rows(config.completions_path))
    if len(rows) != config.expected_rows:
        raise ValueError(f"Expected {config.expected_rows} completion rows, got {len(rows)}")

    seen_ids: set[str] = set()
    for row in rows:
        row_id = row["id"]
        if row_id in seen_ids:
            raise ValueError(f"Duplicate completion row id {row_id!r}")
        seen_ids.add(row_id)

        completions = row["completions"]
        if len(completions) != config.expected_completions_per_row:
            raise ValueError(
                f"Expected {config.expected_completions_per_row} completions for {row_id!r}, got {len(completions)}"
            )

    path = os.path.join(config.output_path, "validation.SUCCESS")
    with fsspec.open(path, "wt") as f:
        f.write("ok\n")
    logger.info("Validated %d completion rows at %s", len(rows), config.completions_path)


def make_validation_step(
    *,
    name: str,
    completions_path,
    n_problems: int,
    n_samples: int,
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(validate_completions, resources=ResourceConfig.with_cpu(cpu=1, ram="1g")),
        config=ValidateCompletionsConfig(
            output_path=this_output_path(),
            completions_path=version_path(completions_path),  # type: ignore[arg-type]
            expected_rows=versioned(n_problems),  # type: ignore[arg-type]
            expected_completions_per_row=versioned(n_samples),  # type: ignore[arg-type]
        ),
    )


def build_run_steps(
    *,
    model_key: str,
    worker_pools: tuple[WorkerPoolConfig, ...],
    chunk_size: int,
    n_problems: int,
    n_samples: int,
    heartbeat_timeout: float,
    poll_backoff: float,
) -> list[ExecutorStep]:
    if model_key not in DELPHI_HF_DOWNLOADS:
        raise ValueError(f"Unknown Delphi model key {model_key!r}; known: {sorted(DELPHI_HF_DOWNLOADS)}")

    pool_slug = "_".join(pool.pool_id for pool in worker_pools)
    model_path = output_path_of(DELPHI_HF_DOWNLOADS[model_key])
    completions = make_eval_step(
        name=(
            f"downstream_scaling/evals/smoke/joint_decode_avg_xregion_per_pair/"
            f"advisor_weight{round(ADVISOR_WEIGHT * 100):03d}/{model_key}/{pool_slug}"
        ),
        model_path=model_path,
        task=make_task(n_problems),
        alg=make_algorithm(
            worker_pools=worker_pools,
            chunk_size=chunk_size,
            n_samples=n_samples,
            heartbeat_timeout=heartbeat_timeout,
            poll_backoff=poll_backoff,
            advisor_model_path=model_path,
        ),
        skip_grades=True,
    )
    validation = make_validation_step(
        name=(
            f"downstream_scaling/evals/smoke/joint_decode_avg_xregion_per_pair/"
            f"advisor_weight{round(ADVISOR_WEIGHT * 100):03d}/{model_key}/{pool_slug}/validate"
        ),
        completions_path=output_path_of(completions) / COMPLETIONS_FILENAME,
        n_problems=n_problems,
        n_samples=n_samples,
    )
    return [validation]


def _regional_download_step(base_step: ExecutorStep, *, name: str, region: str) -> ExecutorStep:
    if base_step.override_output_path is None:
        raise ValueError(f"Download step {base_step.name!r} does not define a stable relative output path")
    bucket = data_config().region_buckets[region]
    regional_step = replace(base_step, name=name)
    return regional_step.with_output_path(f"gs://{bucket}/{base_step.override_output_path}")


def build_preseed_steps(
    *,
    model_key: str,
    regions: list[str],
) -> list[ExecutorStep]:
    if model_key not in DELPHI_HF_DOWNLOADS:
        raise ValueError(f"Unknown Delphi model key {model_key!r}; known: {sorted(DELPHI_HF_DOWNLOADS)}")
    _validate_regions(regions, name="preseed regions")

    steps = []
    for region in regions:
        steps.append(
            _regional_download_step(
                DELPHI_HF_DOWNLOADS[model_key],
                name=f"downstream_scaling/evals/smoke/joint_decode_avg_xregion/preseed/{model_key}/{region}",
                region=region,
            )
        )
    return steps


def _validate_regions(regions: list[str], *, name: str) -> None:
    unknown_regions = sorted(set(regions) - set(data_config().region_buckets))
    if unknown_regions:
        raise ValueError(f"Unknown {name} {unknown_regions}; known: {sorted(data_config().region_buckets)}")


def resolve_worker_regions(worker_regions: list[str] | None, preseed_regions: list[str]) -> list[str] | None:
    if worker_regions is None:
        _validate_regions(preseed_regions, name="preseed regions")
        return preseed_regions

    if worker_regions == [WORKER_REGIONS_ANY]:
        return None

    _validate_regions(worker_regions, name="worker regions")
    return worker_regions


def tpu_pool_key(tpu_type: str) -> tuple[int, int]:
    topology = get_tpu_topology(tpu_type)
    return (topology.vm_count, topology.chips_per_vm)


def pool_id_for_key(key: tuple[int, int]) -> str:
    vm_count, chips_per_vm = key
    return f"{vm_count}vm-{chips_per_vm}chip"


def make_worker_pools(
    *,
    tpu_types: list[str],
    worker_regions: list[str] | None,
    num_workers: int,
) -> tuple[WorkerPoolConfig, ...]:
    if num_workers <= 0:
        raise ValueError(f"--num-workers must be positive, got {num_workers}")

    grouped: dict[tuple[int, int], list[str]] = {}
    for tpu_type in tpu_types:
        key = tpu_pool_key(tpu_type)
        if key[0] != 1:
            raise ValueError(f"joint decode xregion supports only single-VM TPU types, got {tpu_type} topology={key}")
        if key[1] % 2 != 0:
            raise ValueError(f"joint decode xregion needs even chips_per_vm, got {tpu_type} topology={key}")
        grouped.setdefault(key, []).append(tpu_type)

    return tuple(
        WorkerPoolConfig(
            pool_id=pool_id_for_key(key),
            num_workers=num_workers,
            worker_resources=ResourceConfig.with_tpu(pool_tpu_types, regions=worker_regions),
            vm_count=key[0],
            chips_per_vm=key[1],
        )
        for key, pool_tpu_types in sorted(grouped.items())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--mode", choices=("preseed", "run"), required=True)
    parser.add_argument("--model-key", default=MODEL_KEY)
    parser.add_argument("--preseed-regions", nargs="+", default=list(DEFAULT_PRESEED_REGIONS))
    parser.add_argument(
        "--worker-regions",
        nargs="+",
        default=None,
        help=(
            "TPU worker placement regions. Defaults to --preseed-regions. "
            f"Use '{WORKER_REGIONS_ANY}' for unrestricted placement."
        ),
    )
    parser.add_argument("--tpu-types", nargs="+", default=list(TPU_TYPES))
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--heartbeat-timeout", type=float, default=HEARTBEAT_TIMEOUT)
    parser.add_argument("--poll-backoff", type=float, default=POLL_BACKOFF)

    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "preseed":
        steps = build_preseed_steps(
            model_key=args.model_key,
            regions=args.preseed_regions,
        )
        description = f"Joint-decode-avg xregion smoke preseed for {args.model_key}."
    else:
        worker_regions = resolve_worker_regions(args.worker_regions, args.preseed_regions)
        worker_pools = make_worker_pools(
            tpu_types=args.tpu_types,
            worker_regions=worker_regions,
            num_workers=args.num_workers,
        )
        steps = build_run_steps(
            model_key=args.model_key,
            worker_pools=worker_pools,
            chunk_size=args.chunk_size,
            n_problems=args.n_problems,
            n_samples=args.n_samples,
            heartbeat_timeout=args.heartbeat_timeout,
            poll_backoff=args.poll_backoff,
        )
        description = "Joint-decode-avg xregion smoke on a small GSM8K slice."

    executor_main(
        steps=steps,
        description=description,
    )


if __name__ == "__main__":
    configure_logging()
    main()
