# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for IID cross-region Zephyr completion on a small GSM8K slice."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, replace

import fsspec
from fray.cluster import ANY_REGION, ResourceConfig, get_tpu_topology
from rigging.filesystem import data_config
from rigging.log_setup import configure_logging
from thalas.execution.executor import ExecutorStep, executor_main, output_path_of
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.algorithms.iid_xregion import (
    IIDCompletionAlgorithm,
    IIDConfig,
    IIDExecutionConfig,
    IIDModelConfig,
    IIDPoolConfig,
    IIDSamplingConfig,
    iid_tp1_placements,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import COMPLETIONS_FILENAME, read_completion_rows
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement, WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.evals.utils import version_path
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS

logger = logging.getLogger(__name__)

MODEL_KEY = "3e20"
N_PROBLEMS = 128
N_SAMPLES = 4
NUM_WORKERS = 2
CHUNK_SIZE = 32
TPU_TYPES: tuple[str, ...] = ("v4-8", "v5p-8", "v6e-4", "v5litepod-4", "v6e-8", "v5litepod-8")

PLACEMENT_OVERRIDES: dict[tuple[str, str], tuple[EnginePlacement, ...]] = {
    ("1e23", "v6e-8"): (EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2),),
}

TEMPERATURE = 0.6
TOP_P = 1.0
TOP_K = 1000
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")
HEARTBEAT_TIMEOUT = 120.0

SAMPLING_CONFIGS = (
    IIDSamplingConfig(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        stop=STOP_TOKENS,
    ),
    IIDSamplingConfig(
        temperature=0.0,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        stop=STOP_TOKENS,
    ),
)

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234

DEFAULT_PRESEED_REGIONS: tuple[str, ...] = ("us-central1",)
WORKER_REGIONS_ANY = "any"


@dataclass(frozen=True)
class ValidateCompletionsConfig:
    output_path: str
    completions_path: str
    expected_rows: int
    expected_samples_per_config: int
    expected_sampling_configs: tuple[IIDSamplingConfig, ...]


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
    worker_pools: tuple[IIDPoolConfig, ...],
    chunk_size: int,
    n_samples: int,
    heartbeat_timeout: float,
) -> IIDCompletionAlgorithm:
    return IIDCompletionAlgorithm(
        config=IIDConfig(
            n_samples=n_samples,
            seed=SEED,
            sampling_configs=SAMPLING_CONFIGS,
            execution=IIDExecutionConfig(
                worker_pools=worker_pools,
                chunk_size=chunk_size,
                heartbeat_timeout=heartbeat_timeout,
            ),
            model=IIDModelConfig(apply_rpa_block_size_patch=True),
        )
    )


def validate_completions(config: ValidateCompletionsConfig) -> None:
    rows = list(read_completion_rows(config.completions_path))
    if len(rows) != config.expected_rows:
        raise ValueError(f"Expected {config.expected_rows} completion rows, got {len(rows)}")

    expected_config_counts = Counter(
        {
            json.dumps(asdict(sampling_config), sort_keys=True): config.expected_samples_per_config
            for sampling_config in config.expected_sampling_configs
        }
    )
    expected_completions = config.expected_samples_per_config * len(config.expected_sampling_configs)

    seen_ids: set[str] = set()
    for row in rows:
        row_id = row["id"]
        if row_id in seen_ids:
            raise ValueError(f"Duplicate completion row id {row_id!r}")
        seen_ids.add(row_id)

        completions = row["completions"]
        if len(completions) != expected_completions:
            raise ValueError(f"Expected {expected_completions} completions for {row_id!r}, got {len(completions)}")
        actual_config_counts = Counter(
            json.dumps(completion["metadata"]["sampling_config"], sort_keys=True) for completion in completions
        )
        if actual_config_counts != expected_config_counts:
            raise ValueError(
                f"Unexpected sampling config counts for {row_id!r}: "
                f"expected {expected_config_counts}, got {actual_config_counts}"
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
    sampling_configs: tuple[IIDSamplingConfig, ...],
) -> ExecutorStep:
    return ExecutorStep(
        name=name,
        fn=remote(validate_completions, resources=ResourceConfig.with_cpu(cpu=1, ram="1g")),
        config=ValidateCompletionsConfig(
            output_path=this_output_path(),
            completions_path=version_path(completions_path),  # type: ignore[arg-type]
            expected_rows=versioned(n_problems),  # type: ignore[arg-type]
            expected_samples_per_config=versioned(n_samples),  # type: ignore[arg-type]
            expected_sampling_configs=versioned(sampling_configs),  # type: ignore[arg-type]
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
) -> list[ExecutorStep]:
    if model_key not in DELPHI_HF_DOWNLOADS:
        raise ValueError(f"Unknown Delphi model key {model_key!r}; known: {sorted(DELPHI_HF_DOWNLOADS)}")

    pool_slug = "_".join(pool.pool_id for pool in worker_pools)
    resolved_pools = tuple(
        IIDPoolConfig(
            pool=pool,
            placements=PLACEMENT_OVERRIDES.get(
                (model_key, pool.pool_id),
                iid_tp1_placements(pool.chips_per_vm),
            ),
        )
        for pool in worker_pools
    )
    completions = make_eval_step(
        name=(f"downstream_scaling/evals/smoke/iid_xregion_per_chip/{model_key}/tp=1/{pool_slug}"),
        model_path=output_path_of(DELPHI_HF_DOWNLOADS[model_key]),
        task=make_task(n_problems),
        alg=make_algorithm(
            worker_pools=resolved_pools,
            chunk_size=chunk_size,
            n_samples=n_samples,
            heartbeat_timeout=heartbeat_timeout,
        ),
        skip_grades=True,
    )
    validation = make_validation_step(
        name=(f"downstream_scaling/evals/smoke/iid_xregion_per_chip/{model_key}/tp=1/{pool_slug}/validate"),
        completions_path=output_path_of(completions) / COMPLETIONS_FILENAME,
        n_problems=n_problems,
        n_samples=n_samples,
        sampling_configs=SAMPLING_CONFIGS,
    )
    return [validation]


def build_preseed_steps(
    *,
    model_key: str,
    regions: list[str],
) -> list[ExecutorStep]:
    if model_key not in DELPHI_HF_DOWNLOADS:
        raise ValueError(f"Unknown Delphi model key {model_key!r}; known: {sorted(DELPHI_HF_DOWNLOADS)}")
    _validate_regions(regions, name="preseed regions")

    base_step = DELPHI_HF_DOWNLOADS[model_key]
    if base_step.override_output_path is None:
        raise ValueError(f"Delphi download step for {model_key!r} does not define a stable relative output path")

    steps = []
    for region in regions:
        bucket = data_config().region_buckets[region]
        regional_step = replace(
            base_step,
            name=f"downstream_scaling/evals/smoke/iid_xregion/preseed/{model_key}/{region}",
        )
        steps.append(regional_step.with_output_path(f"gs://{bucket.name}/{base_step.override_output_path}"))
    return steps


def _validate_regions(regions: list[str], *, name: str) -> None:
    unknown_regions = sorted(set(regions) - set(data_config().region_buckets))
    if unknown_regions:
        raise ValueError(f"Unknown {name} {unknown_regions}; known: {sorted(data_config().region_buckets)}")


def resolve_worker_regions(worker_regions: list[str] | None, preseed_regions: list[str]) -> list[str]:
    if worker_regions is None:
        _validate_regions(preseed_regions, name="preseed regions")
        return preseed_regions

    if worker_regions == [WORKER_REGIONS_ANY]:
        # Fray's explicit "run anywhere, do not inherit the parent job's
        # region" marker; regions=None (UNSET) would inherit the coordinator
        # job's region for these nested worker pools.
        return [ANY_REGION]

    _validate_regions(worker_regions, name="worker regions")
    return worker_regions


def make_worker_pools(
    *,
    tpu_types: list[str],
    worker_regions: list[str],
    num_workers: int,
) -> tuple[WorkerPoolConfig, ...]:
    # One single-type pool per TPU type: with_tpu sizes the pool's cpu/ram
    # request from its primary type's host, and iris enforces those values as
    # container limits, so mixing host families in one pool either starves
    # the smaller host or caps the larger one. The shared chunk ledger
    # load-balances across pools.
    if num_workers <= 0:
        raise ValueError(f"--num-workers must be positive, got {num_workers}")
    if len(set(tpu_types)) != len(tpu_types):
        raise ValueError(f"duplicate TPU types: {tpu_types}")

    pools = []
    for tpu_type in tpu_types:
        topology = get_tpu_topology(tpu_type)
        pools.append(
            WorkerPoolConfig(
                pool_id=tpu_type,
                num_workers=num_workers,
                worker_resources=ResourceConfig.with_tpu(tpu_type, regions=worker_regions),
                vm_count=topology.vm_count,
                chips_per_vm=topology.chips_per_vm,
            )
        )
    return tuple(pools)


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
        description = f"IID xregion smoke preseed for {args.model_key}."
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
        )
        description = "IID xregion smoke on a small GSM8K slice."

    executor_main(
        steps=steps,
        description=description,
    )


if __name__ == "__main__":
    configure_logging()
    main()
