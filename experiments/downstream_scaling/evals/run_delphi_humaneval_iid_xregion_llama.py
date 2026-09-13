# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Llama-only IID null baseline for the Delphi/Llama HumanEval sweep.

With constant Delphi logits and Llama advisor weight ``w > 0``, sampling at
the joint sweep's temperature 0.4 is equivalent to sampling Llama at
temperature ``0.4 / w``. This experiment evaluates every positive weight from
the joint sweep both with its Llama-side top-16 truncation and without top-k
truncation. The undefined ``w = 0`` endpoint is omitted.

This is the clean zero-signal null whose candidates come from Llama. It does
not reproduce the arbitrary tied top-k candidates that a literal zero-logit
Delphi engine would add to ``bytes_union``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from fray.cluster import ANY_REGION, ResourceConfig, get_tpu_topology
from rigging.filesystem.cluster_config import data_config
from rigging.log_setup import configure_logging
from thalas.execution.executor import ExecutorStep, executor_main, output_path_of

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
from experiments.downstream_scaling.evals.framework.xregion.pool import WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.humaneval import HumanEvalTask, HumanEvalTaskConfig
from experiments.models import ModelConfig, download_model_step

logger = logging.getLogger(__name__)

N_SAMPLES = 64
N_PROBLEMS: int | None = None
WORKERS_PER_TPU_TYPE = 4
AGGREGATE_WORKERS = 32
CHUNK_SIZE = 512
TPU_TYPES: tuple[str, ...] = ("v5p-8", "v5litepod-4", "v5litepod-8", "v6e-4", "v6e-8")

HEARTBEAT_TIMEOUT = 120.0
POLL_BACKOFF = 10.0
MAX_MODEL_LEN = 8192
MAX_TOKENS = 1024
SEED = 42
STOP_TOKENS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")

NUM_FEWSHOT = 0
FEWSHOT_SEED = 1234

BASE_TEMPERATURE = 0.7  # 0.4
ADVISOR_WEIGHTS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TOP_K_REGIMES: tuple[int, ...] = (16, 0)

SAMPLING_CONFIGS = tuple(
    IIDSamplingConfig(
        temperature=BASE_TEMPERATURE / advisor_weight,
        top_p=1.0,
        top_k=top_k,
        max_tokens=MAX_TOKENS,
        stop=STOP_TOKENS,
    )
    for top_k in TOP_K_REGIMES
    for advisor_weight in ADVISOR_WEIGHTS
)

LLAMA_3_1_8B = download_model_step(ModelConfig(hf_repo_id="meta-llama/Llama-3.1-8B", hf_revision="d04e592"))

DEFAULT_PRESEED_REGIONS: tuple[str, ...] = (
    "europe-west4",
    "us-central1",
    "us-central2",
    "us-east1",
    "us-east5",
    "us-west4",
)
WORKER_REGIONS_ANY = "any"


def make_task() -> HumanEvalTask:
    return HumanEvalTask(
        config=HumanEvalTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )


def make_algorithm(*, worker_pools: tuple[IIDPoolConfig, ...]) -> IIDCompletionAlgorithm:
    return IIDCompletionAlgorithm(
        config=IIDConfig(
            n_samples=N_SAMPLES,
            seed=SEED,
            sampling_configs=SAMPLING_CONFIGS,
            execution=IIDExecutionConfig(
                worker_pools=worker_pools,
                chunk_size=CHUNK_SIZE,
                heartbeat_timeout=HEARTBEAT_TIMEOUT,
                poll_backoff=POLL_BACKOFF,
                aggregate_workers=AGGREGATE_WORKERS,
            ),
            model=IIDModelConfig(
                max_model_len=MAX_MODEL_LEN,
                enable_prefix_caching=False,
            ),
        )
    )


def build_run_steps(worker_pools: tuple[WorkerPoolConfig, ...]) -> list[ExecutorStep]:
    resolved_pools = tuple(
        IIDPoolConfig(pool=pool, placements=iid_tp1_placements(pool.chips_per_vm)) for pool in worker_pools
    )
    return [
        make_eval_step(
            name="downstream_scaling/evals/delphi/humaneval/iid_xregion/zero_delphi/llama_3_1_8b",
            model_path=output_path_of(LLAMA_3_1_8B),
            task=make_task(),
            alg=make_algorithm(worker_pools=resolved_pools),
        )
    ]


def _regional_download_step(base_step: ExecutorStep, *, name: str, region: str) -> ExecutorStep:
    if base_step.override_output_path is None:
        raise ValueError(f"Download step {base_step.name!r} does not define a stable relative output path")
    bucket = data_config().region_buckets[region]
    regional_step = replace(base_step, name=name)
    return regional_step.with_output_path(f"gs://{bucket.name}/{base_step.override_output_path}")


def build_preseed_steps(regions: list[str]) -> list[ExecutorStep]:
    _validate_regions(regions, name="preseed regions")
    prefix = "downstream_scaling/evals/delphi/humaneval/iid_xregion/zero_delphi/preseed"
    return [
        _regional_download_step(
            LLAMA_3_1_8B,
            name=f"{prefix}/llama-3.1-8b/{region}",
            region=region,
        )
        for region in regions
    ]


def _validate_regions(regions: list[str], *, name: str) -> None:
    unknown_regions = sorted(set(regions) - set(data_config().region_buckets))
    if unknown_regions:
        raise ValueError(f"Unknown {name} {unknown_regions}; known: {sorted(data_config().region_buckets)}")


def resolve_worker_regions(worker_regions: list[str] | None, preseed_regions: list[str]) -> list[str]:
    if worker_regions is None:
        _validate_regions(preseed_regions, name="preseed regions")
        return preseed_regions

    if worker_regions == [WORKER_REGIONS_ANY]:
        return [ANY_REGION]

    _validate_regions(worker_regions, name="worker regions")
    return worker_regions


def make_worker_pools(
    *,
    tpu_types: list[str],
    worker_regions: list[str],
    num_workers: int,
) -> tuple[WorkerPoolConfig, ...]:
    if num_workers <= 0:
        raise ValueError(f"--num-workers must be positive, got {num_workers}")
    if len(set(tpu_types)) != len(tpu_types):
        raise ValueError(f"duplicate TPU types: {tpu_types}")

    pools = []
    for tpu_type in tpu_types:
        topology = get_tpu_topology(tpu_type)
        if topology.vm_count != 1:
            raise ValueError(f"IID xregion supports only single-VM TPU types, got {tpu_type}")
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
    parser.add_argument("--num-workers", type=int, default=WORKERS_PER_TPU_TYPE, help="Workers per TPU type.")

    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "preseed":
        steps = build_preseed_steps(args.preseed_regions)
        description = "Preseed Llama-3.1-8B for the zero-Delphi-logits HumanEval IID sweep."
    else:
        worker_regions = resolve_worker_regions(args.worker_regions, args.preseed_regions)
        worker_pools = make_worker_pools(
            tpu_types=args.tpu_types,
            worker_regions=worker_regions,
            num_workers=args.num_workers,
        )
        steps = build_run_steps(worker_pools)
        description = "Llama-3.1-8B HumanEval IID temperature sweep for the zero-Delphi-logits null."

    executor_main(steps=steps, description=description)


if __name__ == "__main__":
    configure_logging()
    main()
