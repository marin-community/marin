# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare top-k 16 with no top-k truncation for unhinted Delphi 1e23 on GSM8K.

Both settings use temperature 0.4 and 64 samples per problem. Each worker
reuses its TP=2 vLLM engine across chunks and sampling configs.
"""

from __future__ import annotations

import argparse
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
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement, WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS

N_SAMPLES = 64
N_PROBLEMS = 256
WORKERS_PER_TPU_TYPE = 4
AGGREGATE_WORKERS = 32
CHUNK_SIZE = 512
TPU_TYPES: tuple[str, ...] = ("v5p-8", "v6e-4", "v6e-8")

# Multi-chip declarations libtpu accepts, per pool. v6e rejects every two-chip
# subset, so an engine there is handed a physical 2x2 quad and vLLM uses two of
# its chips. v5p-8 pairs need 1,2,1 bounds.
TP2_PLACEMENTS: dict[str, tuple[EnginePlacement, ...]] = {
    "v5p-8": (EnginePlacement((0, 1), (1, 2, 1), 2), EnginePlacement((2, 3), (1, 2, 1), 2)),
    "v6e-4": (EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2),),
    "v6e-8": (EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2), EnginePlacement((4, 5, 6, 7), (2, 2, 1), 2)),
}

HEARTBEAT_TIMEOUT = 120.0
POLL_BACKOFF = 10.0
MAX_MODEL_LEN = 8192
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234

TEMPERATURE = 0.4
# 0 disables top-k truncation, as in the Llama IID runner's TOP_K_REGIMES.
TOP_K_VALUES: tuple[int, ...] = (16, 0)

SAMPLING_CONFIGS = tuple(
    IIDSamplingConfig(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=top_k,
        max_tokens=MAX_TOKENS,
        stop=STOP_TOKENS,
    )
    for top_k in TOP_K_VALUES
)

DEFAULT_PRESEED_REGIONS: tuple[str, ...] = (
    "europe-west4",
    "us-central1",
    "us-central2",
    "us-east1",
    "us-east5",
    "us-west4",
)
WORKER_REGIONS_ANY = "any"


def make_task() -> GSM8KTask:
    return GSM8KTask(
        config=GSM8KTaskConfig(
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
                apply_rpa_block_size_patch=True,
            ),
        )
    )


def build_run_steps(worker_pools: tuple[WorkerPoolConfig, ...]) -> list[ExecutorStep]:
    resolved_pools = tuple(IIDPoolConfig(pool=pool, placements=TP2_PLACEMENTS[pool.pool_id]) for pool in worker_pools)
    return [
        make_eval_step(
            name="downstream_scaling/evals/delphi/gsm8k/iid_xregion/topk_sweep/1e23",
            model_path=output_path_of(DELPHI_HF_DOWNLOADS["1e23"]),
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
    prefix = "downstream_scaling/evals/delphi/gsm8k/iid_xregion/topk_sweep/preseed"
    return [
        _regional_download_step(
            DELPHI_HF_DOWNLOADS["1e23"],
            name=f"{prefix}/1e23/{region}",
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
        description = "Preseed Delphi 1e23 for the GSM8K IID top-k comparison."
    else:
        worker_regions = resolve_worker_regions(args.worker_regions, args.preseed_regions)
        worker_pools = make_worker_pools(
            tpu_types=args.tpu_types,
            worker_regions=worker_regions,
            num_workers=args.num_workers,
        )
        steps = build_run_steps(worker_pools)
        description = "Delphi 1e23 GSM8K IID comparison of top-k 16 with no top-k truncation at temperature 0.4."

    executor_main(steps=steps, description=description)


if __name__ == "__main__":
    configure_logging()
    main()
