# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Luke's word-truncation GSM8K hints on the Delphi training-seed catalog.

Evaluates the 33 training-seed checkpoints through 1e22 FLOPs plus the
original 1e23 checkpoint with IID sampling on Luke Bailey's hinted GSM8K
prompts (``tasks/gsm8k_luke_hints.py``). Each checkpoint has two steps: one
covering the truncation method's ten hint levels, and the unhinted ``no_hint``
baseline, which the masking runner builds identically so either runner
reuses the other's completed baseline. Sampling uses temperature 0.4, top-p
1.0, top-k 1000, and 512 generated tokens.

``--mode preseed`` downloads HF models and copies the GCS-only seed exports
from us-central2 into each preseed region. The copies incur storage and
cross-region transfer costs. Before preseeding, repair the source configs:

    uv run python -m experiments.downstream_scaling.scripts.convert_delphi_seed_configs --apply

Complete preseeding in every worker region before ``--mode run``.
``--worker-regions any`` is supported after the default regions are preseeded.
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
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement, WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.gsm8k_luke_hints import (
    LukeHintMethod,
    LukeHintsGSM8KTask,
    LukeHintsGSM8KTaskConfig,
)
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS
from experiments.downstream_scaling.models.delphi_seed_sweep import MODEL_STEPS

logger = logging.getLogger(__name__)

HINT_METHOD = LukeHintMethod.TRUNCATE

N_SAMPLES = 32
WORKERS_PER_TPU_TYPE = 4
AGGREGATE_WORKERS = 32
CHUNK_SIZE = 512
TPU_TYPES: tuple[str, ...] = ("v5p-8", "v5litepod-4", "v5litepod-8", "v6e-4", "v6e-8")

HEARTBEAT_TIMEOUT = 120.0
POLL_BACKOFF = 10.0
MAX_MODEL_LEN = 8192
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

TEMPERATURE = 0.4
TOP_P = 1.0
TOP_K = 1000

SAMPLING_CONFIGS = (
    IIDSamplingConfig(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        stop=STOP_TOKENS,
    ),
)

RUN_MODEL_STEPS = {**MODEL_STEPS, "1e23": DELPHI_HF_DOWNLOADS["1e23"]}

# 1e22's BF16 weights exceed one 16-GiB v5litepod chip; 1e23's exceed two, so it
# needs TP2 on v5p and v6e and is skipped on v5litepod.
V5LITEPOD_TPU_TYPES = frozenset({"v5litepod-4", "v5litepod-8"})
V5LITEPOD_TP2_MODEL_KEYS = frozenset({"1e22", "1e22_seed42", "1e22_seed62746"})
TP2_MODEL_KEYS = frozenset({"1e23"})

# Multi-chip declarations libtpu accepts, per pool. v6e rejects every two-chip
# subset, so an engine there is handed a physical 2x2 quad and vLLM uses two of
# its chips. v5p-8 pairs need 1,2,1 bounds (probe 2026-10-16, Iris job
# /rohith/probe-v5p8-chip-bounds-145416). v5litepod-8 has exactly two valid pairs
# and v5litepod-4 none catalogued (.agents/projects/20260721_tpu_env_probe_plan.md);
# pools without an entry are skipped for TP2 models.
TP2_PLACEMENTS: dict[str, tuple[EnginePlacement, ...]] = {
    "v5p-8": (EnginePlacement((0, 1), (1, 2, 1), 2), EnginePlacement((2, 3), (1, 2, 1), 2)),
    "v6e-4": (EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2),),
    "v6e-8": (EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2), EnginePlacement((4, 5, 6, 7), (2, 2, 1), 2)),
    "v5litepod-8": (EnginePlacement((0, 1), (2, 1, 1), 2), EnginePlacement((2, 3), (1, 2, 1), 2)),
}

# Every region hosting a TPU_TYPES zone: preseeding all of them is what
# makes `--worker-regions any` safe.
DEFAULT_PRESEED_REGIONS: tuple[str, ...] = (
    "europe-west4",
    "us-central1",
    "us-central2",
    "us-east1",
    "us-east5",
    "us-west4",
)
WORKER_REGIONS_ANY = "any"


def make_task(method: LukeHintMethod) -> LukeHintsGSM8KTask:
    return LukeHintsGSM8KTask(config=LukeHintsGSM8KTaskConfig(method=method))


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


def model_pool_configs(model_key: str, worker_pools: tuple[WorkerPoolConfig, ...]) -> tuple[IIDPoolConfig, ...]:
    pools = []
    for pool in worker_pools:
        v5litepod = pool.pool_id in V5LITEPOD_TPU_TYPES
        if model_key in TP2_MODEL_KEYS:
            if v5litepod:
                continue
            placements = TP2_PLACEMENTS[pool.pool_id]
        elif v5litepod and model_key in V5LITEPOD_TP2_MODEL_KEYS:
            if pool.pool_id not in TP2_PLACEMENTS:
                continue
            placements = TP2_PLACEMENTS[pool.pool_id]
        else:
            placements = iid_tp1_placements(pool.chips_per_vm)
        pools.append(IIDPoolConfig(pool=pool, placements=placements))
    if not pools:
        raise ValueError(f"No selected TPU pool can serve {model_key}: {[pool.pool_id for pool in worker_pools]}")
    return tuple(pools)


def build_run_steps(worker_pools: tuple[WorkerPoolConfig, ...]) -> list[ExecutorStep]:
    return [
        make_eval_step(
            name=f"downstream_scaling/evals/delphi/gsm8k/iid_xregion/luke_hints_seeds/{method}/{model_key}",
            model_path=output_path_of(model_step),
            task=make_task(method),
            alg=make_algorithm(worker_pools=model_pool_configs(model_key, worker_pools)),
        )
        for model_key, model_step in RUN_MODEL_STEPS.items()
        for method in (HINT_METHOD, LukeHintMethod.NO_HINT)
    ]


def _regional_download_step(base_step: ExecutorStep, *, name: str, region: str) -> ExecutorStep:
    if base_step.override_output_path is None:
        raise ValueError(f"Model step {base_step.name!r} does not define a stable relative output path")
    bucket = data_config().region_buckets[region]
    regional_step = replace(base_step, name=name)
    return regional_step.with_output_path(f"gs://{bucket.name}/{base_step.override_output_path}")


def build_preseed_steps(regions: list[str]) -> list[ExecutorStep]:
    _validate_regions(regions, name="preseed regions")
    prefix = "downstream_scaling/evals/delphi/gsm8k/joint_decode_avg_xtok/preseed"
    return [
        _regional_download_step(
            model_step,
            name=f"{prefix}/{model_key}/{region}",
            region=region,
        )
        for region in regions
        for model_key, model_step in RUN_MODEL_STEPS.items()
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
        description = "Preseed the Delphi seed-sweep catalog and 1e23 for Luke's hinted GSM8K IID sweeps."
    else:
        worker_regions = resolve_worker_regions(args.worker_regions, args.preseed_regions)
        worker_pools = make_worker_pools(
            tpu_types=args.tpu_types,
            worker_regions=worker_regions,
            num_workers=args.num_workers,
        )
        steps = build_run_steps(worker_pools)
        description = f"Delphi training-seed IID evals on Luke's {HINT_METHOD} GSM8K hints, with the no_hint baseline."

    executor_main(steps=steps, description=description)


if __name__ == "__main__":
    configure_logging()
    main()
