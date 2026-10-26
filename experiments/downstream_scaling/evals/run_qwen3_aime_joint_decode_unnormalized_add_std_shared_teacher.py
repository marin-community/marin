# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 shared-teacher std-matched logit-summing sweeps on AIME, without grading.

Every catalog model is a student; Qwen3-14B is the fixed teacher.
Students receive the question and the teacher receives a stored full hint.
Each problem has ten hint samples, each swept over the positive alpha grid.
One completion step per student reuses its engines across the alpha sweep.

Complete ``--mode preseed`` in every worker region before ``--mode run``.
``--worker-regions any`` is supported after the default regions are preseeded.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from fray.cluster import ANY_REGION, ResourceConfig, get_tpu_topology
from rigging.filesystem.cluster_config import data_config
from rigging.log_setup import configure_logging
from thalas.execution.executor import ExecutorStep, executor_main, output_path_of

from experiments.downstream_scaling.evals.algorithms.joint_decode_avg_v2 import (
    JointDecodeCompletionAlgorithm,
    JointDecodeConfig,
    JointDecodeExecutionConfig,
    JointDecodeModelConfig,
    JointDecodePlacement,
    JointDecodePoolConfig,
    JointDecodeSamplingConfig,
    XtokSelectionRule,
    joint_decode_pool_configs,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement, WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.luke_hinted_benchmarks import (
    AIMEAdvisorTask,
    AIMETask,
    HintedBenchmarkTaskConfig,
)
from experiments.downstream_scaling.models.posttrained import QWEN3_MODEL_STEPS as MODEL_STEPS

N_SAMPLES = 1
WORKERS_PER_TPU_TYPE = 4
AGGREGATE_WORKERS = 32
CHUNK_SIZE = 64
TPU_TYPES: tuple[str, ...] = ("v5p-8", "v6e-8")
ADVISOR_KEY = "Qwen3-14B"

# v6e TP2 uses a four-chip declaration, following the existing placement pattern.
PLACEMENT_OVERRIDES: dict[tuple[str, str], tuple[JointDecodePlacement, ...]] = {
    (model_key, "v6e-8"): (
        JointDecodePlacement(
            decoder=EnginePlacement((4,), (1, 1, 1), 1),
            advisor=EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2),
        ),
    )
    for model_key in MODEL_STEPS
    if model_key != ADVISOR_KEY
}

# Allow first-step XLA compilation skew between the two engines.
BARRIER_TIMEOUT_S = 1200.0
HEARTBEAT_TIMEOUT = 120.0
POLL_BACKOFF = 10.0

MAX_TOKENS = 32000
ADVISOR_MAX_TOKENS = MAX_TOKENS
MAX_MODEL_LEN = 32768
SEED = 42
TEMPERATURE = 0.6
TOP_K_A = 16
TOP_K_B = 16
SELECTION_RULE = XtokSelectionRule.UNNORMALIZED_ADD_NO_VOCAB_STD
ALPHAS: tuple[float, ...] = (0.11, 0.25, 0.43, 0.67, 1.0, 1.5, 2.33, 4.0, 9.0, 16.0)

# Preseed every worker region before using --worker-regions any.
DEFAULT_PRESEED_REGIONS: tuple[str, ...] = (
    "europe-west4",
    "us-central1",
    "us-central2",
    "us-east1",
    "us-east5",
)
WORKER_REGIONS_ANY = "any"


def make_algorithm(*, worker_pools: tuple[JointDecodePoolConfig, ...]) -> JointDecodeCompletionAlgorithm:
    advisor_model_path = output_path_of(MODEL_STEPS[ADVISOR_KEY])
    return JointDecodeCompletionAlgorithm(
        config=JointDecodeConfig(
            sampling=JointDecodeSamplingConfig(
                n_samples=N_SAMPLES,
                max_tokens=MAX_TOKENS,
                advisor_max_tokens=ADVISOR_MAX_TOKENS,
                top_k_a=TOP_K_A,
                top_k_b=TOP_K_B,
                seed=SEED,
                selection_rule=SELECTION_RULE,
                advisor_weights=ALPHAS,
                temperature=TEMPERATURE,
                stop=(),
            ),
            advisor_model_path=advisor_model_path,
            advisor_task=AIMEAdvisorTask(config=HintedBenchmarkTaskConfig(tokenizer_path=advisor_model_path)),
            decoder_model=JointDecodeModelConfig(
                max_model_len=MAX_MODEL_LEN,
                apply_rpa_block_size_patch=False,
                enable_prefix_caching=False,
                gpu_memory_utilization=0.85,
            ),
            advisor_model=JointDecodeModelConfig(
                max_model_len=MAX_MODEL_LEN,
                apply_rpa_block_size_patch=False,
                enable_prefix_caching=False,
                gpu_memory_utilization=0.85,
            ),
            execution=JointDecodeExecutionConfig(
                worker_pools=worker_pools,
                chunk_size=CHUNK_SIZE,
                microbatch_size=None,
                max_num_batched_tokens=MAX_MODEL_LEN,
                heartbeat_timeout=HEARTBEAT_TIMEOUT,
                poll_backoff=POLL_BACKOFF,
                barrier_timeout_s=BARRIER_TIMEOUT_S,
                aggregate_workers=AGGREGATE_WORKERS,
            ),
        )
    )


def model_worker_pools(model_key: str, worker_pools: tuple[WorkerPoolConfig, ...]) -> tuple[WorkerPoolConfig, ...]:
    # Two four-chip engines cannot share a v6e-8 VM: libtpu refuses the second
    # process, so the self-teacher step runs without the v6e-8 pool.
    if model_key != ADVISOR_KEY:
        return worker_pools
    return tuple(pool for pool in worker_pools if pool.pool_id != "v6e-8")


def build_run_steps(worker_pools: tuple[WorkerPoolConfig, ...]) -> list[ExecutorStep]:
    return [
        make_eval_step(
            name=(
                f"downstream_scaling/evals/qwen3/aime/joint_decode_avg_v2/"
                f"{SELECTION_RULE.value}/shared_teacher_{ADVISOR_KEY}/{model_key}"
            ),
            model_path=output_path_of(model_step),
            task=AIMETask(config=HintedBenchmarkTaskConfig(tokenizer_path=output_path_of(model_step))),
            alg=make_algorithm(
                worker_pools=joint_decode_pool_configs(
                    model_key, model_worker_pools(model_key, worker_pools), PLACEMENT_OVERRIDES
                )
            ),
            skip_grades=True,
        )
        for model_key, model_step in MODEL_STEPS.items()
    ]


def _regional_download_step(base_step: ExecutorStep, *, name: str, region: str) -> ExecutorStep:
    if base_step.override_output_path is None:
        raise ValueError(f"Model step {base_step.name!r} does not define a stable relative output path")
    bucket = data_config().region_buckets[region]
    regional_step = replace(base_step, name=name)
    return regional_step.with_output_path(f"gs://{bucket.name}/{base_step.override_output_path}")


def build_preseed_steps(regions: list[str]) -> list[ExecutorStep]:
    _validate_regions(regions, name="preseed regions")
    prefix = "downstream_scaling/evals/qwen3/joint_decode_avg_v2/preseed"
    return [
        _regional_download_step(
            model_step,
            name=f"{prefix}/{slug}/{region}",
            region=region,
        )
        for region in regions
        for slug, model_step in MODEL_STEPS.items()
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
        if topology.vm_count != 1:
            raise ValueError(f"joint decode xtok supports only single-VM TPU types, got {tpu_type}")
        if topology.chips_per_vm % 2 != 0:
            raise ValueError(f"joint decode xtok needs even chips_per_vm, got {tpu_type}")
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
    parser.add_argument("--tpu-types", nargs="+", choices=TPU_TYPES, default=list(TPU_TYPES))
    parser.add_argument(
        "--num-workers",
        type=int,
        default=WORKERS_PER_TPU_TYPE,
        help="Workers per TPU type.",
    )

    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "preseed":
        steps = build_preseed_steps(args.preseed_regions)
        description = "Preseed Qwen3 models for shared-teacher std-matched logit-summing sweeps."
    else:
        worker_regions = resolve_worker_regions(args.worker_regions, args.preseed_regions)
        worker_pools = make_worker_pools(
            tpu_types=args.tpu_types,
            worker_regions=worker_regions,
            num_workers=args.num_workers,
        )
        steps = build_run_steps(worker_pools)
        description = (
            "Qwen3 AIME std-matched unnormalized-add alpha sweep with fixed Qwen3-14B teacher "
            "and full hints; completions only."
        )

    executor_main(steps=steps, description=description)


if __name__ == "__main__":
    configure_logging()
    main()
