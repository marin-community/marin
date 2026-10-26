# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GSM8K logit averaging with the student as its own two-solution advisor.

Decoder (A) = each of the 33 Delphi training-seed checkpoints through 1e22 FLOPs.
Advisor (B) = the same checkpoint, RPA patch on. The decoder uses the standard
five-shot prompt. The advisor sees the reference solution and is prompted
for a second derivation, using the existing base-Llama two-solution prompts
whose exemplars were sampled from Llama-3.1-8B. Floor-completed top-k-union
logits are averaged using the existing AVG_LOGITS rule.

``--mode preseed`` downloads HF models, including Llama-3.1-8B for the reused
prompts, and copies the GCS-only seed exports from us-central2 into each
preseed region. The copies incur storage and cross-region transfer costs.
Before preseeding, repair the source configs:

    uv run python -m experiments.downstream_scaling.scripts.convert_delphi_seed_configs --apply

Complete preseeding in every worker region before ``--mode run``.
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
from experiments.downstream_scaling.evals.framework.xregion.pool import WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.evals.tasks.gsm8k_two_solution import (
    TwoSolutionGSM8KTask,
    TwoSolutionGSM8KTaskConfig,
)
from experiments.downstream_scaling.models.delphi_seed_sweep import MODEL_STEPS
from experiments.models import ModelConfig, download_model_step

N_SAMPLES = 32
N_PROBLEMS = 256
WORKERS_PER_TPU_TYPE = 4
AGGREGATE_WORKERS = 32
CHUNK_SIZE = 512
TPU_TYPES: tuple[str, ...] = ("v5p-8", "v5litepod-4", "v5litepod-8", "v6e-4", "v6e-8")
PLACEMENT_OVERRIDES: dict[tuple[str, str], tuple[JointDecodePlacement, ...]] = {}

BARRIER_TIMEOUT_S = 1200.0
HEARTBEAT_TIMEOUT = 120.0
POLL_BACKOFF = 10.0

MAX_TOKENS = 512
ADVISOR_MAX_TOKENS = MAX_TOKENS
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234

TEMPERATURE = 0.4
TOP_K_A = 16
TOP_K_B = 16
SELECTION_RULE = XtokSelectionRule.AVG_LOGITS
ADVISOR_WEIGHTS: tuple[float, ...] = tuple(i / 10.0 for i in range(11))

# Base Llama that sampled the reused two-solution exemplars; not an engine here.
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


def make_task() -> GSM8KTask:
    return GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )


def make_algorithm(
    *,
    worker_pools: tuple[JointDecodePoolConfig, ...],
    advisor_model_path,
    advisor_worker_pools: tuple[WorkerPoolConfig, ...],
) -> JointDecodeCompletionAlgorithm:
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
                advisor_weights=ADVISOR_WEIGHTS,
                temperature=TEMPERATURE,
                stop=STOP_TOKENS,
            ),
            advisor_model_path=advisor_model_path,
            advisor_task=TwoSolutionGSM8KTask(
                config=TwoSolutionGSM8KTaskConfig(
                    advisor_model_path=output_path_of(LLAMA_3_1_8B),
                    worker_pools=tuple(replace(pool, num_workers=1) for pool in advisor_worker_pools),
                    heartbeat_timeout=HEARTBEAT_TIMEOUT,
                    poll_backoff=POLL_BACKOFF,
                    n_problems=N_PROBLEMS,
                )
            ),
            decoder_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            advisor_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            execution=JointDecodeExecutionConfig(
                worker_pools=worker_pools,
                chunk_size=CHUNK_SIZE,
                microbatch_size=None,
                heartbeat_timeout=HEARTBEAT_TIMEOUT,
                poll_backoff=POLL_BACKOFF,
                barrier_timeout_s=BARRIER_TIMEOUT_S,
                aggregate_workers=AGGREGATE_WORKERS,
            ),
        )
    )


def build_run_steps(worker_pools: tuple[WorkerPoolConfig, ...]) -> list[ExecutorStep]:
    return [
        make_eval_step(
            name=(
                f"downstream_scaling/evals/delphi/gsm8k/joint_decode_avg_v2/"
                f"{SELECTION_RULE.value}/delphi_self_two_solution/{slug}"
            ),
            model_path=output_path_of(model_step),
            task=make_task(),
            alg=make_algorithm(
                worker_pools=joint_decode_pool_configs(slug, worker_pools, PLACEMENT_OVERRIDES),
                advisor_model_path=output_path_of(model_step),
                advisor_worker_pools=worker_pools,
            ),
        )
        for slug, model_step in MODEL_STEPS.items()
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
    steps = []
    for region in regions:
        steps.append(
            _regional_download_step(
                LLAMA_3_1_8B,
                name=f"{prefix}/llama-3.1-8b/{region}",
                region=region,
            )
        )
        steps.extend(
            _regional_download_step(
                model_step,
                name=f"{prefix}/{slug}/{region}",
                region=region,
            )
            for slug, model_step in MODEL_STEPS.items()
        )
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
            raise ValueError(f"joint decode supports only single-VM TPU types, got {tpu_type}")
        if topology.chips_per_vm % 2 != 0:
            raise ValueError(f"joint decode needs even chips_per_vm, got {tpu_type}")
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
        description = "Preseed Llama-3.1-8B and the Delphi seed catalog for the GSM8K two-solution-advisor sweep."
    else:
        worker_regions = resolve_worker_regions(args.worker_regions, args.preseed_regions)
        worker_pools = make_worker_pools(
            tpu_types=args.tpu_types,
            worker_regions=worker_regions,
            num_workers=args.num_workers,
        )
        steps = build_run_steps(worker_pools)
        description = (
            "Delphi compute and training-seed logit-averaging evals on GSM8K "
            "(two-solution self advisor, advisor-weight sweep)."
        )

    executor_main(steps=steps, description=description)


if __name__ == "__main__":
    configure_logging()
    main()
