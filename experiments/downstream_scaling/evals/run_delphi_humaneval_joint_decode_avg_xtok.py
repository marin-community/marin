# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-tokenizer joint-decode-avg evals on HumanEval for the Delphi ladder.

Decoder (A) = each Delphi checkpoint in turn (llama tokenizer, RPA patch on).
Advisor (B) = Qwen3-4B-Base (Qwen tokenizer). Selection rule =
anchored_prefix_mass; one step per checkpoint sweeps the full advisor-weight
grid through a single engine load per worker child.

Generation settings follow lm-eval's ``humaneval.yaml``: zero-shot, stop at
code-block boundaries, and 1024 generated tokens.

``--mode preseed`` downloads the advisor and the (non-skipped) Delphi ladder
from Hugging Face into each preseed region — per-region downloads, so the
cost is storage only, no cross-region transfer. ``--mode run`` launches the
eval steps; ``--worker-regions any`` places workers in any configured TPU
region, correct by construction once the default regions are preseeded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from fray.cluster import ANY_REGION, ResourceConfig, get_tpu_topology
from rigging.filesystem import data_config
from rigging.log_setup import configure_logging
from thalas.execution.executor import ExecutorStep, executor_main, output_path_of

from experiments.downstream_scaling.evals.algorithms.joint_decode_avg_xtok import (
    JointDecodeCompletionAlgorithm,
    JointDecodeConfig,
    JointDecodeExecutionConfig,
    JointDecodeModelConfig,
    JointDecodeSamplingConfig,
    XtokSelectionRule,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.xregion.pool import WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.humaneval import HumanEvalTask, HumanEvalTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS
from experiments.models import ModelConfig, download_model_step

logger = logging.getLogger(__name__)

N_SAMPLES = 32
# All 164 HumanEval problems.
N_PROBLEMS: int | None = None
WORKERS_PER_TPU_TYPE = 2
AGGREGATE_WORKERS = 32
CHUNK_SIZE = 64
# Single-VM, even-chip TPU types and their configured zones (from
# lib/iris/config/marin.yaml):
#   v4-8         us-central2-b
#   v5p-8        us-central1-a, us-east5-a
#   v5litepod-4  europe-west4-b, us-west4-a
#   v5litepod-8  europe-west4-b, us-west4-a
#   v6e-4        europe-west4-a, us-east1-d, us-east5-b
#   v6e-8        europe-west4-a, us-east1-d, us-east5-b
TPU_TYPES: tuple[str, ...] = ("v4-8", "v5p-8", "v5litepod-4", "v5litepod-8", "v6e-4", "v6e-8")

# Generous enough to absorb the first-step XLA compilation skew between the
# two engines for the largest delphi sizes (60s was too short for 1e22).
BARRIER_TIMEOUT_S = 1200.0
HEARTBEAT_TIMEOUT = 120.0
POLL_BACKOFF = 10.0

# lm-eval humaneval.yaml parity: max_gen_toks 1024, stop at code-block
# boundaries (see the HumanEval task plan).
MAX_TOKENS = 1024
# 2x decoder cap: fertility headroom for the advisor on the same text
# (Qwen tokenizes digits singly; llama 3 groups up to three).
ADVISOR_MAX_TOKENS = 2 * MAX_TOKENS
SEED = 42
STOP_TOKENS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")

NUM_FEWSHOT = 0
FEWSHOT_SEED = 1234

TEMPERATURE = 0.4
TOP_K_A = 16
# Deeper than same-tokenizer runs: byte overlap across vocabs is sparse and
# prefix mass improves with depth; cost is only JSON payload size.
TOP_K_B = 64
SELECTION_RULE = XtokSelectionRule.ANCHORED_PREFIX_MASS
ADVISOR_WEIGHTS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PREFIX_CREDIT = 1.0

# Same pin as experiments.models.qwen3_4b_base, as an ExecutorStep so the
# regional preseed machinery below applies to it.
QWEN3_4B_BASE = download_model_step(ModelConfig(hf_repo_id="Qwen/Qwen3-4B-Base", hf_revision="906bfd4"))

# 1e23 doesn't fit on a single TPU chip at TP=1; tensor-parallelism support
# is on the todo list. Skip for now.
SKIP_DELPHI_KEYS = frozenset({"1e23"})

# Microbatch the largest delphi sizes whose A/B vLLM schedulers otherwise
# desync; absent keys → None (whole chunk in one round-trip).
MICROBATCH_SIZE_BY_DELPHI_KEY: dict[str, int] = {"1e22": 8}

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


def make_task() -> HumanEvalTask:
    return HumanEvalTask(
        config=HumanEvalTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )


def make_algorithm(
    *,
    worker_pools: tuple[WorkerPoolConfig, ...],
    microbatch_size: int | None,
    advisor_model_path,
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
                prefix_credit=PREFIX_CREDIT,
                stop=STOP_TOKENS,
            ),
            advisor_model_path=advisor_model_path,
            decoder_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            # The RPA patch is delphi-specific and harms standard models; the
            # Qwen advisor runs unpatched.
            advisor_model=JointDecodeModelConfig(),
            execution=JointDecodeExecutionConfig(
                worker_pools=worker_pools,
                chunk_size=CHUNK_SIZE,
                microbatch_size=microbatch_size,
                heartbeat_timeout=HEARTBEAT_TIMEOUT,
                poll_backoff=POLL_BACKOFF,
                barrier_timeout_s=BARRIER_TIMEOUT_S,
                aggregate_workers=AGGREGATE_WORKERS,
            ),
        )
    )


def build_run_steps(worker_pools: tuple[WorkerPoolConfig, ...]) -> list[ExecutorStep]:
    advisor_model_path = output_path_of(QWEN3_4B_BASE)
    return [
        make_eval_step(
            name=(
                f"downstream_scaling/evals/delphi/humaneval/joint_decode_avg_xtok/"
                f"{SELECTION_RULE.value}/qwen3_4b_base/{slug}"
            ),
            model_path=output_path_of(DELPHI_HF_DOWNLOADS[slug]),
            task=make_task(),
            alg=make_algorithm(
                worker_pools=worker_pools,
                microbatch_size=MICROBATCH_SIZE_BY_DELPHI_KEY.get(slug),
                advisor_model_path=advisor_model_path,
            ),
        )
        for slug in DELPHI_HF_DOWNLOADS
        if slug not in SKIP_DELPHI_KEYS
    ]


def _regional_download_step(base_step: ExecutorStep, *, name: str, region: str) -> ExecutorStep:
    if base_step.override_output_path is None:
        raise ValueError(f"Download step {base_step.name!r} does not define a stable relative output path")
    bucket = data_config().region_buckets[region]
    regional_step = replace(base_step, name=name)
    return regional_step.with_output_path(f"gs://{bucket.name}/{base_step.override_output_path}")


def build_preseed_steps(regions: list[str]) -> list[ExecutorStep]:
    _validate_regions(regions, name="preseed regions")
    prefix = "downstream_scaling/evals/delphi/humaneval/joint_decode_avg_xtok/preseed"
    steps = []
    for region in regions:
        steps.append(
            _regional_download_step(
                QWEN3_4B_BASE,
                name=f"{prefix}/qwen3-4b-base/{region}",
                region=region,
            )
        )
        steps.extend(
            _regional_download_step(
                download,
                name=f"{prefix}/{slug}/{region}",
                region=region,
            )
            for slug, download in DELPHI_HF_DOWNLOADS.items()
            if slug not in SKIP_DELPHI_KEYS
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
        description = "Preseed Qwen3-4B-Base and the Delphi ladder for the delphi HumanEval joint-decode-avg-xtok sweep."
    else:
        worker_regions = resolve_worker_regions(args.worker_regions, args.preseed_regions)
        worker_pools = make_worker_pools(
            tpu_types=args.tpu_types,
            worker_regions=worker_regions,
            num_workers=args.num_workers,
        )
        steps = build_run_steps(worker_pools)
        description = (
            "Delphi scaling-ladder cross-tokenizer joint-decode-avg evals on HumanEval "
            "(anchored prefix mass, Qwen3-4B-Base advisor, advisor-weight sweep)."
        )

    executor_main(
        steps=steps,
        description=description,
    )


if __name__ == "__main__":
    configure_logging()
    main()
