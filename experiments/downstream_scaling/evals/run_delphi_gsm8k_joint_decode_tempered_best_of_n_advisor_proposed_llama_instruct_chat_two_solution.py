# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Llama-proposed, Delphi-ranked tempered best-of-N two-solution evals on GSM8K.

Decoder (A) proposes from Llama-3.1-8B-Instruct with a reference-conditioned prompt.
Its assistant prefill contains the reference under ``Solution 1:`` and
continues from ``Solution 2:`` in the same turn.
Advisor (B) ranks with each Delphi checkpoint using the five-shot GSM8K prompt
and the RPA patch. Only Llama receives the reference solution.

The existing ``tempered_best_of_n`` selector supports real N >= 1 and aligns
distinct EOS ids. N=1 samples Llama at TEMPERATURE; as N grows without bound,
the output approaches Delphi at its ranking temperature. Both endpoints use
the floor-filled candidate union. Final temperature is TEMPERATURE divided by
the Llama proposal temperature.

One step per Delphi checkpoint sweeps N and both temperatures, reusing each
worker child's engines and recording token paths. Metadata keeps its existing
names: advisor_weight is N, proposal_temperature is Llama's proposal temperature,
advisor_temperature is Delphi's ranking temperature, and final_temperature is
the derived ratio. The engine-B advisor_* fields refer to Delphi.

``--mode preseed`` downloads Llama and the non-skipped Delphi ladder from Hugging
Face into each preseed region, without cross-region transfers. ``--mode run``
launches eval steps; ``--worker-regions any`` places workers in any configured
TPU region once the default regions are preseeded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from itertools import product

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
from experiments.downstream_scaling.evals.tasks.gsm8k_chat_reference import (
    ReferenceChatGSM8KTaskConfig,
    TwoSolutionChatGSM8KTask,
)
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS
from experiments.models import ModelConfig, download_model_step

logger = logging.getLogger(__name__)

N_SAMPLES = 32
N_PROBLEMS = 256
WORKERS_PER_TPU_TYPE = 4
AGGREGATE_WORKERS = 32
CHUNK_SIZE = 512
# Single-VM, even-chip TPU types and their configured zones (from
# lib/iris/config/marin.yaml):
#   v5p-8        us-central1-a, us-east5-a
#   v5litepod-4  europe-west4-b, us-west4-a
#   v5litepod-8  europe-west4-b, us-west4-a
#   v6e-4        europe-west4-a, us-east1-d, us-east5-b
#   v6e-8        europe-west4-a, us-east1-d, us-east5-b
TPU_TYPES: tuple[str, ...] = ("v5p-8", "v5litepod-4", "v5litepod-8", "v6e-4", "v6e-8")
PLACEMENT_OVERRIDES: dict[tuple[str, str], tuple[JointDecodePlacement, ...]] = {}

# Generous enough to absorb the first-step XLA compilation skew between the
# two engines for the largest delphi sizes (60s was too short for 1e22).
BARRIER_TIMEOUT_S = 1200.0
HEARTBEAT_TIMEOUT = 120.0
POLL_BACKOFF = 10.0

MAX_TOKENS = 512
# Shared token-id vocabulary: both sides emit identical token counts for the
# same text, so the caps stay symmetric.
ADVISOR_MAX_TOKENS = MAX_TOKENS
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234

TEMPERATURE = 0.4  # Llama sampling temperature at N=1.
LLAMA_PROPOSAL_TEMPERATURES = (0.8, 1.0, 1.2, 1.6,)
DELPHI_RANKING_TEMPERATURES = (0.4,)
# Same top-k as the corresponding averaging experiment.
TOP_K_A = 16
TOP_K_B = 16
SELECTION_RULE = XtokSelectionRule.TEMPERED_BEST_OF_N
EFFECTIVE_NS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,)
ADVISOR_WEIGHTS = tuple(product(EFFECTIVE_NS, LLAMA_PROPOSAL_TEMPERATURES, DELPHI_RANKING_TEMPERATURES))

# Same pin as experiments.models.llama_3_1_8b_instruct, as an ExecutorStep so
# the regional preseed machinery below applies to it.
LLAMA_3_1_8B = download_model_step(ModelConfig(hf_repo_id="meta-llama/Llama-3.1-8B-Instruct", hf_revision="0e9e39f"))

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
    microbatch_size: int | None,
    delphi_model_path,
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
            advisor_model_path=delphi_model_path,
            decoder_model=JointDecodeModelConfig(),
            # The RPA patch stays with Delphi, now the ranking engine.
            advisor_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            execution=JointDecodeExecutionConfig(
                worker_pools=worker_pools,
                chunk_size=CHUNK_SIZE,
                microbatch_size=microbatch_size,
                heartbeat_timeout=HEARTBEAT_TIMEOUT,
                poll_backoff=POLL_BACKOFF,
                barrier_timeout_s=BARRIER_TIMEOUT_S,
                aggregate_workers=AGGREGATE_WORKERS,
            ),
            advisor_task=make_task(),
        )
    )


def build_run_steps(worker_pools: tuple[WorkerPoolConfig, ...]) -> list[ExecutorStep]:
    llama_model_path = output_path_of(LLAMA_3_1_8B)
    steps = []
    for slug, download in DELPHI_HF_DOWNLOADS.items():
        if slug in SKIP_DELPHI_KEYS:
            continue
        delphi_model_path = output_path_of(download)
        steps.append(
            make_eval_step(
                name=(
                    f"downstream_scaling/evals/delphi/gsm8k/joint_decode_avg_v2/"
                    f"{SELECTION_RULE.value}/advisor_proposed/llama_3_1_8b_instruct_chat_two_solution/{slug}"
                ),
                model_path=llama_model_path,
                task=TwoSolutionChatGSM8KTask(
                    config=ReferenceChatGSM8KTaskConfig(
                        tokenizer_path=llama_model_path,
                        n_problems=N_PROBLEMS,
                    )
                ),
                alg=make_algorithm(
                    worker_pools=joint_decode_pool_configs(slug, worker_pools, PLACEMENT_OVERRIDES),
                    microbatch_size=MICROBATCH_SIZE_BY_DELPHI_KEY.get(slug),
                    delphi_model_path=delphi_model_path,
                ),
            )
        )
    return steps


def _regional_download_step(base_step: ExecutorStep, *, name: str, region: str) -> ExecutorStep:
    if base_step.override_output_path is None:
        raise ValueError(f"Download step {base_step.name!r} does not define a stable relative output path")
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
                name=f"{prefix}/llama-3.1-8b-instruct/{region}",
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
        description = (
            "Preseed Llama-3.1-8B-Instruct and the Delphi ladder for the delphi GSM8K "
            "Llama-proposed, Delphi-ranked tempered best-of-N two-solution chat sweep."
        )
    else:
        worker_regions = resolve_worker_regions(args.worker_regions, args.preseed_regions)
        worker_pools = make_worker_pools(
            tpu_types=args.tpu_types,
            worker_regions=worker_regions,
            num_workers=args.num_workers,
        )
        steps = build_run_steps(worker_pools)
        description = (
            "Llama-proposed, Delphi-ranked tempered best-of-N evals on GSM8K "
            "(two-solution Llama-3.1-8B-Instruct prompts, N and temperature sweep)."
        )

    executor_main(
        steps=steps,
        description=description,
    )


if __name__ == "__main__":
    configure_logging()
    main()
