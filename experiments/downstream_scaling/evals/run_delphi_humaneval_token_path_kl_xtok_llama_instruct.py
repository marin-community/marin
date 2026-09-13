# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-path KL for the delphi HumanEval joint-decode-avg-xtok Llama-3.1-8B-Instruct sweep.

Each statistic teacher-forces one Delphi checkpoint and Llama-3.1-8B-Instruct
along the sweep's recorded token paths, one engine per model on a chip pair,
and emits per-completion, per-step KL(delphi || llama).

Use ``--mode preseed`` to stage the models, then run with
``--mode run --prefix mirror://``. Missing completion steps run as dependencies.
"""

from __future__ import annotations

import argparse
import sys

from rigging.log_setup import configure_logging
from thalas.execution.executor import ExecutorStep, executor_main, output_path_of

from experiments.downstream_scaling.evals import run_delphi_humaneval_joint_decode_avg_xtok_llama_instruct as sweep
from experiments.downstream_scaling.evals.algorithms.joint_decode_avg_xtok import joint_decode_pool_configs
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import PROMPTS_FILENAME
from experiments.downstream_scaling.evals.measurements.kl import (
    TokenPathExecutionConfig,
    TokenPathKl,
    TokenPathModelConfig,
    TokenPathPoolConfig,
    token_path_pair_tp1_placements,
)
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS

MAX_MODEL_LEN = 8192
# The ~2 GiB prompt-logprob logits buffer (bounded by MAX_NUM_BATCHED_TOKENS)
# must fit in the HBM above this cap; below it, the advisor's 15 GiB of weights
# or the 1e22 decoder's 19 GiB plus a KV cache must fit. 0.85 satisfies both on
# v6e's 31 GiB; 0.7 left 1e22 no room for a KV cache there.
GPU_MEMORY_UTILIZATION = 0.8
# Matches the sweep's TOP_K_A and TOP_K_B, so the union the KL is taken over
# is the one the joint decoder selected from.
TOP_K = 16
# Temperature both models' top-k distributions are softmaxed at before the
# KL. A parameter of the statistic, independent of the sweep's sampling
# temperature.
TEMPERATURE = 0.7 # 1.0


def build_run_steps(worker_regions: list[str]) -> list[ExecutorStep]:
    worker_pools = sweep.make_worker_pools(
        tpu_types=list(sweep.TPU_TYPES),
        worker_regions=worker_regions,
        num_workers=sweep.WORKERS_PER_TPU_TYPE,
    )
    kl_execution = TokenPathExecutionConfig(
        worker_pools=tuple(
            TokenPathPoolConfig(
                pool=pool,
                placements=token_path_pair_tp1_placements(pool.chips_per_vm),
            )
            for pool in worker_pools
        ),
        microbatch_size=sweep.CHUNK_SIZE,
        chunk_size=sweep.CHUNK_SIZE,
    )
    prompts_path = output_path_of(sweep.make_task().make_prompts_step()) / PROMPTS_FILENAME
    advisor_model_path = output_path_of(sweep.LLAMA_3_1_8B)

    steps = []
    for slug in DELPHI_HF_DOWNLOADS:
        if slug in sweep.SKIP_DELPHI_KEYS:
            continue
        completions = make_eval_step(
            name=f"{sweep.EVAL_NAME_BASE}/{slug}",
            model_path=output_path_of(DELPHI_HF_DOWNLOADS[slug]),
            task=sweep.make_task(),
            alg=sweep.make_algorithm(
                worker_pools=joint_decode_pool_configs(slug, worker_pools, sweep.PLACEMENT_OVERRIDES),
                microbatch_size=sweep.MICROBATCH_SIZE_BY_DELPHI_KEY.get(slug),
                advisor_model_path=advisor_model_path,
            ),
            skip_grades=True,
        )
        steps.append(
            TokenPathKl(
                decoder_model=TokenPathModelConfig(
                    model_path=output_path_of(DELPHI_HF_DOWNLOADS[slug]),
                    max_model_len=MAX_MODEL_LEN,
                    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                    apply_rpa_block_size_patch=True,
                ),
                # The RPA patch is delphi-specific and harms standard models;
                # the llama advisor runs unpatched.
                advisor_model=TokenPathModelConfig(
                    model_path=advisor_model_path,
                    max_model_len=MAX_MODEL_LEN,
                    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                ),
                execution=kl_execution,
                k=TOP_K,
                temperature=TEMPERATURE,
            ).make_statistic_step(
                name=f"{sweep.EVAL_NAME_BASE}/{slug}/kl",
                prompts_path=prompts_path,
                alg_output_path=output_path_of(completions),
            )
        )
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--mode", choices=("preseed", "run"), required=True)
    parser.add_argument("--preseed-regions", nargs="+", default=list(sweep.DEFAULT_PRESEED_REGIONS))
    parser.add_argument(
        "--worker-regions",
        nargs="+",
        default=None,
        help=(
            "TPU worker placement regions. Defaults to --preseed-regions. "
            f"Use '{sweep.WORKER_REGIONS_ANY}' for unrestricted placement."
        ),
    )
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "preseed":
        steps = sweep.build_preseed_steps(args.preseed_regions)
        description = "Preseed Llama-3.1-8B-Instruct and the Delphi ladder for HumanEval token-path KL scoring."
    else:
        worker_regions = sweep.resolve_worker_regions(args.worker_regions, args.preseed_regions)
        steps = build_run_steps(worker_regions)
        description = (
            "Per-completion, per-step KL(delphi || llama) along the delphi HumanEval "
            "joint-decode-avg-xtok Llama-3.1-8B-Instruct sweep's token paths."
        )

    executor_main(
        steps=steps,
        description=description,
    )


if __name__ == "__main__":
    configure_logging()
    main()
