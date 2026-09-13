# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Advisor/decoder entropy correlation for the delphi GSM8K entropy-gated sweep.

Each statistic teacher-forces Llama-3.1-8B and one Delphi checkpoint along the
sweep's recorded token paths, then emits a per-completion Spearman correlation.

Use ``--mode preseed`` to stage the models, then run with
``--mode run --prefix mirror://``. Missing completion steps run as dependencies.
"""

from __future__ import annotations

import argparse
import sys

from rigging.log_setup import configure_logging
from thalas.execution.executor import ExecutorStep, executor_main, output_path_of

from experiments.downstream_scaling.evals import run_delphi_gsm8k_joint_decode_entropy_xtok_llama as sweep
from experiments.downstream_scaling.evals.algorithms.joint_decode_entropy_xtok import joint_decode_pool_configs
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import PROMPTS_FILENAME
from experiments.downstream_scaling.evals.measurements.entropy import (
    TokenPathExecutionConfig,
    TokenPathModelConfig,
    TokenPathPoolConfig,
    token_path_tp1_placements,
)
from experiments.downstream_scaling.evals.measurements.entropy_correlation import TokenPathEntropyCorrelation
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS

MAX_MODEL_LEN = 8192
TOP_K = 16


def build_run_steps(worker_regions: list[str]) -> list[ExecutorStep]:
    worker_pools = sweep.make_worker_pools(
        tpu_types=list(sweep.TPU_TYPES),
        worker_regions=worker_regions,
        num_workers=sweep.WORKERS_PER_TPU_TYPE,
    )
    entropy_execution = TokenPathExecutionConfig(
        worker_pools=tuple(
            TokenPathPoolConfig(
                pool=pool,
                placements=token_path_tp1_placements(pool.chips_per_vm),
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
            TokenPathEntropyCorrelation(
                decoder_model=TokenPathModelConfig(
                    model_path=output_path_of(DELPHI_HF_DOWNLOADS[slug]),
                    max_model_len=MAX_MODEL_LEN,
                    gpu_memory_utilization=0.5,
                    apply_rpa_block_size_patch=True,
                ),
                advisor_model=TokenPathModelConfig(
                    model_path=advisor_model_path,
                    max_model_len=MAX_MODEL_LEN,
                    gpu_memory_utilization=0.5,
                ),
                execution=entropy_execution,
                entropy_name=f"{sweep.EVAL_NAME_BASE}/{slug}/token_path_entropy",
                k=TOP_K,
            ).make_statistic_step(
                name=f"{sweep.EVAL_NAME_BASE}/{slug}/entropy_correlation",
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
        description = "Preseed Llama-3.1-8B and the Delphi ladder for GSM8K entropy-correlation scoring."
    else:
        worker_regions = sweep.resolve_worker_regions(args.worker_regions, args.preseed_regions)
        steps = build_run_steps(worker_regions)
        description = (
            "Per-completion advisor/decoder entropy correlations for the delphi GSM8K "
            "advisor-entropy-gated joint-decode sweep."
        )

    executor_main(
        steps=steps,
        description=description,
    )


if __name__ == "__main__":
    configure_logging()
    main()
