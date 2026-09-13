# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sampling-rule statistics along the GSM8K Llama-3.1-8B unnormalized-add paths.

Use --mode preseed to stage the models, then --mode run --prefix mirror://.
Missing completion steps run as dependencies.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from rigging.log_setup import configure_logging
from thalas.execution.executor import ExecutorStep, executor_main, output_path_of

from experiments.downstream_scaling.evals import run_delphi_gsm8k_joint_decode_unnormalized_add_llama as sweep
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.schema import PROMPTS_FILENAME
from experiments.downstream_scaling.evals.measurements.joint_decode_replay import (
    JointDecodeExecutionConfig,
    JointDecodeModelConfig,
    JointDecodePlacement,
    JointDecodeReplay,
    JointDecodeReplayConfig,
    JointDecodeReplaySamplingConfig,
    ReplayStatistic,
    XtokSelectionRule,
    joint_decode_pool_configs,
)
from experiments.downstream_scaling.models.delphi import DELPHI_HF_DOWNLOADS

STATISTICS: tuple[ReplayStatistic, ...] = (
    ReplayStatistic.KL,
    ReplayStatistic.REVERSE_KL,
    ReplayStatistic.TOTAL_VARIATION,
    ReplayStatistic.JENSEN_SHANNON,
    ReplayStatistic.HELLINGER_SQUARED,
)
# Each entry must match one source generation batch, including alpha order.
ALPHA_SWEEPS: tuple[tuple[float, ...], ...] = (
    (*(i / 10.0 for i in range(13)), 2.0),
    # (1.6, 3.0, 4.0, 5.0, 7.0, 10.0),
)


def build_run_steps(worker_regions: list[str]) -> list[ExecutorStep]:
    worker_pools = sweep.make_worker_pools(
        tpu_types=list(sweep.TPU_TYPES),
        worker_regions=worker_regions,
        num_workers=sweep.WORKERS_PER_TPU_TYPE,
    )
    prompts_path = output_path_of(sweep.make_task().make_prompts_step()) / PROMPTS_FILENAME
    advisor_model_path = output_path_of(sweep.LLAMA_3_1_8B)
    sampling = JointDecodeReplaySamplingConfig(
        max_tokens=sweep.MAX_TOKENS,
        advisor_max_tokens=sweep.ADVISOR_MAX_TOKENS,
        top_k_a=sweep.TOP_K_A,
        top_k_b=sweep.TOP_K_B,
        seed=sweep.SEED,
        selection_rule=XtokSelectionRule(sweep.SELECTION_RULE.value),
        temperature=sweep.TEMPERATURE,
        stop=sweep.STOP_TOKENS,
    )
    overrides = {
        key: tuple(JointDecodePlacement(decoder=pair.decoder, advisor=pair.advisor) for pair in placements)
        for key, placements in sweep.PLACEMENT_OVERRIDES.items()
    }
    steps = []
    for slug in DELPHI_HF_DOWNLOADS:
        if slug in sweep.SKIP_DELPHI_KEYS:
            continue
        eval_name = (
            "downstream_scaling/evals/delphi/gsm8k/joint_decode_avg_v2/"
            f"{sweep.SELECTION_RULE.value}/llama_3_1_8b/{slug}"
        )
        decoder_model_path = output_path_of(DELPHI_HF_DOWNLOADS[slug])
        microbatch_size = sweep.MICROBATCH_SIZE_BY_DELPHI_KEY.get(slug)
        algorithm = sweep.make_algorithm(
            worker_pools=sweep.joint_decode_pool_configs(slug, worker_pools, sweep.PLACEMENT_OVERRIDES),
            microbatch_size=microbatch_size,
            advisor_model_path=advisor_model_path,
        )
        for alphas in ALPHA_SWEEPS:
            completions = make_eval_step(
                name=eval_name,
                model_path=decoder_model_path,
                task=sweep.make_task(),
                alg=replace(
                    algorithm,
                    config=replace(
                        algorithm.config,
                        sampling=replace(algorithm.config.sampling, advisor_weights=alphas),
                    ),
                ),
                skip_grades=True,
            )
            steps.append(
                JointDecodeReplay(
                    decoder_model_path=decoder_model_path,
                    config=JointDecodeReplayConfig(
                        sampling=sampling,
                        statistics=STATISTICS,
                        advisor_model_path=advisor_model_path,
                        advisor_prompts_path=completions.config.advisor_prompts_path,
                        decoder_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
                        advisor_model=JointDecodeModelConfig(),
                        execution=JointDecodeExecutionConfig(
                            worker_pools=joint_decode_pool_configs(slug, worker_pools, overrides),
                            chunk_size=sweep.CHUNK_SIZE,
                            microbatch_size=microbatch_size,
                            heartbeat_timeout=sweep.HEARTBEAT_TIMEOUT,
                            poll_backoff=sweep.POLL_BACKOFF,
                            barrier_timeout_s=sweep.BARRIER_TIMEOUT_S,
                            aggregate_workers=sweep.AGGREGATE_WORKERS,
                        ),
                    ),
                ).make_statistic_step(
                    name=f"{eval_name}/joint_decode_replay",
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
        description = "Preseed Llama-3.1-8B and the Delphi ladder for GSM8K unnormalized-add replay."
    else:
        worker_regions = sweep.resolve_worker_regions(args.worker_regions, args.preseed_regions)
        steps = build_run_steps(worker_regions)
        description = "Per-decision statistics for the GSM8K Llama-3.1-8B unnormalized-add sweep."
    executor_main(steps=steps, description=description)


if __name__ == "__main__":
    configure_logging()
    main()
