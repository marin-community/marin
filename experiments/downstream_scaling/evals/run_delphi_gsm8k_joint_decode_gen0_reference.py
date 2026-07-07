# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference side of the joint-decode Phase 3 A/B gate: generation-0
implementation, delphi GSM8K.

Runs the generation-0 xregion joint-decode-avg algorithm (the only gen-0
variant that runs at this branch's pins — the non-xregion modules patch the
removed ``get_tuned_block_sizes`` RPA API) with a single single-VM worker
pool. Sampling constants match ``run_delphi_gsm8k_joint_decode_avg.py`` and
the unified script on the jd-unify-phase3 branch, so scores are directly
comparable. Lives on the ``jd-gen0-reference`` branch only; see
.agents/projects/joint_decode_unification_plan.md.

Gate configuration:

    --delphi-keys 3e18 --advisor-weights 0.5 --region <region>
"""

from __future__ import annotations

import argparse
import sys

from fray.cluster import ResourceConfig, get_tpu_topology
from marin.execution.executor import InputName, executor_main, output_path_of
from rigging.filesystem import marin_region

from experiments.downstream_scaling.evals.algorithms.joint_decode_avg_xregion import (
    JointDecodeCompletionAlgorithm,
    JointDecodeConfig,
    JointDecodeExecutionConfig,
    JointDecodeModelConfig,
    JointDecodeSamplingConfig,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.framework.xregion.pool import WorkerPoolConfig
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS
from experiments.models import llama_3_1_8b

N_SAMPLES = 32
N_PROBLEMS = 256
NUM_WORKERS = 1
CHUNK_SIZE = 64
TPU_TYPE = "v5p-8"

# Generous enough to absorb the first-step XLA compilation skew between the
# two engines for the largest delphi sizes (60s was too short for 1e22).
BARRIER_TIMEOUT_S = 1200.0

MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>")

NUM_FEWSHOT = 5
FEWSHOT_SEED = 1234

TEMPERATURE = 0.4
TOP_K_A = 16
TOP_K_B = 16
ADVISOR_WEIGHTS: tuple[float, ...] = (0.5,)
DELPHI_KEYS: tuple[str, ...] = ("3e18",)


def make_task() -> GSM8KTask:
    return GSM8KTask(
        config=GSM8KTaskConfig(
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
        )
    )


def make_worker_pool(tpu_type: str, region: str) -> WorkerPoolConfig:
    topology = get_tpu_topology(tpu_type)
    if topology.vm_count != 1:
        raise ValueError(f"joint decode xregion supports only single-VM TPU types, got {tpu_type}")
    if topology.chips_per_vm % 2 != 0:
        raise ValueError(f"joint decode xregion needs even chips_per_vm, got {tpu_type}")
    return WorkerPoolConfig(
        pool_id=f"{topology.vm_count}vm-{topology.chips_per_vm}chip",
        num_workers=NUM_WORKERS,
        worker_resources=ResourceConfig.with_tpu([tpu_type], regions=[region]),
        vm_count=topology.vm_count,
        chips_per_vm=topology.chips_per_vm,
    )


def make_algorithm(
    tpu_type: str,
    region: str,
    advisor_weight: float,
    advisor_model_path,
) -> JointDecodeCompletionAlgorithm:
    return JointDecodeCompletionAlgorithm(
        config=JointDecodeConfig(
            sampling=JointDecodeSamplingConfig(
                n_samples=N_SAMPLES,
                max_tokens=MAX_TOKENS,
                top_k_a=TOP_K_A,
                top_k_b=TOP_K_B,
                seed=SEED,
                temperature=TEMPERATURE,
                advisor_weight=advisor_weight,
                stop=STOP_TOKENS,
            ),
            advisor_model_path=advisor_model_path,
            decoder_model=JointDecodeModelConfig(apply_rpa_block_size_patch=True),
            advisor_model=JointDecodeModelConfig(),
            execution=JointDecodeExecutionConfig(
                worker_pools=(make_worker_pool(tpu_type, region),),
                chunk_size=CHUNK_SIZE,
                microbatch_size=None,
                barrier_timeout_s=BARRIER_TIMEOUT_S,
            ),
        )
    )


def build_steps(tpu_type: str, region: str, delphi_keys: list[str], advisor_weights: list[float]):
    advisor_model_path = output_path_of(llama_3_1_8b)
    unknown_keys = sorted(set(delphi_keys) - set(DELPHI_CHECKPOINTS))
    if unknown_keys:
        raise ValueError(f"Unknown delphi keys {unknown_keys}; known: {sorted(DELPHI_CHECKPOINTS)}")
    return [
        make_eval_step(
            name=(
                f"downstream_scaling/evals/delphi/gsm8k/joint_decode_gen0_reference/"
                f"advisor_weight{round(advisor_weight * 100):03d}/{slug}"
            ),
            model_path=InputName.hardcoded(DELPHI_CHECKPOINTS[slug]),
            task=make_task(),
            alg=make_algorithm(
                tpu_type,
                region,
                advisor_weight,
                advisor_model_path,
            ),
        )
        for advisor_weight in advisor_weights
        for slug in delphi_keys
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--tpu-type", default=TPU_TYPE)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--delphi-keys", nargs="+", default=list(DELPHI_KEYS))
    parser.add_argument("--advisor-weights", nargs="+", type=float, default=list(ADVISOR_WEIGHTS))
    args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_args]

    region = args.region or marin_region()
    if region is None:
        parser.error("--region not given and not inferable from the environment")

    executor_main(
        steps=build_steps(args.tpu_type, region, args.delphi_keys, args.advisor_weights),
        description="Generation-0 joint-decode reference evals on GSM8K (Phase 3 A/B gate).",
    )
