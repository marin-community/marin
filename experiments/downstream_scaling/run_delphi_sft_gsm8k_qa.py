# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launcher: SFT the Delphi ladder + midtrained variants on GSM8K Q+A, then eval.

For each base checkpoint (Delphi 3e18→1e22, the 1e20_iso stand-in, and 6
midtrained variants from issue #4547), this launcher builds an SFT step
initialized from the latest HF checkpoint and a downstream-scaling eval step
that vLLM-samples the final SFT checkpoint on zero-shot GSM8K Q+A.

Required env before submit:

    export MARIN_PREFIX=gs://marin-us-east5

The launcher reads `MARIN_PREFIX` to resolve base-checkpoint paths and derive
each model's Levanter config at plan time.
"""

from __future__ import annotations

import argparse
import sys

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, output_path_of

from experiments.defaults import default_tokenize
from experiments.downstream_scaling.evals.algorithms.iid import (
    IIDCompletionAlgorithm,
    IIDConfig,
    IIDExecutionConfig,
    IIDSamplingConfig,
)
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.tasks.gsm8k_qa import GSM8KQATask, GSM8KQATaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS
from experiments.downstream_scaling.models.delphi_extra import DELPHI_EXTRA_BASE_CHECKPOINTS
from experiments.downstream_scaling.models.midtrain import DELPHI_MIDTRAIN_CHECKPOINTS
from experiments.downstream_scaling.sft.data.gsm8k_qa import build_gsm8k_qa_transform_step
from experiments.downstream_scaling.sft.tokenize import GSM8K_QA_CHAT_FORMAT
from experiments.downstream_scaling.sft.train import build_sft_step
from experiments.llama import llama3_tokenizer

EVAL_TPU = "v5p-8"
N_SAMPLES = 32
N_PROBLEMS = 256
TEMPERATURE = 0.6
TOP_P = 1.0
TOP_K = 1000
MAX_TOKENS = 512
SEED = 42
STOP_TOKENS = ("Question:", "</s>", "<|im_end|>", "<|end_of_text|>")


def make_iid_alg(tpu_type: str) -> IIDCompletionAlgorithm:
    return IIDCompletionAlgorithm(
        config=IIDConfig(
            sampling=IIDSamplingConfig(
                n_samples=N_SAMPLES,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_TOKENS,
                seed=SEED,
                stop=STOP_TOKENS,
            ),
            execution=IIDExecutionConfig(
                num_workers=1,
                worker_resources=ResourceConfig.with_tpu(tpu_type),
            ),
        )
    )


def _checkpoint_registry() -> dict[str, str]:
    """Union of base ladder (minus 1e23), 1e20_iso, and midtrain variants."""
    base_without_1e23 = {k: v for k, v in DELPHI_CHECKPOINTS.items() if k != "1e23"}
    overlap = set(base_without_1e23) & set(DELPHI_EXTRA_BASE_CHECKPOINTS) | set(base_without_1e23) & set(
        DELPHI_MIDTRAIN_CHECKPOINTS
    )
    if overlap:
        raise ValueError(f"slug collision between base and extra/midtrain registries: {overlap}")
    return {**base_without_1e23, **DELPHI_EXTRA_BASE_CHECKPOINTS, **DELPHI_MIDTRAIN_CHECKPOINTS}


def build_steps(eval_tpu: str = EVAL_TPU):
    transform_step = build_gsm8k_qa_transform_step()
    tokenize_step = default_tokenize(
        name="downstream_scaling/sft/data/gsm8k_qa_llama3",
        dataset=transform_step,
        tokenizer=llama3_tokenizer,
        format=GSM8K_QA_CHAT_FORMAT,
    )
    eval_task = GSM8KQATask(config=GSM8KQATaskConfig(n_problems=N_PROBLEMS))
    eval_alg = make_iid_alg(eval_tpu)

    steps = []
    for slug, rel_path in _checkpoint_registry().items():
        sft_step = build_sft_step(slug, rel_path, tokenize_step)
        eval_step = make_eval_step(
            name=f"downstream_scaling/evals/delphi_sft/gsm8k_qa/iid/{slug}",
            model_path=output_path_of(sft_step) / "hf",
            task=eval_task,
            alg=eval_alg,
        )
        steps.extend([sft_step, eval_step])
    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-tpu", type=str, default=EVAL_TPU)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    executor_main(
        steps=build_steps(args.eval_tpu),
        description="Delphi ladder + midtrain variants: SFT on GSM8K Q+A, then zero-shot Q+A eval.",
    )
