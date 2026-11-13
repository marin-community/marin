# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate every self-distilled curriculum checkpoint on one shared transfer benchmark set.

A gain that appears on benchmarks unrelated to a checkpoint's curriculum comes from the SFT itself,
for example reliably entering and closing thinking mode, rather than from the curriculum's subject.
The trained checkpoints are reused from their trials, so this only runs evaluations:

    uv run iris --cluster=cw-rno2a job run --no-wait --job-name curriculum-cross-eval-<date> \\
      -e MARIN_PREFIX s3://marin-us-east-02a/tmp/ttl=7d/curriculum-math-20260924 \\
      -- uv run python experiments/post_training/curriculum_sft/cross_eval.py --version <version> --run
"""

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.post_training.curriculum_sft.code_trial import build_code_trial
from experiments.post_training.curriculum_sft.finance_trial import build_finance_trial
from experiments.post_training.curriculum_sft.instruction_following_trial import build_instruction_following_trial
from experiments.post_training.curriculum_sft.math_trial import build_math_trial
from experiments.post_training.curriculum_sft.science_trial import build_science_trial
from experiments.post_training.curriculum_sft.trial import trained_eval

TRANSFER_EVALS = "math500,humanevalplus,mbppplus"
# The trial versions, learning rate, and warmup that produced each self-distilled checkpoint.
TRIAL_VERSION = "2026.09.26.1"
MATH_TRIAL_VERSION = "2026.09.26.6"
LEARNING_RATE = 1e-5
WARMUP = 1


def trained_checkpoints() -> dict[str, ArtifactStep]:
    return {
        "math": build_math_trial(MATH_TRIAL_VERSION, LEARNING_RATE, WARMUP, data="self")["train"],
        "code": build_code_trial(TRIAL_VERSION, LEARNING_RATE, WARMUP)["train"],
        "finance": build_finance_trial(TRIAL_VERSION, LEARNING_RATE, WARMUP)["train"],
        "science": build_science_trial(TRIAL_VERSION, LEARNING_RATE, WARMUP)["train"],
        "if": build_instruction_following_trial(TRIAL_VERSION, LEARNING_RATE, WARMUP)["train"],
    }


@click.command()
@build_options
def main() -> dict[str, ArtifactStep]:
    version = resolve_version("curriculum-transfer-sep20", None)
    return {
        key: trained_eval(train, eval_name=f"curriculum-transfer-{key}", evals=TRANSFER_EVALS, version=version)
        for key, train in trained_checkpoints().items()
    }


if __name__ == "__main__":
    main()
