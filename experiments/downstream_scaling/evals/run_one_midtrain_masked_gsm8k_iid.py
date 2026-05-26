# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-cell midtrain-matrix eval launcher.

Pick one midtrain checkpoint via `--slug`, run `MaskedGSM8KTask(mask_fraction=
0.0, num_fewshot=5)` IID eval. No SFT, no retraining — pure vLLM inference.

Usage:

    uv run iris --cluster=marin job run --no-wait \\
        --job-name aa-mtx-3e20-p33m67-lr0p33 \\
        --zone us-east5-a \\
        --tpu v5p-8 --enable-extra-resources \\
        --cpu 4 --memory 32GB --disk 50GB \\
        --extra tpu \\
        --max-retries 2 \\
        --priority interactive \\
        -e MARIN_PREFIX gs://marin-us-east5 \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python experiments/downstream_scaling/evals/run_one_midtrain_masked_gsm8k_iid.py --slug 3e20_p33m67_lr0.33

Valid slugs are in `DELPHI_MIDTRAIN_MATRIX` (36 total): `{scale}_{mix}_lr{X.YY}`
where scale ∈ {3e20, 1e21, 1e22}, mix ∈ {p33m67, p50m50, p67m33}, lr ∈ {0.33,
0.5, 0.67, 0.83}.
"""

from __future__ import annotations

import argparse
import sys

from marin.execution.executor import InputName, executor_main

from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.run_delphi_masked_gsm8k_iid import (
    FEWSHOT_SEED,
    MASK_TEXT,
    N_PROBLEMS,
    NUM_FEWSHOT,
    make_algorithm,
)
from experiments.downstream_scaling.evals.tasks.gsm8k_masked import MaskedGSM8KTask, MaskedGSM8KTaskConfig
from experiments.downstream_scaling.models.midtrain import DELPHI_MIDTRAIN_MATRIX
from experiments.llama import llama3_tokenizer

# Pinned to mask_00 — the matrix axis is (scale, mix, lr), not mask_fraction.
MASK_FRACTION = 0.0


def build_steps(slug: str, eval_tpu: str = "v5p-8"):
    if slug not in DELPHI_MIDTRAIN_MATRIX:
        raise ValueError(f"Unknown slug {slug!r}. Available: {sorted(DELPHI_MIDTRAIN_MATRIX)}")
    checkpoint = DELPHI_MIDTRAIN_MATRIX[slug]
    task = MaskedGSM8KTask(
        config=MaskedGSM8KTaskConfig(
            tokenizer_path=llama3_tokenizer,
            num_fewshot=NUM_FEWSHOT,
            fewshot_seed=FEWSHOT_SEED,
            n_problems=N_PROBLEMS,
            mask_fraction=MASK_FRACTION,
            mask_text=MASK_TEXT,
        )
    )
    eval_step = make_eval_step(
        name=f"downstream_scaling/evals/delphi_midtrain/masked_gsm8k_mask00/{slug}",
        model_path=InputName.hardcoded(checkpoint),
        task=task,
        alg=make_algorithm(eval_tpu),
    )
    return [eval_step]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--slug", type=str, required=True)
    parser.add_argument("--eval-tpu", type=str, default="v5p-8")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    executor_main(
        steps=build_steps(slug=args.slug, eval_tpu=args.eval_tpu),
        description=f"Midtrain matrix mask_00 eval: {args.slug}.",
    )
