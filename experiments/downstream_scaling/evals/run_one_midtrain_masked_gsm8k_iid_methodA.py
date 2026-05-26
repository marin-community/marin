# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Method-A single-cell launcher: existing iid.py + shared GCS XLA compile cache.

Identical to `run_one_midtrain_masked_gsm8k_iid.py` except the eval-step name
writes under a distinct prefix (`masked_gsm8k_mask00_methodA_cache/`). Used in
the 2026-05-26 A/B engineering test on the 1e22 row (logbook section
"2026-05-26 (later) — A/B comparison on the 1e22 row").

The compile-cache env vars (`JAX_COMPILATION_CACHE_DIR`, `VLLM_XLA_CACHE_PATH`)
are set in `VLLM_TPU_ENV_VARS` inside `evals/algorithms/iid.py`, so they apply
globally to any IID load through that module. This launcher exists to namespace
the output paths so we don't conflate A's outputs with the original (pre-cache)
matrix-eval prefix.
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
        name=f"downstream_scaling/evals/delphi_midtrain/masked_gsm8k_mask00_methodA_cache/{slug}",
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
        description=f"Method-A midtrain matrix mask_00 eval: {args.slug}.",
    )
