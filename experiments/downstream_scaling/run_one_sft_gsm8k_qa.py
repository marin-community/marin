# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-cell SFT launcher: pick one base checkpoint via --slug, run SFT + eval.

Usage:

    uv run iris --cluster=marin job run --no-wait \\
        --job-name aa-sft-1e20-iso \\
        --zone us-east5-a \\
        --tpu v5p-8 --enable-extra-resources \\
        --cpu 32 --memory 128GB --disk 50GB \\
        --extra tpu \\
        --max-retries 2 \\
        -e MARIN_PREFIX gs://marin-us-east5 \\
        -- python experiments/downstream_scaling/run_one_sft_gsm8k_qa.py --slug 1e20_iso

Valid slugs:
  - Base ladder (from `models/delphi.py`, minus 1e23): 3e18, 9e18, 2e19, 3e19,
    9e19, 2e20, 3e20, 1e21, 1e22.
  - 1e20 stand-in base (from `models/delphi_extra.py`): 1e20_iso.
  - Midtrain variants (from `models/midtrain.py`): 1e20_p33m67_lr0.67,
    1e20_p67m33_lr0.33, 1e21_p33m67_lr0.67, 1e21_p67m33_lr0.33,
    1e22_p33m67_lr0.67, 1e22_p67m33_lr0.33.

This avoids `run_delphi_sft_gsm8k_qa.py`'s 16-cell plan-time GCS scan when you
only want one cell. Until SFT-as-child-Iris-job dispatch is fixed (cf. logbook
'Open framework question'), each cell needs its own coordinator with TPU
resources, so submitting cells in parallel = submitting separate iris jobs.
"""

from __future__ import annotations

import argparse
import sys

from marin.execution.executor import executor_main, output_path_of

from experiments.defaults import default_tokenize
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.tasks.gsm8k_qa import GSM8KQATask, GSM8KQATaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS
from experiments.downstream_scaling.models.delphi_extra import DELPHI_EXTRA_BASE_CHECKPOINTS
from experiments.downstream_scaling.models.midtrain import DELPHI_MIDTRAIN_CHECKPOINTS
from experiments.downstream_scaling.run_delphi_sft_gsm8k_qa import make_iid_alg
from experiments.downstream_scaling.sft.data.gsm8k_qa import build_gsm8k_qa_transform_step
from experiments.downstream_scaling.sft.tokenize import GSM8K_QA_CHAT_FORMAT
from experiments.downstream_scaling.sft.train import SFT_OUTPUT_PREFIX, build_sft_step
from experiments.llama import llama3_tokenizer


def _checkpoint_registry() -> dict[str, str]:
    base_without_1e23 = {k: v for k, v in DELPHI_CHECKPOINTS.items() if k != "1e23"}
    return {**base_without_1e23, **DELPHI_EXTRA_BASE_CHECKPOINTS, **DELPHI_MIDTRAIN_CHECKPOINTS}


def build_steps(slug: str, eval_tpu: str = "v5p-8"):
    registry = _checkpoint_registry()
    if slug not in registry:
        raise ValueError(f"Unknown slug {slug!r}. Available: {sorted(registry)}")
    rel_path = registry[slug]

    transform_step = build_gsm8k_qa_transform_step()
    tokenize_step = default_tokenize(
        name="downstream_scaling/sft/data/gsm8k_qa_llama3",
        dataset=transform_step,
        tokenizer=llama3_tokenizer,
        format=GSM8K_QA_CHAT_FORMAT,
    )

    sft_step = build_sft_step(slug, rel_path, tokenize_step)
    # Derive the eval-output namespace from the SFT variant so changing
    # SFT_OUTPUT_PREFIX (e.g. gsm8k_qa -> gsm8k_qa_nopack_1ep) forces fresh
    # eval steps. Without this the eval hash is stable across SFT-config
    # changes that aren't `versioned()`, and the executor cache-hits the
    # previous grades — see the 19-epoch -> 1-epoch cache miss that landed us
    # here.
    sft_variant = SFT_OUTPUT_PREFIX.rstrip("/").split("/")[-1]  # e.g. "gsm8k_qa_nopack_1ep"
    eval_step = make_eval_step(
        name=f"downstream_scaling/evals/delphi_sft/{sft_variant}/iid/{slug}",
        model_path=output_path_of(sft_step) / "hf",
        task=GSM8KQATask(config=GSM8KQATaskConfig(n_problems=256)),
        alg=make_iid_alg(eval_tpu),
    )
    return [sft_step, eval_step]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--slug", type=str, required=True)
    parser.add_argument("--eval-tpu", type=str, default="v5p-8")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    executor_main(
        steps=build_steps(slug=args.slug, eval_tpu=args.eval_tpu),
        description=f"Single-cell SFT on GSM8K Q+A: {args.slug}.",
    )
