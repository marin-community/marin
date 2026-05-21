# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: SFT 3e18 for 10 steps, then eval on 8 GSM8K problems.

Purpose is to verify wiring (transform → tokenize → SFT → HF export → prompts
→ completions → grades), not measure quality. Expected GSM8K accuracy after
10 SFT steps is near 0%.

Required env before submit:

    export MARIN_PREFIX=gs://marin-us-east5
"""

from __future__ import annotations

import argparse
import sys

from fray.cluster import ResourceConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize import lm_data_config

from experiments.defaults import default_sft, default_tokenize
from experiments.downstream_scaling.evals.framework.core import make_eval_step
from experiments.downstream_scaling.evals.tasks.gsm8k_qa import GSM8KQATask, GSM8KQATaskConfig
from experiments.downstream_scaling.models.delphi import DELPHI_CHECKPOINTS
from experiments.downstream_scaling.run_delphi_sft_gsm8k_qa import make_iid_alg
from experiments.downstream_scaling.sft.data.gsm8k_qa import build_gsm8k_qa_transform_step
from experiments.downstream_scaling.sft.tokenize import GSM8K_QA_CHAT_FORMAT
from experiments.downstream_scaling.sft.train import _resolve_latest_hf_checkpoint
from experiments.llama import llama3_tokenizer
from experiments.simple_sft_config import SimpleSFTConfig

SMOKE_SLUG = "3e18"


def build_smoke_steps(eval_tpu: str = "v5p-8"):
    transform_step = build_gsm8k_qa_transform_step()
    tokenize_step = default_tokenize(
        name="downstream_scaling/sft/data/smoke_gsm8k_qa_llama3",
        dataset=transform_step,
        tokenizer=llama3_tokenizer,
        format=GSM8K_QA_CHAT_FORMAT,
    )

    rel_path = DELPHI_CHECKPOINTS[SMOKE_SLUG]
    base_checkpoint = _resolve_latest_hf_checkpoint(rel_path)
    model_config = HFCheckpointConverter.from_hf(base_checkpoint).config_from_hf_checkpoint(base_checkpoint)

    smoke_sft_config = SimpleSFTConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=16,
        max_seq_len=1024,
        num_train_steps=10,
        learning_rate=5e-6,
        warmup=0.0,
        weight_decay=0.0,
        lr_schedule="linear",
        decay=0.0,
        steps_per_hf_export=10,
        steps_per_eval=10,
        steps_per_checkpoint=None,
        pad_tokenizer_to_match_model=True,
        seed=0,
        initialize_from_hf=base_checkpoint,
    )

    sft_step = default_sft(
        name=f"downstream_scaling/sft/smoke/delphi/gsm8k_qa/{SMOKE_SLUG}",
        tokenized=lm_data_config(
            training_set=tokenize_step,
            num_validation_sequences={"smoke_gsm8k_qa_llama3": 32},
        ),
        model_config=model_config,
        sft_config=smoke_sft_config,
        tags=["sft", "smoke", "downstream_scaling", "gsm8k_qa", SMOKE_SLUG],
    )

    eval_step = make_eval_step(
        name=f"downstream_scaling/evals/smoke_sft/gsm8k_qa/iid/{SMOKE_SLUG}",
        model_path=output_path_of(sft_step) / "hf",
        task=GSM8KQATask(config=GSM8KQATaskConfig(n_problems=8, grade_workers=2)),
        alg=make_iid_alg(eval_tpu),
    )

    return [sft_step, eval_step]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--eval-tpu", type=str, default="v5p-8")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    executor_main(
        steps=build_smoke_steps(args.eval_tpu),
        description=f"Smoke: SFT Delphi {SMOKE_SLUG} for 10 steps on GSM8K Q+A, then 8-problem eval.",
    )
