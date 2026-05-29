# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw perplexity-gap evals for tokenizer sensitivity d1024 runs.

Uses the finished d1024 tokenizer-sensitivity checkpoints from #5821 and
compares each tokenizer arm against the llama3-128k arm. The unfinished
marin-128k arm is intentionally excluded.
"""

from __future__ import annotations

from dataclasses import replace

from fray.types import ResourceConfig
from levanter.tokenizers import TokenizerBackend

from experiments.defaults import default_raw_validation_sets
from experiments.grug.moe.model import GrugModelConfig
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    model_perplexity_gap_from_scores,
    model_perplexity_scores,
)
from marin.execution.executor import executor_main


RESOURCE_CONFIG = ResourceConfig.with_tpu(["v6e-8", "v5litepod-8"], regions=["europe-west4"])
DATASETS = default_raw_validation_sets()

BASE_CHECKPOINT_DIR = "gs://marin-eu-west4/data/datakit/train/tokenizer-sensitivity-moe-ladder/2026-05-25/d1024"

BASE_MODEL = GrugModelConfig(
    hidden_dim=1024,
    intermediate_dim=512,
    shared_expert_intermediate_dim=1024,
    num_layers=11,
    num_heads=8,
    num_kv_heads=2,
    num_experts=64,
    num_experts_per_token=4,
    max_seq_len=4096,
    sliding_window=4096,
    qk_mult=1.3,
    initializer_std=0.015625,
    layer_norm_eps=1e-5,
    router_z_loss_coef=0.001,
    vocab_size=128_256,
)


def native_model(
    *,
    arm: str,
    tokenizer: str,
    vocab_size: int,
    tokenizer_backend: TokenizerBackend = TokenizerBackend.HF,
) -> GapFinderModelConfig:
    return GapFinderModelConfig(
        checkpoint_path=f"{BASE_CHECKPOINT_DIR}/{arm}/checkpoints",
        checkpoint_is_hf=False,
        model=replace(BASE_MODEL, vocab_size=vocab_size),
        tokenizer=tokenizer,
        tokenizer_backend=tokenizer_backend,
    )


MODELS = {
    "llama3-128k": native_model(
        arm="llama3-128k",
        tokenizer="meta-llama/Meta-Llama-3.1-8B",
        vocab_size=128_256,
    ),
    "qwen3-152k": native_model(
        arm="qwen3-152k",
        tokenizer="Qwen/Qwen3-0.6B",
        vocab_size=151_936,
    ),
    "gemma3-262k": native_model(
        arm="gemma3-262k",
        tokenizer="google/gemma-3-4b-it",
        vocab_size=262_145,
    ),
    "gpt-oss-200k": native_model(
        arm="gpt-oss-200k",
        tokenizer="openai/gpt-oss-20b",
        vocab_size=200_019,
    ),
    "tokenmonster-englishcode-32k": native_model(
        arm="tokenmonster-englishcode-32k",
        tokenizer="tokenmonster:englishcode-32000-consistent-v1",
        tokenizer_backend=TokenizerBackend.TOKENMONSTER,
        vocab_size=32_000,
    ),
}

SCORE_STEPS = {
    arm: model_perplexity_scores(
        name=f"tokenizer-sensitivity-d1024/{arm}",
        model=model,
        datasets=DATASETS,
        resource_config=RESOURCE_CONFIG,
        per_device_batch_size=16,
        max_eval_length=4096,
        max_docs_per_dataset=256,
        max_doc_bytes=32_768,
        wandb_tags=[
            "eval=model-perplexity",
            "issue=5821",
            "tokenizer-sensitivity",
            "d1024",
            f"model={arm}",
            "baseline=llama3-128k" if arm == "llama3-128k" else "comparison=llama3-128k",
            "region=europe-west4",
        ],
    )
    for arm, model in MODELS.items()
}

BASELINE = "llama3-128k"
GAP_STEPS = [
    model_perplexity_gap_from_scores(
        name=f"tokenizer-sensitivity-d1024/{arm}-vs-{BASELINE}",
        model_a_name=arm,
        model_b_name=BASELINE,
        model_a_scores_path=SCORE_STEPS[arm].as_input_name(),
        model_b_scores_path=SCORE_STEPS[BASELINE].as_input_name(),
        wandb_tags=[
            "eval=perplexity-gap",
            "issue=5821",
            "tokenizer-sensitivity",
            "d1024",
            f"model_a={arm}",
            f"model_b={BASELINE}",
            "region=europe-west4",
        ],
    )
    for arm in MODELS
    if arm != BASELINE
]


if __name__ == "__main__":
    executor_main(
        GAP_STEPS,
        description=(
            "Compare finished tokenizer-sensitivity d1024 arms against the llama3-128k baseline on raw "
            "Paloma and uncheatable eval datasets. The unfinished marin-128k arm is excluded."
        ),
    )
