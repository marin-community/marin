# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.types import ResourceConfig
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    model_perplexity_gap_from_scores,
    model_perplexity_scores,
)
from marin.execution.executor import executor_main

from experiments.defaults import default_raw_validation_sets

RESOURCE_CONFIG = ResourceConfig.with_tpu(
    "v5p-8",
    regions=["us-central1"],
)
DATASETS = default_raw_validation_sets()

MARIN_MODEL = GapFinderModelConfig(
    checkpoint_path="marin-community/marin-8b-base",
    checkpoint_is_hf=True,
    tokenizer="meta-llama/Llama-3.1-8B",
)
LLAMA_MODEL = GapFinderModelConfig(
    checkpoint_path="meta-llama/Llama-3.1-8B",
    checkpoint_is_hf=True,
    tokenizer="meta-llama/Llama-3.1-8B",
)

MARIN_SCORES = model_perplexity_scores(
    model=MARIN_MODEL,
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=4,
    max_eval_length=4096,
    max_docs_per_dataset=256,
    max_doc_bytes=32_768,
    wandb_tags=["model=marin-community/marin-8b-base"],
)
LLAMA_SCORES = model_perplexity_scores(
    model=LLAMA_MODEL,
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=4,
    max_eval_length=4096,
    max_docs_per_dataset=256,
    max_doc_bytes=32_768,
    wandb_tags=["model=meta-llama/Llama-3.1-8B"],
)

STEP = model_perplexity_gap_from_scores(
    name="marin-8b-base-vs-llama-3.1-8b-base",
    model_a_name="marin-community/marin-8b-base",
    model_b_name="meta-llama/Llama-3.1-8B",
    model_a_scores_path=MARIN_SCORES.as_input_name(),
    model_b_scores_path=LLAMA_SCORES.as_input_name(),
    wandb_tags=[
        "eval=perplexity-gap",
        "model_a=marin-community/marin-8b-base",
        "model_b=meta-llama/Llama-3.1-8B",
        "region=us-central1",
    ],
)


if __name__ == "__main__":
    executor_main(
        [STEP],
        description="Compare Marin 8B base and Llama 3.1 8B base on raw Paloma and uncheatable eval datasets.",
    )
