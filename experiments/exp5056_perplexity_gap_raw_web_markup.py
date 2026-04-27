# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pilot perplexity-gap run for the #5056 raw web/markup slices."""

from fray.types import ResourceConfig

from experiments.exp5056_raw_web_markup_ppl import raw_web_markup_raw_validation_sets
from marin.evaluation.perplexity_gap import GapFinderModelConfig, default_model_perplexity_gap
from marin.execution.executor import executor_main

STEP = default_model_perplexity_gap(
    name="raw-web-markup-marin-8b-base-vs-llama-3.1-8b-base",
    model_a=GapFinderModelConfig(
        checkpoint_path="marin-community/marin-8b-base",
        checkpoint_is_hf=True,
        tokenizer="meta-llama/Llama-3.1-8B",
    ),
    model_b=GapFinderModelConfig(
        checkpoint_path="meta-llama/Llama-3.1-8B",
        checkpoint_is_hf=True,
        tokenizer="meta-llama/Llama-3.1-8B",
    ),
    datasets=raw_web_markup_raw_validation_sets(),
    resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"]),
    per_device_batch_size=4,
    max_eval_length=4096,
    max_docs_per_dataset=256,
    max_doc_bytes=32_768,
    wandb_tags=[
        "eval=perplexity-gap",
        "issue=5056",
        "dataset_family=raw_web_markup",
        "model_a=marin-community/marin-8b-base",
        "model_b=meta-llama/Llama-3.1-8B",
        "region=us-central1",
    ],
)


if __name__ == "__main__":
    executor_main(
        [STEP],
        description="Compare Marin 8B base and Llama 3.1 8B base on #5056 raw web/markup/image-text slices.",
    )
