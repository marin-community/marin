# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.types import ResourceConfig

from experiments.defaults import default_raw_validation_sets
from experiments.evals.fineweb2_multilingual import fineweb2_multilingual_raw_validation_sets
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    model_perplexity_gap_from_scores,
    model_perplexity_scores,
)
from marin.execution.executor import executor_main

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
MAX_DOCS_PER_DATASET = 256
MAX_DOC_BYTES = 32_768

DATASETS = {
    **default_raw_validation_sets(),
    **fineweb2_multilingual_raw_validation_sets(),
}

MARIN_MODEL = GapFinderModelConfig(
    checkpoint_path="marin-community/marin-8b-base",
    checkpoint_is_hf=True,
    tokenizer="meta-llama/Llama-3.1-8B",
)

MARIN_SCORES = model_perplexity_scores(
    model=MARIN_MODEL,
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=4,
    max_eval_length=4096,
    max_docs_per_dataset=MAX_DOCS_PER_DATASET,
    max_doc_bytes=MAX_DOC_BYTES,
)
LLAMA_SCORES = model_perplexity_scores(
    model=GapFinderModelConfig(
        checkpoint_path="meta-llama/Llama-3.1-8B",
        checkpoint_is_hf=True,
        tokenizer="meta-llama/Llama-3.1-8B",
    ),
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=4,
    max_eval_length=4096,
    max_docs_per_dataset=MAX_DOCS_PER_DATASET,
    max_doc_bytes=MAX_DOC_BYTES,
)
QWEN3_SCORES = model_perplexity_scores(
    model=GapFinderModelConfig(
        checkpoint_path="Qwen/Qwen3-8B-Base",
        checkpoint_is_hf=True,
        tokenizer="Qwen/Qwen3-8B",
    ),
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=4,
    max_eval_length=4096,
    max_docs_per_dataset=MAX_DOCS_PER_DATASET,
    max_doc_bytes=MAX_DOC_BYTES,
)

MARIN_VS_LLAMA = model_perplexity_gap_from_scores(
    name="fineweb2-multilingual-marin-8b-base-vs-llama-3.1-8b-base-doccap256",
    model_a_name="marin-community/marin-8b-base",
    model_b_name="meta-llama/Llama-3.1-8B",
    model_a_scores_path=MARIN_SCORES.as_input_name(),
    model_b_scores_path=LLAMA_SCORES.as_input_name(),
    wandb_tags=[
        "eval=perplexity-gap",
        "rerun=fineweb2-multilingual",
        "model_a=marin-community/marin-8b-base",
        "model_b=meta-llama/Llama-3.1-8B",
        "dataset_bundle=default_raw_plus_fineweb2_multilingual",
        "region=us-central1",
        f"max_docs_per_dataset={MAX_DOCS_PER_DATASET}",
    ],
)

MARIN_VS_QWEN3 = model_perplexity_gap_from_scores(
    name="fineweb2-multilingual-marin-8b-base-vs-qwen3-8b-base-doccap256",
    model_a_name="marin-community/marin-8b-base",
    model_b_name="Qwen/Qwen3-8B-Base",
    model_a_scores_path=MARIN_SCORES.as_input_name(),
    model_b_scores_path=QWEN3_SCORES.as_input_name(),
    wandb_tags=[
        "eval=perplexity-gap",
        "rerun=fineweb2-multilingual",
        "model_a=marin-community/marin-8b-base",
        "model_b=Qwen/Qwen3-8B-Base",
        "dataset_bundle=default_raw_plus_fineweb2_multilingual",
        "region=us-central1",
        f"max_docs_per_dataset={MAX_DOCS_PER_DATASET}",
    ],
)


if __name__ == "__main__":
    executor_main(
        [MARIN_VS_LLAMA, MARIN_VS_QWEN3],
        description="Run Marin perplexity-gap reports with FineWeb2 multilingual held-out eval sets.",
    )
