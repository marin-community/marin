# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.v2.types import ResourceConfig

from experiments.evals.long_tail_ppl_runnable import runnable_long_tail_raw_validation_sets
from marin.evaluation.perplexity_gap import GapFinderModelConfig, default_model_perplexity_gap
from marin.execution.executor import executor_main

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
MAX_DOCS_PER_DATASET = 32
MAX_DOC_BYTES = 32_768

DATASETS = runnable_long_tail_raw_validation_sets()

MARIN_MODEL = GapFinderModelConfig(
    checkpoint_path="marin-community/marin-8b-base",
    checkpoint_is_hf=True,
    tokenizer="meta-llama/Llama-3.1-8B",
)

STEP = default_model_perplexity_gap(
    name="long-tail-smoke-marin-8b-base-vs-llama-3.1-8b-base-doccap32",
    model_a=MARIN_MODEL,
    model_b=GapFinderModelConfig(
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
    wandb_tags=[
        "eval=perplexity-gap",
        "smoke=long-tail-ppl",
        "source_split=hf_dataset",
        "region=us-central1",
        "dataset_bundle=runnable_long_tail_hf_backed",
        f"max_docs_per_dataset={MAX_DOCS_PER_DATASET}",
    ],
)


if __name__ == "__main__":
    executor_main([STEP], description="Smoke-run runnable long-tail PPL slices from public Hugging Face datasets.")
