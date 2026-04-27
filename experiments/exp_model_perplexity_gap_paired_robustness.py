# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run opt-in paired robustness perplexity-gap reports for issue #5096."""

from fray import ResourceConfig

from experiments.defaults import default_raw_validation_sets
from experiments.evals.paired_robustness_ppl import paired_robustness_raw_steps, paired_robustness_raw_validation_sets
from marin.evaluation.perplexity_gap import GapFinderModelConfig, default_model_perplexity_gap
from marin.execution.executor import executor_main

RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
MAX_DOCS_PER_DATASET = 256
MAX_DOC_BYTES = 32_768

PAIRED_ROBUSTNESS_RAW_STEPS = paired_robustness_raw_steps()

DATASETS = {
    **default_raw_validation_sets(),
    **paired_robustness_raw_validation_sets(raw_steps=PAIRED_ROBUSTNESS_RAW_STEPS),
}

MARIN_MODEL = GapFinderModelConfig(
    checkpoint_path="marin-community/marin-8b-base",
    checkpoint_is_hf=True,
    tokenizer="meta-llama/Llama-3.1-8B",
)

MARIN_VS_LLAMA = default_model_perplexity_gap(
    name="paired-robustness-marin-8b-base-vs-llama-3.1-8b-base-doccap256",
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
        "rerun=paired-robustness",
        "model_a=marin-community/marin-8b-base",
        "model_b=meta-llama/Llama-3.1-8B",
        "dataset_bundle=default_raw_plus_paired_robustness",
        "region=us-central1",
        f"max_docs_per_dataset={MAX_DOCS_PER_DATASET}",
    ],
)

MARIN_VS_QWEN3 = default_model_perplexity_gap(
    name="paired-robustness-marin-8b-base-vs-qwen3-8b-base-doccap256",
    model_a=MARIN_MODEL,
    model_b=GapFinderModelConfig(
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
    wandb_tags=[
        "eval=perplexity-gap",
        "rerun=paired-robustness",
        "model_a=marin-community/marin-8b-base",
        "model_b=Qwen/Qwen3-8B-Base",
        "dataset_bundle=default_raw_plus_paired_robustness",
        "region=us-central1",
        f"max_docs_per_dataset={MAX_DOCS_PER_DATASET}",
    ],
)


if __name__ == "__main__":
    executor_main(
        [*PAIRED_ROBUSTNESS_RAW_STEPS.values(), MARIN_VS_LLAMA, MARIN_VS_QWEN3],
        description="Run Marin perplexity-gap reports with opt-in paired paraphrase and translation robustness slices.",
    )
