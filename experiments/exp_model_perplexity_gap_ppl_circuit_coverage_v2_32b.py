# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin 32B vs Qwen3 32B gap run for PPL circuit coverage v2 probes."""

from __future__ import annotations

from fray.types import ResourceConfig
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    model_perplexity_gap_from_scores,
    model_perplexity_scores,
)
from marin.execution.executor import ExecutorStep, executor_main

from experiments.evals.ppl_circuit_coverage_v2 import (
    PPL_CIRCUIT_COVERAGE_V2_ISSUE,
    ppl_circuit_coverage_v2_raw_validation_sets,
)
from experiments.marin_models import marin_tokenizer

RUN_KEY = "main_gap_32b_ppl_circuit_coverage_v2_issue6070_v3"
RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
MAX_DOCS_PER_DATASET = None
MAX_DOC_BYTES = 32_768
DATASET_BUNDLE = "ppl_circuit_coverage_v2"

DATASETS = ppl_circuit_coverage_v2_raw_validation_sets()

MARIN_MODEL = GapFinderModelConfig(
    checkpoint_path="marin-community/marin-32b-base",
    checkpoint_is_hf=True,
    tokenizer=marin_tokenizer,
)
QWEN3_MODEL = GapFinderModelConfig(
    checkpoint_path="Qwen/Qwen3-32B",
    checkpoint_is_hf=True,
    tokenizer="Qwen/Qwen3-32B",
)

MARIN_SCORES = model_perplexity_scores(
    name=f"{RUN_KEY}/marin_32b",
    model=MARIN_MODEL,
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=1,
    max_eval_length=4096,
    max_docs_per_dataset=MAX_DOCS_PER_DATASET,
    max_doc_bytes=MAX_DOC_BYTES,
    wandb_tags=[
        "eval=model-perplexity",
        f"dataset_bundle={DATASET_BUNDLE}",
        "model=marin-community/marin-32b-base",
        "template=compact_v2",
        "region=us-central1",
        f"issue:{PPL_CIRCUIT_COVERAGE_V2_ISSUE}",
    ],
)

QWEN3_SCORES = model_perplexity_scores(
    name=f"{RUN_KEY}/qwen3_32b",
    model=QWEN3_MODEL,
    datasets=DATASETS,
    resource_config=RESOURCE_CONFIG,
    per_device_batch_size=1,
    max_eval_length=4096,
    max_docs_per_dataset=MAX_DOCS_PER_DATASET,
    max_doc_bytes=MAX_DOC_BYTES,
    wandb_tags=[
        "eval=model-perplexity",
        f"dataset_bundle={DATASET_BUNDLE}",
        "model=Qwen/Qwen3-32B",
        "template=compact_v2",
        "region=us-central1",
        f"issue:{PPL_CIRCUIT_COVERAGE_V2_ISSUE}",
    ],
)

GAP = model_perplexity_gap_from_scores(
    name=f"{RUN_KEY}/marin_32b-vs-qwen3_32b",
    model_a_name="marin-community/marin-32b-base",
    model_b_name="Qwen/Qwen3-32B",
    model_a_scores_path=MARIN_SCORES.as_input_name(),
    model_b_scores_path=QWEN3_SCORES.as_input_name(),
    wandb_tags=[
        "eval=perplexity-gap",
        f"dataset_bundle={DATASET_BUNDLE}",
        "model_a=marin-community/marin-32b-base",
        "model_b=Qwen/Qwen3-32B",
        "template=compact_v2",
        "region=us-central1",
        f"issue:{PPL_CIRCUIT_COVERAGE_V2_ISSUE}",
    ],
)

STEPS: list[ExecutorStep] = [GAP]


if __name__ == "__main__":
    executor_main(
        STEPS,
        description="Run Marin 32B vs Qwen3 32B perplexity gap on PPL circuit coverage v2 slices.",
    )
