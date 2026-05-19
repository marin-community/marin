# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Marin 32B vs Qwen3 32B on every committed raw PPL provider.

This is the durable all-available launcher for the perplexity-gap dashboard
workflow. It intentionally composes providers already committed on ``main`` and
keeps run-local skips explicit so construction can be audited without launching
Iris jobs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from fray.types import ResourceConfig
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    RawTextEvaluationDataset,
    model_perplexity_gap_from_scores,
    model_perplexity_scores,
)
from marin.execution.executor import ExecutorStep, executor_main

from experiments.bio_chem_notation import bio_chem_raw_validation_sets
from experiments.defaults import default_raw_validation_sets
from experiments.evals.asr_ocr_noisy_ppl import noisy_asr_ocr_raw_validation_sets
from experiments.evals.diagnostic_logs import diagnostic_log_raw_validation_sets
from experiments.evals.exp5053_lm_eval_bridge import lm_eval_bridge_raw_validation_sets
from experiments.evals.exp5057_binary_network_security_evals import binary_network_security_raw_validation_sets
from experiments.evals.exp5061_package_metadata_evals import package_metadata_raw_validation_sets
from experiments.evals.exp5062_game_music_evals import game_music_raw_validation_sets
from experiments.evals.fineweb2_multilingual import fineweb2_multilingual_raw_validation_sets
from experiments.evals.formal_hardware_ppl import formal_hardware_raw_validation_sets
from experiments.evals.formal_methods_ppl import formal_methods_hardware_rtl_raw_validation_sets
from experiments.evals.gh_archive_structured_output import gh_archive_structured_output_raw_validation_sets
from experiments.evals.raw_web_markup_ppl import raw_web_markup_raw_validation_sets
from experiments.evals.web_markup_image_text_ppl import web_markup_image_text_raw_validation_sets
from experiments.marin_models import marin_tokenizer
from experiments.structured_evals import structured_evals_raw_validation_sets

RUN_KEY = "main_gap_all_available_v1"
RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8", regions=["us-central1"])
GAP_REPORT_RESOURCE_CONFIG = ResourceConfig.with_cpu(
    cpu=1,
    ram="64g",
    disk="20g",
    regions=["us-central1"],
    preemptible=True,
)
MAX_DOCS_PER_DATASET = 256
MAX_DOC_BYTES = 32_768
SKIPPED_DATASETS_FOR_THIS_RUN: set[str] = {
    # Hugging Face returned repeated 500s while listing these FineWeb2 parquet shards
    # during the May 2026 dashboard refresh.
    "fineweb2_multilingual/ces_Latn",
    "fineweb2_multilingual/ell_Grek",
    # NCBI no longer publishes the viral GFF URL used by this slice; keep the
    # cached FASTA slice and skip only the missing annotation surface.
    "bio_chem/refseq/refseq_viral_gff",
}


def _add(
    datasets: dict[str, RawTextEvaluationDataset],
    label: str,
    provider: Mapping[str, RawTextEvaluationDataset],
) -> None:
    skipped = sorted(set(provider).intersection(SKIPPED_DATASETS_FOR_THIS_RUN))
    provider = {key: dataset for key, dataset in provider.items() if key not in SKIPPED_DATASETS_FOR_THIS_RUN}
    overlap = set(datasets).intersection(provider)
    if overlap:
        raise ValueError(f"Duplicate dataset keys from {label}: {sorted(overlap)}")
    if skipped:
        print(f"{label}: skipped broken dashboard slices: {skipped}")
    provider = {
        key: dataset if dataset.tags else replace(dataset, tags=(key.split("/", maxsplit=1)[0],))
        for key, dataset in provider.items()
    }
    datasets.update(provider)


def all_available_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Compose every committed raw PPL provider intended for dashboard coverage."""

    datasets: dict[str, RawTextEvaluationDataset] = {}
    for label, provider in (
        ("default", default_raw_validation_sets()),
        ("fineweb2_multilingual", fineweb2_multilingual_raw_validation_sets()),
        ("web_markup_image_text", web_markup_image_text_raw_validation_sets()),
        ("formal_hardware", formal_hardware_raw_validation_sets()),
        ("bio_chem", bio_chem_raw_validation_sets()),
        ("lm_eval_bridge", lm_eval_bridge_raw_validation_sets()),
        ("binary_network_security", binary_network_security_raw_validation_sets()),
        ("package_metadata", package_metadata_raw_validation_sets()),
        ("game_music", game_music_raw_validation_sets()),
        ("raw_web_markup", raw_web_markup_raw_validation_sets()),
        ("formal_methods_hardware_rtl", formal_methods_hardware_rtl_raw_validation_sets()),
        ("structured_text", structured_evals_raw_validation_sets()),
        ("gh_archive_structured_output", gh_archive_structured_output_raw_validation_sets()),
        ("noisy_asr_ocr", noisy_asr_ocr_raw_validation_sets()),
        ("diagnostic_logs", diagnostic_log_raw_validation_sets()),
    ):
        _add(datasets, label, provider)
    return datasets


DATASETS = all_available_raw_validation_sets()

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
        "dataset_bundle=all_available_current_main",
        "model=marin-community/marin-32b-base",
        "region=us-central1",
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
        "dataset_bundle=all_available_current_main",
        "model=Qwen/Qwen3-32B",
        "region=us-central1",
    ],
)

GAP = model_perplexity_gap_from_scores(
    name=f"{RUN_KEY}/marin_32b-vs-qwen3_32b",
    model_a_name="marin-community/marin-32b-base",
    model_b_name="Qwen/Qwen3-32B",
    model_a_scores_path=MARIN_SCORES.as_input_name(),
    model_b_scores_path=QWEN3_SCORES.as_input_name(),
    resource_config=GAP_REPORT_RESOURCE_CONFIG,
    wandb_tags=[
        "eval=perplexity-gap",
        "dataset_bundle=all_available_current_main",
        "model_a=marin-community/marin-32b-base",
        "model_b=Qwen/Qwen3-32B",
        "region=us-central1",
    ],
)

STEPS: list[ExecutorStep] = [GAP]


if __name__ == "__main__":
    executor_main(
        STEPS,
        description=(
            "Run Marin 32B vs Qwen3 32B perplexity gap on every committed raw PPL provider "
            "plus capped public diagnostic-log slices."
        ),
    )
