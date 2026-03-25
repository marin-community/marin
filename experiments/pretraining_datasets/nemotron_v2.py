# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron v2 pre-training dataset definitions and tokenization.

These datasets come from the nvidia/Nemotron-Pre-Training-Datasets collection
on HuggingFace. They are additive to the original Nemotron-CC (v1) dataset
defined in nemotron.py.

Most of these datasets are gated and require HF_TOKEN at download time.
All use parquet format with a "text" field.
"""

import os.path

from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep

# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

# Each entry: (hf_id, revision, subsets_dict)
# subsets_dict maps subset_name -> glob pattern for parquet files within the download

NEMOTRON_V2_DATASETS = {
    "nemotron_cc_v2": {
        "hf_dataset_id": "nvidia/Nemotron-CC-v2",
        "revision": "229a2e7",
        "subsets": {
            "diverse_qa": "Diverse-QA/**/*.parquet",
            "high_quality": "High-Quality/**/*.parquet",
            "high_quality_synthetic": "High-Quality-Synthetic/**/*.parquet",
            "medium_high_quality": "Medium-High-Quality/**/*.parquet",
            "medium_quality": "Medium-Quality/**/*.parquet",
            "translated_diverse_qa": "Translated-Diverse-QA/**/*.parquet",
        },
    },
    "nemotron_cc_v2_1": {
        "hf_dataset_id": "nvidia/Nemotron-CC-v2.1",
        "revision": "ba6f2aa",
        "subsets": {
            "high_quality": "High-Quality/**/*.parquet",
            "high_quality_dqa": "High-Quality-DQA/**/*.parquet",
            "high_quality_synthetic": "High-Quality-Synthetic/**/*.parquet",
            "high_quality_translated": "High-Quality-Translated-To-English/**/*.parquet",
            "high_quality_translated_synthetic": "High-Quality-Translated-To-English-Synthetic/**/*.parquet",
            "medium_high_quality": "Medium-High-Quality/**/*.parquet",
            "medium_high_quality_synthetic": "Medium-High-Quality-Synthetic/**/*.parquet",
            "medium_high_quality_translated": "Medium-High-Quality-Translated-To-English/**/*.parquet",
            "medium_quality": "Medium-Quality/**/*.parquet",
        },
    },
    "nemotron_cc_code_v1": {
        "hf_dataset_id": "nvidia/Nemotron-CC-Code-v1",
        "revision": "5c5bebc",
        "subsets": {
            "all": "data/**/*.parquet",
        },
    },
    "nemotron_cc_math_v1": {
        "hf_dataset_id": "nvidia/Nemotron-CC-Math-v1",
        "revision": "397a250",
        "subsets": {
            "3": "3/**/*.parquet",
            "4plus": "4plus/**/*.parquet",
            "4plus_mind": "4plus_MIND/**/*.parquet",
        },
    },
    "nemotron_pretraining_code_v1": {
        "hf_dataset_id": "nvidia/Nemotron-Pretraining-Code-v1",
        "revision": "01393d3",
        "subsets": {
            "synthetic_code": "Synthetic-Code/**/*.parquet",
            "code_metadata": "Nemotron-Code-Metadata/**/*.parquet",
        },
    },
    "nemotron_pretraining_code_v2": {
        "hf_dataset_id": "nvidia/Nemotron-Pretraining-Code-v2",
        "revision": "7b1a453",
        "subsets": {
            "code_metadata": "Nemotron-Code-Metadata/**/*.parquet",
            "synthetic_question_answering": "Synthetic-Question-Answering/**/*.parquet",
            "synthetic_student_teacher": "Synthetic-Student-Teacher/**/*.parquet",
            "synthetic_code_review": "Synthetic-Code-Review/**/*.parquet",
            "synthetic_rewriting": "Synthetic-Rewriting/**/*.parquet",
            "synthetic_transpilation": "Synthetic-Transpilation/**/*.parquet",
        },
    },
    "nemotron_pretraining_specialized_v1": {
        "hf_dataset_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
        "revision": "9ed3718",
        "subsets": {
            "wiki_rewrite": "Nemotron-Pretraining-Wiki-Rewrite/**/*.parquet",
            "math_textbooks": "Nemotron-Pretraining-Math-Textbooks/**/*.parquet",
            "stem_sft": "Nemotron-Pretraining-STEM-SFT/**/*.parquet",
            "scientific_coding": "Nemotron-Pretraining-Scientific-Coding/**/*.parquet",
            "rqa": "Nemotron-Pretraining-RQA/**/*.parquet",
            "infinibyte_reasoning": "Nemotron-Pretraining-InfiniByte-Reasoning/**/*.parquet",
        },
    },
    "nemotron_pretraining_sft_v1": {
        "hf_dataset_id": "nvidia/Nemotron-Pretraining-SFT-v1",
        "revision": "3f1a5b8",
        "subsets": {
            "sft_code": "Nemotron-SFT-Code/**/*.parquet",
            "sft_general": "Nemotron-SFT-General/**/*.parquet",
            "sft_math": "Nemotron-SFT-MATH/**/*.parquet",
        },
    },
}


# ============================================================================
# RAW DATASET DOWNLOADS
# ============================================================================

downloads: dict[str, ExecutorStep] = {}
for _family, _info in NEMOTRON_V2_DATASETS.items():
    downloads[_family] = ExecutorStep(
        name=f"raw/{_family}",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=_info["hf_dataset_id"],
            revision=versioned(_info["revision"]),
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
        ),
    )


# ============================================================================
# TOKENIZATION
# ============================================================================


def tokenize_nemotron_v2_family(
    family: str,
    *,
    tokenizer: str | None = None,
) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all subsets of a Nemotron HF dataset family."""
    if tokenizer is None:
        from experiments.llama import llama3_tokenizer

        tokenizer = llama3_tokenizer

    info = NEMOTRON_V2_DATASETS[family]
    download_step = downloads[family]

    steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for subset, glob_pattern in info["subsets"].items():
        output_name = os.path.join("tokenized", family, subset)
        step = ExecutorStep(
            name=output_name,
            fn=tokenize,
            config=TokenizeConfig(
                train_paths=[download_step / glob_pattern],
                validation_paths=versioned([]),
                cache_path=this_output_path(),
                tokenizer=versioned(tokenizer),
            ),
        )
        steps[f"{family}/{subset}"] = step

    return steps


def tokenize_all_nemotron_v2(*, tokenizer: str | None = None) -> dict[str, TokenizerStep]:
    """Generate tokenization steps for all Nemotron HF datasets."""
    all_steps: dict[str, TokenizerStep] = {}
    for family in NEMOTRON_V2_DATASETS:
        all_steps.update(tokenize_nemotron_v2_family(family, tokenizer=tokenizer))
    return all_steps
