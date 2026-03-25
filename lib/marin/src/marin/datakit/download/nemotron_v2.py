# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron v2 pre-training dataset download definitions.

These datasets come from the nvidia/Nemotron-Pre-Training-Datasets collection
on HuggingFace. They are additive to the original Nemotron-CC (v1) dataset.

Most of these datasets are gated and require HF_TOKEN at download time.
All use parquet format with a "text" field.
"""

from dataclasses import dataclass, field

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec


@dataclass(frozen=True)
class NemotronV2Dataset:
    """Metadata for a single Nemotron v2 HuggingFace dataset."""

    hf_dataset_id: str
    revision: str
    subsets: dict[str, str] = field(default_factory=dict)
    """Maps subset_name -> glob pattern for parquet files within the download."""


NEMOTRON_V2_DATASETS: dict[str, NemotronV2Dataset] = {
    "nemotron_cc_v2": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-CC-v2",
        revision="229a2e7",
        subsets={
            "diverse_qa": "Diverse-QA/**/*.parquet",
            "high_quality": "High-Quality/**/*.parquet",
            "high_quality_synthetic": "High-Quality-Synthetic/**/*.parquet",
            "medium_high_quality": "Medium-High-Quality/**/*.parquet",
            "medium_quality": "Medium-Quality/**/*.parquet",
            "translated_diverse_qa": "Translated-Diverse-QA/**/*.parquet",
        },
    ),
    "nemotron_cc_v2_1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-CC-v2.1",
        revision="ba6f2aa",
        subsets={
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
    ),
    "nemotron_cc_code_v1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-CC-Code-v1",
        revision="5c5bebc",
        subsets={"all": "data/**/*.parquet"},
    ),
    "nemotron_cc_math_v1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-CC-Math-v1",
        revision="397a250",
        subsets={
            "3": "3/**/*.parquet",
            "4plus": "4plus/**/*.parquet",
            "4plus_mind": "4plus_MIND/**/*.parquet",
        },
    ),
    "nemotron_pretraining_code_v1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-Code-v1",
        revision="01393d3",
        subsets={
            "synthetic_code": "Synthetic-Code/**/*.parquet",
            "code_metadata": "Nemotron-Code-Metadata/**/*.parquet",
        },
    ),
    "nemotron_pretraining_code_v2": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-Code-v2",
        revision="7b1a453",
        subsets={
            "code_metadata": "Nemotron-Code-Metadata/**/*.parquet",
            "synthetic_question_answering": "Synthetic-Question-Answering/**/*.parquet",
            "synthetic_student_teacher": "Synthetic-Student-Teacher/**/*.parquet",
            "synthetic_code_review": "Synthetic-Code-Review/**/*.parquet",
            "synthetic_rewriting": "Synthetic-Rewriting/**/*.parquet",
            "synthetic_transpilation": "Synthetic-Transpilation/**/*.parquet",
        },
    ),
    "nemotron_pretraining_specialized_v1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-Specialized-v1",
        revision="9ed3718",
        subsets={
            "wiki_rewrite": "Nemotron-Pretraining-Wiki-Rewrite/**/*.parquet",
            "math_textbooks": "Nemotron-Pretraining-Math-Textbooks/**/*.parquet",
            "stem_sft": "Nemotron-Pretraining-STEM-SFT/**/*.parquet",
            "scientific_coding": "Nemotron-Pretraining-Scientific-Coding/**/*.parquet",
            "rqa": "Nemotron-Pretraining-RQA/**/*.parquet",
            "infinibyte_reasoning": "Nemotron-Pretraining-InfiniByte-Reasoning/**/*.parquet",
        },
    ),
    "nemotron_pretraining_sft_v1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-SFT-v1",
        revision="3f1a5b8",
        subsets={
            "sft_code": "Nemotron-SFT-Code/**/*.parquet",
            "sft_general": "Nemotron-SFT-General/**/*.parquet",
            "sft_math": "Nemotron-SFT-MATH/**/*.parquet",
        },
    ),
}


def download_nemotron_v2_step(family: str) -> StepSpec:
    """Create a download StepSpec for a Nemotron v2 dataset family."""
    info = NEMOTRON_V2_DATASETS[family]
    return download_hf_step(
        f"raw/{family}",
        hf_dataset_id=info.hf_dataset_id,
        revision=info.revision,
    )
