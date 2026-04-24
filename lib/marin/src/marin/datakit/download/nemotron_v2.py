# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron v2 pre-training dataset download definitions.

These datasets come from the nvidia/Nemotron-Pre-Training-Datasets collection
on HuggingFace. They are additive to the original Nemotron-CC (v1) dataset.

Most of these datasets are gated and require HF_TOKEN at download time.
All use parquet format with a "text" field.
"""

from dataclasses import dataclass, field
from functools import cache

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec


@dataclass(frozen=True)
class NemotronV2Dataset:
    """Metadata for a single Nemotron v2 HuggingFace dataset."""

    hf_dataset_id: str
    revision: str
    subsets: dict[str, str] = field(default_factory=dict)
    """Maps subset_name -> glob pattern for parquet files within the download."""
    override_output_path: str | None = None
    """Allow to point at existing download output to avoid re-downloading"""


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
        override_output_path="raw/nemotron_cc_v2-674913",
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
        override_output_path="raw/nemotron_cc_v2_1-a7afb6",
    ),
    "nemotron_cc_code_v1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-CC-Code-v1",
        revision="5c5bebc",
        subsets={"all": "data/**/*.parquet"},
        override_output_path="raw/nemotron_cc_code_v1-c55cd9",
    ),
    "nemotron_cc_math_v1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-CC-Math-v1",
        revision="397a250",
        subsets={
            "3": "3/**/*.parquet",
            "4plus": "4plus/**/*.parquet",
            "4plus_mind": "4plus_MIND/**/*.parquet",
        },
        override_output_path="raw/nemotron_cc_math_v1-322fe4",
    ),
    "nemotron_pretraining_code_v1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-Code-v1",
        revision="01393d3",
        subsets={
            "synthetic_code": "Synthetic-Code/**/*.parquet",
            "code_metadata": "Nemotron-Code-Metadata/**/*.parquet",
        },
        override_output_path="raw/nemotron_pretraining_code_v1-175b37",
    ),
    "nemotron_pretraining_code_v2": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-Code-v2",
        revision="7b1a453",
        subsets={
            "code_metadata": "Nemotron-Code-Metadata/**/*.parquet",
            "synthetic_question_answering": "Synthetic-Code/synthetic-question-answering/**/*.parquet",
            "synthetic_student_teacher": "Synthetic-Code/synthetic-student-teacher/**/*.parquet",
            "synthetic_code_review": "Synthetic-Code/synthetic-code-review/**/*.parquet",
            "synthetic_rewriting": "Synthetic-Code/synthetic-rewriting/**/*.parquet",
            "synthetic_transpilation": "Synthetic-Code/synthetic-transpilation/**/*.parquet",
        },
        override_output_path="raw/nemotron_pretraining_code_v2-d15a24",
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
        override_output_path="raw/nemotron_pretraining_specialized_v1-a31fae",
    ),
    "nemotron_pretraining_sft_v1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-SFT-v1",
        revision="3f1a5b8",
        subsets={
            "sft_code": "Nemotron-SFT-Code/**/*.parquet",
            "sft_general": "Nemotron-SFT-General/**/*.parquet",
            "sft_math": "Nemotron-SFT-MATH/**/*.parquet",
        },
        override_output_path="raw/nemotron_pretraining_sft_v1-10f77e",
    ),
    "nemotron_pretraining_specialized_v1_1": NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-Specialized-v1.1",
        revision="13fa979",
        subsets={
            "code_concepts": "Nemotron-Pretraining-Code-Concepts/**/*.parquet",
            "economics": "Nemotron-Pretraining-Economics/**/*.parquet",
            "formal_logic": "Nemotron-Pretraining-Formal-Logic/**/*.parquet",
            "multiple_choice": "Nemotron-Pretraining-Multiple-Choice/**/*.parquet",
            "unconditional_algorithmic": "Nemotron-Pretraining-Unconditional-Algorithmic/**/*.parquet",
        },
        override_output_path="raw/nemotron_pretraining_specialized_v1_1-b12f71",
    ),
}


@cache
def download_nemotron_v2_step(family: str) -> StepSpec:
    """Create a download StepSpec for a Nemotron v2 dataset family.

    Cached because the registry flattens each family to per-subset rows; all
    subsets of a family must see the SAME download StepSpec object so the
    ferry dedupes it once across the DAG.
    """
    info = NEMOTRON_V2_DATASETS[family]
    return download_hf_step(
        f"raw/{family}",
        hf_dataset_id=info.hf_dataset_id,
        revision=info.revision,
        override_output_path=info.override_output_path,
    )


def normalize_nemotron_v2_step(download: StepSpec, *, family: str, subset: str) -> StepSpec:
    """Normalize one subset of a Nemotron v2 family.

    Each subset gets its own normalize step because normalize now processes a
    single directory. The subset's glob pattern (e.g. ``Diverse-QA/**/*.parquet``)
    is used to derive the input subdirectory under the family download.
    """
    info = NEMOTRON_V2_DATASETS[family]
    glob_pattern = info.subsets[subset]
    # Extract the directory prefix from the glob (e.g. "Diverse-QA/**/*.parquet" → "Diverse-QA")
    subset_dir = glob_pattern.split("/**")[0]
    return normalize_step(
        name=f"normalized/{family}/{subset}",
        download=download,
        text_field="text",
        id_field="id",
        file_extensions=(".parquet",),
        input_path=f"{download.output_path}/{subset_dir}",
    )


def nemotron_v2_normalize_steps(family: str) -> dict[str, tuple[StepSpec, ...]]:
    """Full ``(download, normalize)`` chain per subset of a Nemotron v2 family.

    One download step is shared across every subset; each subset gets its own
    normalize step parameterized by the subset glob's directory prefix.
    Returns ``{marin_name: (download, normalize)}`` where marin_name is
    ``f"{family}/{subset}"`` — matching the ``DatakitSource.name`` convention.
    """
    info = NEMOTRON_V2_DATASETS[family]
    download = download_nemotron_v2_step(family)
    return {
        f"{family}/{subset}": (download, normalize_nemotron_v2_step(download, family=family, subset=subset))
        for subset in info.subsets
    }
