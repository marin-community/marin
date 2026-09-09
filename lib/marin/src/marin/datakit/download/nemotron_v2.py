# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nemotron v2 pre-training dataset download definitions.

These datasets come from the nvidia/Nemotron-Pre-Training-Datasets collection
on HuggingFace. They are additive to the original Nemotron-CC (v1) dataset.

Most of these datasets are gated and require HF_TOKEN at download time.
All use parquet format with a "text" field.
"""

import re
from dataclasses import dataclass, field
from functools import cache

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import checked_openai_chat_document, load_parquet_batched
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec


@dataclass(frozen=True)
class NemotronV2Dataset:
    """Metadata for a single Nemotron v2 HuggingFace dataset."""

    hf_dataset_id: str
    revision: str
    subsets: dict[str, str] = field(default_factory=dict)
    """Maps subset_name -> glob pattern for parquet files within the download."""
    subset_text_fields: dict[str, str] = field(default_factory=dict)
    """Per-subset overrides for the text column. Subsets not listed use ``"text"``."""
    subset_normalize_worker_resources: dict[str, ResourceConfig] = field(default_factory=dict)
    """Optional custom per-subset overrides for normalize worker resources"""
    override_output_path: str | None = None
    """Allow to point at existing download output to avoid re-downloading"""


NEMOTRON_PRETRAINING_LEGAL_V1 = "nemotron_pretraining_legal_v1"
NEMOTRON_PRETRAINING_SPECIALIZED_V1_2 = "nemotron_pretraining_specialized_v1_2"
NEMOTRON_PRETRAINING_SFT_V1 = "nemotron_pretraining_sft_v1"

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
        # Synthetic-Code/* parquet files store the document under `content`, not `text`.
        subset_text_fields={
            "synthetic_question_answering": "content",
            "synthetic_student_teacher": "content",
            "synthetic_code_review": "content",
            "synthetic_rewriting": "content",
            "synthetic_transpilation": "content",
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
    NEMOTRON_PRETRAINING_SFT_V1: NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-SFT-v1",
        revision="3f1a5b8",
        subsets={
            "sft_code": "Nemotron-SFT-Code/**/*.parquet",
            "sft_general": "Nemotron-SFT-General/**/*.parquet",
            "sft_math": "Nemotron-SFT-MATH/**/*.parquet",
        },
        # SFT-General and SFT-MATH have skewed shards and unrealistic row group sizes, e.g.:
        # "NumRowGroups": 3", "RowGroups": [{ "NumRows": 262144, "TotalByteSize": 4715315268, ...
        # Instead of fighting it, let's just normalize it to something reasonable and move
        # forward.
        subset_normalize_worker_resources={
            "sft_general": ResourceConfig(cpu=2, ram="64g", disk="10g"),
            "sft_math": ResourceConfig(cpu=2, ram="64g", disk="10g"),
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
    NEMOTRON_PRETRAINING_SPECIALIZED_V1_2: NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-Specialized-v1.2",
        revision="807afc1",
        subsets={
            "fact_seeking": "Nemotron-Pretraining-Fact-Seeking/**/*.parquet",
            "generative": "Nemotron-Pretraining-Generative/**/*.parquet",
            "moral_scenarios": "Nemotron-Pretraining-Moral-Scenarios/**/*.parquet",
            "multiple_choice": "Nemotron-Pretraining-Multiple-Choice/**/*.parquet",
        },
    ),
    NEMOTRON_PRETRAINING_LEGAL_V1: NemotronV2Dataset(
        hf_dataset_id="nvidia/Nemotron-Pretraining-Legal-v1",
        revision="3d91d58",
        subsets={
            "california_code_of_regulations": "Nemotron-Pretraining-Legal-California-Code-Of-Regulations/**/*.parquet",
            "case_law_summary": "Nemotron-Pretraining-Legal-Case-Law-Summary/**/*.parquet",
            "casehold": "Nemotron-Pretraining-Legal-CaseHOLD/**/*.parquet",
            "definition_classification": "Nemotron-Pretraining-Legal-Definition-Classification/**/*.parquet",
            "diversity_jurisdiction": "Nemotron-Pretraining-Legal-Diversity-Jurisdiction/**/*.parquet",
            "ecfr": "Nemotron-Pretraining-Legal-eCFR/**/*.parquet",
            "ecfr_qa": "Nemotron-Pretraining-Legal-eCFR-QA/**/*.parquet",
            "function_of_decision": "Nemotron-Pretraining-Legal-Function-Of-Decision/**/*.parquet",
            "globalcit": "Nemotron-Pretraining-Legal-GlobalCit/**/*.parquet",
            "legalbench_cuad_v2": "Nemotron-Pretraining-Legal-LegalBench-CUAD-v2/**/*.parquet",
            "nycourts_judicial_ethics_opinions": (
                "Nemotron-Pretraining-Legal-NYCourts-Judicial-Ethics-Opinions/**/*.parquet"
            ),
        },
    ),
}

_GENERAL_TURN = re.compile(r"<extra_id_1>(User|Assistant)\n")


def _nemotron_sft_messages(text: str, subset: str) -> list[dict] | None:
    """Recover the documented role delimiters used by each Nemotron SFT subset."""
    if subset == "sft_general":
        matches = list(_GENERAL_TURN.finditer(text))
        if not matches or matches[0].start() != 0:
            return None
        messages = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            messages.append({"role": match.group(1).lower(), "content": text[match.end() : end].strip()})
        return messages
    if subset == "sft_math":
        if not text.startswith("input: ") or " output: " not in text:
            return None
        prompt, response = text.removeprefix("input: ").split(" output: ", 1)
        return [{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": response.strip()}]
    if subset == "sft_code":
        marker = "\n\n<think>"
        if marker not in text:
            return None
        prompt, response = text.split(marker, 1)
        return [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": f"<think>{response}".strip()},
        ]
    raise ValueError(f"Unsupported Nemotron SFT subset {subset!r}")


def _nemotron_sft_row_to_chat(row: dict, subset: str) -> list[dict]:
    text = row.get("text")
    if not isinstance(text, str):
        return []
    messages = _nemotron_sft_messages(text, subset)
    if messages is None:
        counters.pipeline.update_counter(f"nemotron_sft/{subset}/unparsed", 1)
        return []
    return checked_openai_chat_document(
        messages,
        NEMOTRON_V2_DATASETS[NEMOTRON_PRETRAINING_SFT_V1].hf_dataset_id,
        counter_prefix=f"nemotron_sft/{subset}/chat",
    )


def _transform_nemotron_sft_chat(input_path: str, output_path: str, subset: str) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, "**/*.parquet"))
        .flat_map(load_parquet_batched)
        .flat_map(lambda row: _nemotron_sft_row_to_chat(row, subset))
        .write_parquet(prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"), skip_existing=True)
    )
    ZephyrContext(name=f"nemotron-sft-{subset}-chat", resources=ResourceConfig(cpu=1, ram="32g")).execute(pipeline)


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
        text_field=info.subset_text_fields.get(subset, "text"),
        id_field="id",
        file_extensions=(".parquet",),
        relative_input_path=subset_dir,
        worker_resources=info.subset_normalize_worker_resources.get(subset),
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


def nemotron_sft_chat_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return structured-chat chains for the three Nemotron SFT subsets."""
    family = NEMOTRON_PRETRAINING_SFT_V1
    info = NEMOTRON_V2_DATASETS[family]
    download = download_nemotron_v2_step(family)
    chains: dict[str, tuple[StepSpec, ...]] = {}
    for subset, pattern in info.subsets.items():
        subset_dir = pattern.split("/**")[0]
        processed = StepSpec(
            name=f"processed-chat/{family}/{subset}",
            deps=[download],
            fn=lambda output_path, source_subset=subset, source_dir=subset_dir: _transform_nemotron_sft_chat(
                prefix_join(download.output_path, source_dir), output_path, source_subset
            ),
            hash_attrs={"version": "2026.09.05.2.harmony-direct", "subset": subset},
        )
        normalized = normalize_chat_step(
            name=f"normalized-chat/{family}/{subset}",
            download=processed,
            file_extensions=(".parquet",),
            worker_resources=info.subset_normalize_worker_resources.get(subset),
        )
        chains[f"nemotron_sft/{subset}"] = (download, processed, normalized)
    return chains
