# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF-backed numeric and structured-format PPL validation slices."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from enum import StrEnum

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, supervised_text_dataset
from marin.processing.tokenize import HfDatasetSpec

EPIC_5005 = 5005
NUMERIC_FORMAT_ISSUE = 5614
NUMERIC_FORMAT_HF_DATASET_ID = "marin-community/synth-numeric-format-ppl"
NUMERIC_FORMAT_SOURCE = "generated_numeric_format_ppl_v1"
NUMERIC_FORMAT_HF_REVISION = "57cd2ab0f7b507cc2598d0cceeb95db47a739653"
NUMERIC_FORMAT_SEED = 5614
NUMERIC_FORMAT_PROMPT_ABLATION_HF_DATASET_ID = "marin-community/synth-numeric-format-prompt-ablation-ppl"
NUMERIC_FORMAT_PROMPT_ABLATION_SOURCE = "generated_numeric_format_prompt_ablation_ppl_v1"
NUMERIC_FORMAT_PROMPT_ABLATION_HF_REVISION = "bca4b9413bc72ae66614da99dafcc87ab7bc074f"
NUMERIC_FORMAT_PROMPT_ABLATION_SEED = 60935010
NUMERIC_FORMAT_PROMPT_ABLATION_VARIANTS = ("newline", "arrow", "equals")
EXAMPLES_PER_CONFIG = 1000


class NumericFormatFamily(StrEnum):
    NUMERIC_TRANSDUCTION = "numeric_transduction"
    STRUCTURED_RECORDS = "structured_records"
    DENSE_NUMERIC_BLOBS = "dense_numeric_blobs"
    FORMAT_TRANSFORMS = "format_transforms"
    CHUNK_BOUNDARY = "chunk_boundary"


@dataclass(frozen=True)
class NumericFormatPplSlice:
    family: NumericFormatFamily
    task_name: str
    hf_config_name: str

    @property
    def registry_key(self) -> str:
        return posixpath.join("synthetic_numeric_format_ppl", self.family.value, self.task_name)

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "synthetic_numeric_format_ppl",
            f"epic:{EPIC_5005}",
            f"issue:{NUMERIC_FORMAT_ISSUE}",
            f"family:{self.family.value}",
            f"task:{self.task_name}",
            f"seed:{NUMERIC_FORMAT_SEED}",
            f"examples:{EXAMPLES_PER_CONFIG}",
            f"source:{NUMERIC_FORMAT_SOURCE}",
            f"hf_revision:{NUMERIC_FORMAT_HF_REVISION}",
            "loss:target_only",
        )

    def to_raw_text_dataset(self, *, hf_dataset_id: str) -> RawTextEvaluationDataset:
        return supervised_text_dataset(
            HfDatasetSpec(id=hf_dataset_id, name=self.hf_config_name),
            input_key="input",
            target_key="target",
            split="validation",
            tags=self.tags,
        )


@dataclass(frozen=True)
class NumericFormatPromptAblationPplSlice:
    base_slice: NumericFormatPplSlice
    prompt_variant: str

    @property
    def family(self) -> NumericFormatFamily:
        return self.base_slice.family

    @property
    def task_name(self) -> str:
        return self.base_slice.task_name

    @property
    def hf_config_name(self) -> str:
        return f"{self.prompt_variant}_{self.task_name}"

    @property
    def registry_key(self) -> str:
        return posixpath.join(
            "synthetic_numeric_format_prompt_ablation_ppl",
            self.prompt_variant,
            self.family.value,
            self.task_name,
        )

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "synthetic_numeric_format_prompt_ablation_ppl",
            f"epic:{EPIC_5005}",
            f"issue:{NUMERIC_FORMAT_ISSUE}",
            f"family:{self.family.value}",
            f"task:{self.task_name}",
            f"prompt_variant:{self.prompt_variant}",
            f"seed:{NUMERIC_FORMAT_PROMPT_ABLATION_SEED}",
            f"examples:{EXAMPLES_PER_CONFIG}",
            f"source:{NUMERIC_FORMAT_PROMPT_ABLATION_SOURCE}",
            f"hf_revision:{NUMERIC_FORMAT_PROMPT_ABLATION_HF_REVISION}",
            "loss:target_only",
        )

    def to_raw_text_dataset(self, *, hf_dataset_id: str) -> RawTextEvaluationDataset:
        return supervised_text_dataset(
            HfDatasetSpec(id=hf_dataset_id, name=self.hf_config_name),
            input_key="input",
            target_key="target",
            split="validation",
            tags=self.tags,
        )


NUMERIC_FORMAT_PPL_SLICES: tuple[NumericFormatPplSlice, ...] = (
    NumericFormatPplSlice(
        family=NumericFormatFamily.NUMERIC_TRANSDUCTION,
        task_name="numeric_copy_increment",
        hf_config_name="numeric_copy_increment",
    ),
    NumericFormatPplSlice(
        family=NumericFormatFamily.NUMERIC_TRANSDUCTION,
        task_name="numeric_compare_sort",
        hf_config_name="numeric_compare_sort",
    ),
    NumericFormatPplSlice(
        family=NumericFormatFamily.NUMERIC_TRANSDUCTION,
        task_name="numeric_range_checksum_base",
        hf_config_name="numeric_range_checksum_base",
    ),
    NumericFormatPplSlice(
        family=NumericFormatFamily.STRUCTURED_RECORDS,
        task_name="tabular_tsv_csv",
        hf_config_name="tabular_tsv_csv",
    ),
    NumericFormatPplSlice(
        family=NumericFormatFamily.STRUCTURED_RECORDS,
        task_name="network_ip_port_rows",
        hf_config_name="network_ip_port_rows",
    ),
    NumericFormatPplSlice(
        family=NumericFormatFamily.DENSE_NUMERIC_BLOBS,
        task_name="svg_path_numeric_blobs",
        hf_config_name="svg_path_numeric_blobs",
    ),
    NumericFormatPplSlice(
        family=NumericFormatFamily.DENSE_NUMERIC_BLOBS,
        task_name="json_numeric_arrays",
        hf_config_name="json_numeric_arrays",
    ),
    NumericFormatPplSlice(
        family=NumericFormatFamily.DENSE_NUMERIC_BLOBS,
        task_name="mmcif_coordinate_tables",
        hf_config_name="mmcif_coordinate_tables",
    ),
    NumericFormatPplSlice(
        family=NumericFormatFamily.FORMAT_TRANSFORMS,
        task_name="format_preserving_transforms",
        hf_config_name="format_preserving_transforms",
    ),
    NumericFormatPplSlice(
        family=NumericFormatFamily.CHUNK_BOUNDARY,
        task_name="chunk_boundary_stress",
        hf_config_name="chunk_boundary_stress",
    ),
)


NUMERIC_FORMAT_PROMPT_ABLATION_PPL_SLICES: tuple[NumericFormatPromptAblationPplSlice, ...] = tuple(
    NumericFormatPromptAblationPplSlice(base_slice=slice_, prompt_variant=prompt_variant)
    for prompt_variant in NUMERIC_FORMAT_PROMPT_ABLATION_VARIANTS
    for slice_ in NUMERIC_FORMAT_PPL_SLICES
)


def numeric_format_raw_validation_sets(
    *, hf_dataset_id: str = NUMERIC_FORMAT_HF_DATASET_ID
) -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset(hf_dataset_id=hf_dataset_id)
        for slice_ in NUMERIC_FORMAT_PPL_SLICES
    }


def numeric_format_prompt_ablation_raw_validation_sets(
    *, hf_dataset_id: str = NUMERIC_FORMAT_PROMPT_ABLATION_HF_DATASET_ID
) -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset(hf_dataset_id=hf_dataset_id)
        for slice_ in NUMERIC_FORMAT_PROMPT_ABLATION_PPL_SLICES
    }
