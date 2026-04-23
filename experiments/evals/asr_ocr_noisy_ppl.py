# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in ASR/OCR noisy-text raw eval slices for perplexity-gap reports.

This module materializes paired noisy/clean text from ASR and OCR sources, then
registers both variants as raw-text datasets so gap reports can compute deltas.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import fsspec
from datasets import load_dataset
from fray.v2 import ResourceConfig
from levanter.utils import fsspec_utils

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import HfDatasetSpec

ASR_OCR_NOISY_DATASET_ROOT = "asr_ocr_noisy_ppl"
NOISY_TEXT_FIELD = "noisy_text"
CLEAN_TEXT_FIELD = "clean_text"
DEFAULT_RAW_SHARD_NAME = "data-00000-of-00001.jsonl.gz"


class NoisyTextFamily(StrEnum):
    ASR = "asr"
    OCR = "ocr"


@dataclass(frozen=True)
class NoisyTextSlice:
    registry_name: str
    family: NoisyTextFamily
    source_url: str
    hf_dataset: HfDatasetSpec
    split: str
    noisy_key: str
    clean_key: str
    max_rows: int
    notes: str = ""

    @property
    def tags(self) -> tuple[str, ...]:
        return (ASR_OCR_NOISY_DATASET_ROOT, f"family:{self.family.value}", f"source:{self.registry_name}")


ASR_OCR_NOISY_SLICES: tuple[NoisyTextSlice, ...] = (
    NoisyTextSlice(
        registry_name="hypr_librispeech_without_lm_test_clean",
        family=NoisyTextFamily.ASR,
        source_url="https://huggingface.co/datasets/ASR-HypR/LibriSpeech_withoutLM",
        hf_dataset=HfDatasetSpec(id="ASR-HypR/LibriSpeech_withoutLM"),
        split="test_clean",
        noisy_key="hyps",
        clean_key="ref",
        max_rows=512,
        notes=(
            "HypR exposes n-best ASR hypotheses per utterance. We linearize top-1 for noisy_text and keep ref "
            "as clean_text. Verify downstream use remains compatible with LibriSpeech-derived licensing terms."
        ),
    ),
    NoisyTextSlice(
        registry_name="rtm_sgt_ocr_v1_train",
        family=NoisyTextFamily.OCR,
        source_url="https://huggingface.co/datasets/ReadingTimeMachine/rtm-sgt-ocr-v1",
        hf_dataset=HfDatasetSpec(id="ReadingTimeMachine/rtm-sgt-ocr-v1"),
        split="train",
        noisy_key="source",
        clean_key="target",
        max_rows=512,
        notes=(
            "ReadingTimeMachine OCR post-correction pairs may inherit source-specific archival rights. Treat as "
            "eval-only until redistribution terms are reviewed per source collection."
        ),
    ),
)


@dataclass(frozen=True)
class NoisyAsrOcrRawConfig:
    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    max_rows_per_slice_override: int | None = None
    slices: tuple[NoisyTextSlice, ...] = ASR_OCR_NOISY_SLICES


def _coerce_text(value: object) -> str | None:
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for item in value:
            text = _coerce_text(item)
            if text is not None:
                return text
        return None
    return None


def linearize_noisy_clean_row(
    row: Mapping[str, object],
    *,
    noisy_key: str,
    clean_key: str,
) -> dict[str, str] | None:
    """Extract paired noisy/clean text fields from one source row."""
    noisy_text = _coerce_text(row.get(noisy_key))
    clean_text = _coerce_text(row.get(clean_key))
    if noisy_text is None or clean_text is None:
        return None
    return {NOISY_TEXT_FIELD: noisy_text, CLEAN_TEXT_FIELD: clean_text}


def _iter_linearized_rows(slice_: NoisyTextSlice) -> Iterable[dict[str, str]]:
    dataset = load_dataset(
        slice_.hf_dataset.id,
        name=slice_.hf_dataset.name,
        split=slice_.split,
        streaming=True,
    )
    for row in dataset:
        linearized = linearize_noisy_clean_row(row, noisy_key=slice_.noisy_key, clean_key=slice_.clean_key)
        if linearized is not None:
            yield linearized


def _slice_output_path(output_path: str, registry_name: str) -> str:
    return os.path.join(output_path, registry_name, DEFAULT_RAW_SHARD_NAME)


def materialize_noisy_asr_ocr_raw(config: NoisyAsrOcrRawConfig) -> None:
    """Materialize paired noisy/clean text rows into jsonl.gz shards."""
    fsspec_utils.mkdirs(config.output_path)
    for slice_ in config.slices:
        output_file = _slice_output_path(config.output_path, slice_.registry_name)
        fsspec_utils.mkdirs(os.path.dirname(output_file))
        row_cap = slice_.max_rows if config.max_rows_per_slice_override is None else config.max_rows_per_slice_override
        if row_cap <= 0:
            raise ValueError(f"row cap must be positive, got {row_cap}.")
        with fsspec.open(output_file, "wt", compression="gzip") as sink:
            for index, record in enumerate(_iter_linearized_rows(slice_)):
                if index >= row_cap:
                    break
                sink.write(json.dumps(record, ensure_ascii=True))
                sink.write("\n")


noisy_asr_ocr_raw = ExecutorStep(
    name=os.path.join("raw", "evals", ASR_OCR_NOISY_DATASET_ROOT),
    description="Materialize paired ASR/OCR noisy-clean raw eval slices from Hugging Face.",
    fn=remote(
        materialize_noisy_asr_ocr_raw,
        resources=ResourceConfig.with_cpu(cpu=4, ram="32g", disk="40g"),
        pip_dependency_groups=["cpu"],
    ),
    config=NoisyAsrOcrRawConfig(),
)


def noisy_asr_ocr_raw_validation_sets(
    *,
    noisy_asr_ocr_raw: ExecutorStep = noisy_asr_ocr_raw,
) -> dict[str, RawTextEvaluationDataset]:
    """Register clean and noisy variants for each ASR/OCR raw slice."""
    datasets: dict[str, RawTextEvaluationDataset] = {}
    for slice_ in ASR_OCR_NOISY_SLICES:
        raw_pattern = os.path.join(slice_.registry_name, "data-*.jsonl.gz")
        key_root = os.path.join(ASR_OCR_NOISY_DATASET_ROOT, slice_.registry_name)

        datasets[os.path.join(key_root, "noisy")] = raw_text_dataset(
            noisy_asr_ocr_raw.cd(raw_pattern),
            text_key=NOISY_TEXT_FIELD,
            tags=(*slice_.tags, "variant:noisy"),
        )
        datasets[os.path.join(key_root, "clean")] = raw_text_dataset(
            noisy_asr_ocr_raw.cd(raw_pattern),
            text_key=CLEAN_TEXT_FIELD,
            tags=(*slice_.tags, "variant:clean"),
        )

    return datasets
