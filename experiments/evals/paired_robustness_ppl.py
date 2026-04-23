# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in paired paraphrase/translation robustness PPL slices for issue #5096."""

from __future__ import annotations

import logging
import os
import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import datasets
from fray.v2 import ResourceConfig
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep, this_output_path
from marin.execution.remote import remote
from zephyr import write_jsonl_file

logger = logging.getLogger(__name__)

EPIC_5005 = 5005
PAIR_ROBUSTNESS_ISSUE = 5096
DEFAULT_SAMPLE_CAP = 1_024
DEFAULT_SHARD_SIZE = 512


class PairedRobustnessFamily(StrEnum):
    PARAPHRASE = "paraphrase"
    TRANSLATION = "translation"


class PairedTextView(StrEnum):
    SOURCE = "source"
    TARGET = "target"
    TARGET_GIVEN_SOURCE = "target_given_source"


ALL_PAIRED_TEXT_VIEWS: tuple[PairedTextView, ...] = (
    PairedTextView.SOURCE,
    PairedTextView.TARGET,
    PairedTextView.TARGET_GIVEN_SOURCE,
)


@dataclass(frozen=True)
class PairedRobustnessSlice:
    """Definition of one held-out paired robustness source split."""

    name: str
    family: PairedRobustnessFamily
    source_url: str
    hf_dataset_id: str
    hf_dataset_name: str | None
    split: str
    source_field: str
    target_field: str
    source_label: str
    target_label: str
    max_pairs: int
    trust_remote_code: bool = False
    label_field: str | None = None
    required_label: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if (self.label_field is None) != (self.required_label is None):
            raise ValueError("label_field and required_label must both be set or both be None.")
        if self.source_field == self.target_field:
            raise ValueError("source_field and target_field must differ for paired eval slices.")
        if self.max_pairs <= 0:
            raise ValueError(f"max_pairs must be positive, got {self.max_pairs}.")

    @property
    def raw_step_key(self) -> str:
        return posixpath.join(self.family.value, self.name, self.split)

    @property
    def dataset_root(self) -> str:
        return posixpath.join("paired_robustness_ppl", self.family.value, self.name, self.split)

    def dataset_key(self, view: PairedTextView) -> str:
        return posixpath.join(self.dataset_root, view.value)

    def tags_for_view(self, view: PairedTextView) -> tuple[str, ...]:
        return (
            "paired_robustness_ppl",
            f"epic:{EPIC_5005}",
            f"issue:{PAIR_ROBUSTNESS_ISSUE}",
            f"family:{self.family.value}",
            f"source:{self.name}",
            f"split:{self.split}",
            f"view:{view.value}",
        )


PAIRED_ROBUSTNESS_SLICES: tuple[PairedRobustnessSlice, ...] = (
    PairedRobustnessSlice(
        name="paws_labeled_final",
        family=PairedRobustnessFamily.PARAPHRASE,
        source_url="https://huggingface.co/datasets/google-research-datasets/paws",
        hf_dataset_id="google-research-datasets/paws",
        hf_dataset_name="labeled_final",
        split="validation",
        source_field="sentence1",
        target_field="sentence2",
        source_label="sentence_1",
        target_label="sentence_2",
        label_field="label",
        required_label=1,
        max_pairs=DEFAULT_SAMPLE_CAP,
        notes="Held-out PAWS validation paraphrase pairs, positives only.",
    ),
    PairedRobustnessSlice(
        name="paws_labeled_final",
        family=PairedRobustnessFamily.PARAPHRASE,
        source_url="https://huggingface.co/datasets/google-research-datasets/paws",
        hf_dataset_id="google-research-datasets/paws",
        hf_dataset_name="labeled_final",
        split="test",
        source_field="sentence1",
        target_field="sentence2",
        source_label="sentence_1",
        target_label="sentence_2",
        label_field="label",
        required_label=1,
        max_pairs=DEFAULT_SAMPLE_CAP,
        notes="Held-out PAWS test paraphrase pairs, positives only.",
    ),
    PairedRobustnessSlice(
        name="flores_eng_deu",
        family=PairedRobustnessFamily.TRANSLATION,
        source_url="https://huggingface.co/datasets/facebook/flores",
        hf_dataset_id="facebook/flores",
        hf_dataset_name="eng_Latn-deu_Latn",
        split="dev",
        source_field="sentence_eng_Latn",
        target_field="sentence_deu_Latn",
        source_label="English",
        target_label="German",
        trust_remote_code=True,
        max_pairs=512,
        notes="FLORES-200 held-out dev translation pairs (en->de), capped sample.",
    ),
    PairedRobustnessSlice(
        name="flores_eng_deu",
        family=PairedRobustnessFamily.TRANSLATION,
        source_url="https://huggingface.co/datasets/facebook/flores",
        hf_dataset_id="facebook/flores",
        hf_dataset_name="eng_Latn-deu_Latn",
        split="devtest",
        source_field="sentence_eng_Latn",
        target_field="sentence_deu_Latn",
        source_label="English",
        target_label="German",
        trust_remote_code=True,
        max_pairs=512,
        notes="FLORES-200 held-out devtest translation pairs (en->de), capped sample.",
    ),
)

PAIRED_ROBUSTNESS_REGISTRY: dict[str, PairedRobustnessSlice] = {
    slice_.raw_step_key: slice_ for slice_ in PAIRED_ROBUSTNESS_SLICES
}


@dataclass(frozen=True)
class PairedRobustnessMaterializeConfig:
    """Config for materializing one paired robustness slice into `{text: ...}` records."""

    name: str
    family: PairedRobustnessFamily
    source_url: str
    hf_dataset_id: str
    hf_dataset_name: str | None
    split: str
    source_field: str
    target_field: str
    source_label: str
    target_label: str
    max_pairs: int
    trust_remote_code: bool = False
    label_field: str | None = None
    required_label: int | None = None
    notes: str = ""
    output_path: str = field(default_factory=this_output_path)  # type: ignore[arg-type]
    shard_size: int = DEFAULT_SHARD_SIZE


def paired_robustness_slices(*, family: PairedRobustnessFamily | None = None) -> tuple[PairedRobustnessSlice, ...]:
    if family is None:
        return PAIRED_ROBUSTNESS_SLICES
    return tuple(slice_ for slice_ in PAIRED_ROBUSTNESS_SLICES if slice_.family == family)


def linearized_text_views_for_example(
    slice_: PairedRobustnessSlice, example: Mapping[str, object]
) -> dict[PairedTextView, str] | None:
    if slice_.label_field is not None:
        label_value = example.get(slice_.label_field)
        if label_value != slice_.required_label:
            return None

    source_text = _field_as_text(example, slice_.source_field)
    target_text = _field_as_text(example, slice_.target_field)

    return {
        PairedTextView.SOURCE: f"{slice_.source_label}: {source_text}",
        PairedTextView.TARGET: f"{slice_.target_label}: {target_text}",
        PairedTextView.TARGET_GIVEN_SOURCE: (
            f"{slice_.source_label}: {source_text}\n{slice_.target_label}: {target_text}"
        ),
    }


def materialize_paired_robustness_slice(config: PairedRobustnessMaterializeConfig) -> None:
    slice_ = _slice_from_config(config)
    dataset = datasets.load_dataset(
        path=config.hf_dataset_id,
        name=config.hf_dataset_name,
        split=config.split,
        streaming=True,
        trust_remote_code=config.trust_remote_code,
    )

    buffers = {view: [] for view in ALL_PAIRED_TEXT_VIEWS}
    shard_index = {view: 0 for view in ALL_PAIRED_TEXT_VIEWS}
    counts = {view: 0 for view in ALL_PAIRED_TEXT_VIEWS}
    kept_pairs = 0

    for example in dataset:
        views = linearized_text_views_for_example(slice_, example)
        if views is None:
            continue
        kept_pairs += 1
        for view, text in views.items():
            buffers[view].append({"text": text})
            if len(buffers[view]) >= config.shard_size:
                _flush_view_shard(config.output_path, view, shard_index, counts, buffers)
        if kept_pairs >= config.max_pairs:
            break

    for view in ALL_PAIRED_TEXT_VIEWS:
        _flush_view_shard(config.output_path, view, shard_index, counts, buffers)

    if kept_pairs == 0:
        raise ValueError(
            f"Slice {config.name}/{config.split} produced zero paired examples with the configured filters."
        )

    logger.info(
        "Materialized paired robustness slice %s/%s with %d pairs; source=%d target=%d target_given_source=%d",
        config.name,
        config.split,
        kept_pairs,
        counts[PairedTextView.SOURCE],
        counts[PairedTextView.TARGET],
        counts[PairedTextView.TARGET_GIVEN_SOURCE],
    )


def paired_robustness_raw_steps(
    *,
    slices: Sequence[PairedRobustnessSlice] = PAIRED_ROBUSTNESS_SLICES,
    name_prefix: str = "raw/paired_robustness_ppl",
    resources: ResourceConfig | None = None,
) -> dict[str, ExecutorStep]:
    if resources is None:
        resources = ResourceConfig.with_cpu(cpu=4, ram="16g", disk="30g")

    steps: dict[str, ExecutorStep] = {}
    for slice_ in slices:
        steps[slice_.raw_step_key] = ExecutorStep(
            name=posixpath.join(name_prefix, slice_.family.value, slice_.name, slice_.split),
            description=f"Materialize capped held-out paired robustness eval records for {slice_.name}/{slice_.split}.",
            fn=remote(materialize_paired_robustness_slice, resources=resources, pip_dependency_groups=["cpu"]),
            config=_materialize_config(slice_),
        )
    return steps


def paired_robustness_raw_validation_sets(
    *,
    slices: Sequence[PairedRobustnessSlice] = PAIRED_ROBUSTNESS_SLICES,
    include_views: Sequence[PairedTextView] = ALL_PAIRED_TEXT_VIEWS,
    raw_steps: Mapping[str, ExecutorStep] | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    if raw_steps is None:
        raw_steps = paired_robustness_raw_steps(slices=slices)

    selected_views = tuple(include_views)
    datasets_out: dict[str, RawTextEvaluationDataset] = {}
    for slice_ in slices:
        raw_step = raw_steps[slice_.raw_step_key]
        for view in selected_views:
            datasets_out[slice_.dataset_key(view)] = raw_text_dataset(
                raw_step.cd(posixpath.join(view.value, "shard-*.jsonl.gz")),
                tags=slice_.tags_for_view(view),
            )
    return datasets_out


def _materialize_config(slice_: PairedRobustnessSlice) -> PairedRobustnessMaterializeConfig:
    return PairedRobustnessMaterializeConfig(
        name=slice_.name,
        family=slice_.family,
        source_url=slice_.source_url,
        hf_dataset_id=slice_.hf_dataset_id,
        hf_dataset_name=slice_.hf_dataset_name,
        split=slice_.split,
        source_field=slice_.source_field,
        target_field=slice_.target_field,
        source_label=slice_.source_label,
        target_label=slice_.target_label,
        max_pairs=slice_.max_pairs,
        trust_remote_code=slice_.trust_remote_code,
        label_field=slice_.label_field,
        required_label=slice_.required_label,
        notes=slice_.notes,
    )


def _slice_from_config(config: PairedRobustnessMaterializeConfig) -> PairedRobustnessSlice:
    return PairedRobustnessSlice(
        name=config.name,
        family=config.family,
        source_url=config.source_url,
        hf_dataset_id=config.hf_dataset_id,
        hf_dataset_name=config.hf_dataset_name,
        split=config.split,
        source_field=config.source_field,
        target_field=config.target_field,
        source_label=config.source_label,
        target_label=config.target_label,
        max_pairs=config.max_pairs,
        trust_remote_code=config.trust_remote_code,
        label_field=config.label_field,
        required_label=config.required_label,
        notes=config.notes,
    )


def _field_as_text(example: Mapping[str, object], field_name: str) -> str:
    if field_name not in example:
        raise KeyError(f"Expected field {field_name!r} in paired robustness example.")
    value = example[field_name]
    if value is None:
        raise ValueError(f"Field {field_name!r} cannot be None for paired robustness linearization.")
    if isinstance(value, str):
        return value
    return str(value)


def _flush_view_shard(
    output_path: str,
    view: PairedTextView,
    shard_index: dict[PairedTextView, int],
    counts: dict[PairedTextView, int],
    buffers: dict[PairedTextView, list[dict[str, str]]],
) -> None:
    records = buffers[view]
    if not records:
        return
    shard_path = os.path.join(output_path, view.value, f"shard-{shard_index[view]:05d}.jsonl.gz")
    result = write_jsonl_file(records, shard_path)
    counts[view] += int(result["count"])
    shard_index[view] += 1
    buffers[view] = []
