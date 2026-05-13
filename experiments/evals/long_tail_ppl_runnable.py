# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Runnable first-pass long-tail PPL slices backed by public Hugging Face datasets."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec

from experiments.evals.long_tail_ppl import LongTailPplFamily

RUNNABLE_LONG_TAIL_SOURCE_NOTE = (
    "These slices are directly executable from public Hugging Face datasets and do not require a bulk mirror."
)


@dataclass(frozen=True)
class RunnableLongTailPplSlice:
    """A runnable long-tail slice backed by a small public Hugging Face dataset."""

    name: str
    family: LongTailPplFamily
    source_url: str
    hf_dataset: HfDatasetSpec
    text_key: str
    split: str
    notes: str = ""

    @property
    def registry_key(self) -> str:
        return posixpath.join("long_tail_ppl_runnable", self.family.value, self.name)

    @property
    def tags(self) -> tuple[str, ...]:
        return ("long_tail_ppl", "long_tail_ppl_runnable", self.family.value, f"split:{self.split}")

    def to_raw_text_dataset(self) -> RawTextEvaluationDataset:
        return raw_text_dataset(self.hf_dataset, text_key=self.text_key, split=self.split, tags=self.tags)


RUNNABLE_LONG_TAIL_PPL_SLICES: tuple[RunnableLongTailPplSlice, ...] = (
    RunnableLongTailPplSlice(
        name="svg_stack_val",
        family=LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        source_url="https://huggingface.co/datasets/starvector/svg-stack",
        hf_dataset=HfDatasetSpec(id="starvector/svg-stack"),
        text_key="Svg",
        split="val",
        notes="Preserve SVG XML and caption-adjacent markup in the validation split.",
    ),
    RunnableLongTailPplSlice(
        name="svg_stack_test",
        family=LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        source_url="https://huggingface.co/datasets/starvector/svg-stack",
        hf_dataset=HfDatasetSpec(id="starvector/svg-stack"),
        text_key="Svg",
        split="test",
        notes="Preserve SVG XML in the held-out test split.",
    ),
    RunnableLongTailPplSlice(
        name="verilogeval_prompt",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        source_url="https://huggingface.co/datasets/dakies/nvlabs-verilogeval",
        hf_dataset=HfDatasetSpec(id="dakies/nvlabs-verilogeval"),
        text_key="prompt",
        split="test",
        notes="Keep VerilogEval problem statements and interface text intact.",
    ),
    RunnableLongTailPplSlice(
        name="verilogeval_canonical_solution",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        source_url="https://huggingface.co/datasets/dakies/nvlabs-verilogeval",
        hf_dataset=HfDatasetSpec(id="dakies/nvlabs-verilogeval"),
        text_key="canonical_solution",
        split="test",
        notes="Keep VerilogEval reference implementations and formatting intact.",
    ),
)

RUNNABLE_LONG_TAIL_PPL_REGISTRY: dict[str, RunnableLongTailPplSlice] = {
    slice_.registry_key: slice_ for slice_ in RUNNABLE_LONG_TAIL_PPL_SLICES
}


def runnable_long_tail_ppl_slices(*, family: LongTailPplFamily | None = None) -> tuple[RunnableLongTailPplSlice, ...]:
    if family is None:
        return RUNNABLE_LONG_TAIL_PPL_SLICES
    return tuple(slice_ for slice_ in RUNNABLE_LONG_TAIL_PPL_SLICES if slice_.family == family)


def runnable_long_tail_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Materialize the runnable HF-backed slices into raw-text datasets."""

    return {slice_.registry_key: slice_.to_raw_text_dataset() for slice_ in RUNNABLE_LONG_TAIL_PPL_SLICES}


def render_runnable_long_tail_registry_markdown() -> str:
    lines = ["# Runnable long-tail PPL registry", "", RUNNABLE_LONG_TAIL_SOURCE_NOTE, ""]
    for current_family in LongTailPplFamily:
        family_slices = runnable_long_tail_ppl_slices(family=current_family)
        if not family_slices:
            continue
        lines.append(f"## {current_family.value}")
        for slice_ in family_slices:
            lines.append(f"- `{slice_.registry_key}`: split={slice_.split} | {slice_.text_key} | {slice_.source_url}")
            if slice_.notes:
                lines.append(f"  - {slice_.notes}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
