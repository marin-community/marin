# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Registry helpers for raw web, markup, and image-text perplexity-gap slices.

The first non-empty slices use ``starvector/svg-stack`` directly from Hugging Face
so we can preserve exact SVG XML without adding a downloader.
"""

from collections.abc import Mapping

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec

RAW_WEB_MARKUP_PREFIX = "raw_web_markup"
RAW_WEB_MARKUP_ISSUE_TAG = "issue:5056"
SVG_STACK_DATASET = HfDatasetSpec(id="starvector/svg-stack")
SVG_TEXT_KEY = "Svg"


def raw_web_markup_slice_key(source: str, surface: str) -> str:
    """Return the unprefixed raw-web-markup slice key for a source and surface."""
    return f"{source}/{surface}"


def raw_web_markup_tags(
    *,
    source: str,
    surface: str,
    extra_tags: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return standard tags for a raw-web-markup evaluation slice."""
    return (
        RAW_WEB_MARKUP_PREFIX,
        RAW_WEB_MARKUP_ISSUE_TAG,
        f"source:{source}",
        f"surface:{surface}",
        *extra_tags,
    )


def _hf_raw_web_markup_dataset(
    hf_dataset: HfDatasetSpec,
    *,
    text_key: str,
    split: str,
    source: str,
    surface: str,
) -> RawTextEvaluationDataset:
    return raw_text_dataset(
        hf_dataset,
        text_key=text_key,
        split=split,
        tags=raw_web_markup_tags(source=source, surface=surface, extra_tags=(f"split:{split}",)),
    )


ACTIVE_RAW_WEB_MARKUP_DATASETS: dict[str, RawTextEvaluationDataset] = {
    raw_web_markup_slice_key("svg_stack", "svg_xml_val"): _hf_raw_web_markup_dataset(
        SVG_STACK_DATASET,
        text_key=SVG_TEXT_KEY,
        split="val",
        source="svg_stack",
        surface="svg_xml",
    ),
    raw_web_markup_slice_key("svg_stack", "svg_xml_test"): _hf_raw_web_markup_dataset(
        SVG_STACK_DATASET,
        text_key=SVG_TEXT_KEY,
        split="test",
        source="svg_stack",
        surface="svg_xml",
    ),
}


def prefixed_raw_web_markup_validation_sets(
    datasets: Mapping[str, RawTextEvaluationDataset],
) -> dict[str, RawTextEvaluationDataset]:
    """Prefix raw-web-markup slice names with ``raw_web_markup/``."""
    return {f"{RAW_WEB_MARKUP_PREFIX}/{slice_name}": dataset for slice_name, dataset in datasets.items()}


def raw_web_markup_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Raw web/markup evaluation slices keyed by ``raw_web_markup/<slice>``."""
    return prefixed_raw_web_markup_validation_sets(ACTIVE_RAW_WEB_MARKUP_DATASETS)
