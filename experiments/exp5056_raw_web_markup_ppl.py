# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Registry helpers for raw web, markup, and image-text perplexity-gap slices.

This module stays opt-in: call ``raw_web_markup_raw_validation_sets()`` explicitly
from a pilot gap experiment instead of extending ``default_raw_validation_sets()``.
The first non-empty slices use ``starvector/svg-stack`` directly from Hugging Face
so we can preserve exact SVG XML without adding a downloader.
"""

from collections.abc import Mapping

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec

RAW_WEB_MARKUP_PREFIX = "raw_web_markup"
RAW_WEB_MARKUP_ISSUE_TAG = "issue:5056"
SVG_STACK_DATASET = HfDatasetSpec(id="starvector/svg-stack")
SVG_STACK_SOURCE_TAG = "source:svg_stack"
SVG_XML_SURFACE_TAG = "surface:svg_xml"
SVG_TEXT_KEY = "Svg"


def _slice_key(source: str, surface: str) -> str:
    return f"{source}/{surface}"


def _hf_raw_web_markup_dataset(
    hf_dataset: HfDatasetSpec,
    *,
    text_key: str,
    split: str,
    source_tag: str,
    surface_tag: str,
) -> RawTextEvaluationDataset:
    return raw_text_dataset(
        hf_dataset,
        text_key=text_key,
        split=split,
        tags=(RAW_WEB_MARKUP_PREFIX, RAW_WEB_MARKUP_ISSUE_TAG, source_tag, surface_tag, f"split:{split}"),
    )


ACTIVE_RAW_WEB_MARKUP_DATASETS: dict[str, RawTextEvaluationDataset] = {
    _slice_key("svg_stack", "svg_xml_val"): _hf_raw_web_markup_dataset(
        SVG_STACK_DATASET,
        text_key=SVG_TEXT_KEY,
        split="val",
        source_tag=SVG_STACK_SOURCE_TAG,
        surface_tag=SVG_XML_SURFACE_TAG,
    ),
    _slice_key("svg_stack", "svg_xml_test"): _hf_raw_web_markup_dataset(
        SVG_STACK_DATASET,
        text_key=SVG_TEXT_KEY,
        split="test",
        source_tag=SVG_STACK_SOURCE_TAG,
        surface_tag=SVG_XML_SURFACE_TAG,
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
