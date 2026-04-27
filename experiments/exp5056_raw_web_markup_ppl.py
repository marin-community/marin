# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in raw web, markup, and image-text perplexity-gap slices for #5056."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep, this_output_path
from marin.transform.huggingface.raw_text import (
    HfRawTextMaterializationConfig,
    HfRawTextRenderMode,
    HfRawTextSurfaceConfig,
    materialize_hf_raw_text,
)

RAW_WEB_MARKUP_PREFIX = "raw_web_markup"
RAW_WEB_MARKUP_ISSUE_TAG = "issue:5056"
RAW_WEB_MARKUP_MAX_ROWS = 2_000
TEXTOCR_DATASET_ID = "Yesianrohn/OCR-Data"
OCR_VQA_DATASET_ID = "howard-hou/OCR-VQA"
SVG_STACK_DATASET_ID = "starvector/svg-stack"


def _surface_tags(source: str, surface: str, split: str) -> tuple[str, ...]:
    return (RAW_WEB_MARKUP_PREFIX, RAW_WEB_MARKUP_ISSUE_TAG, f"source:{source}", f"surface:{surface}", f"split:{split}")


RAW_WEB_MARKUP_HF_SURFACES: tuple[HfRawTextSurfaceConfig, ...] = (
    HfRawTextSurfaceConfig(
        name="svg_stack_svg_xml_val",
        dataset_id=SVG_STACK_DATASET_ID,
        config_name="default",
        split="val",
        output_filename="svg_stack/svg_xml_val.jsonl.gz",
        render_mode=HfRawTextRenderMode.STRING_FIELD,
        field="Svg",
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/starvector/svg-stack",
        license_note="No license declared on the Hugging Face dataset card.",
    ),
    HfRawTextSurfaceConfig(
        name="svg_stack_svg_xml_test",
        dataset_id=SVG_STACK_DATASET_ID,
        config_name="default",
        split="test",
        output_filename="svg_stack/svg_xml_test.jsonl.gz",
        render_mode=HfRawTextRenderMode.STRING_FIELD,
        field="Svg",
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/starvector/svg-stack",
        license_note="No license declared on the Hugging Face dataset card.",
    ),
    HfRawTextSurfaceConfig(
        name="textocr_ocr_strings",
        dataset_id=TEXTOCR_DATASET_ID,
        config_name="default",
        split="TextOCR",
        output_filename="textocr/ocr_strings.jsonl.gz",
        render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
        field="texts",
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/Yesianrohn/OCR-Data",
        license_note="apache-2.0 on the Hugging Face dataset card.",
    ),
    HfRawTextSurfaceConfig(
        name="textocr_annotations_json",
        dataset_id=TEXTOCR_DATASET_ID,
        config_name="default",
        split="TextOCR",
        output_filename="textocr/annotations_json.jsonl.gz",
        render_mode=HfRawTextRenderMode.JSON_FIELDS,
        fields=("texts", "bboxes", "polygons", "num_text_regions"),
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/Yesianrohn/OCR-Data",
        license_note="apache-2.0 on the Hugging Face dataset card.",
    ),
    HfRawTextSurfaceConfig(
        name="ocr_vqa_ocr_tokens_validation",
        dataset_id=OCR_VQA_DATASET_ID,
        config_name="default",
        split="validation",
        output_filename="ocr_vqa/ocr_tokens_validation.jsonl.gz",
        render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
        field="ocr_tokens",
        join_separator=" ",
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/howard-hou/OCR-VQA",
        license_note="No license declared on the Hugging Face dataset card.",
    ),
    HfRawTextSurfaceConfig(
        name="ocr_vqa_question_context_validation",
        dataset_id=OCR_VQA_DATASET_ID,
        config_name="default",
        split="validation",
        output_filename="ocr_vqa/question_context_validation.jsonl.gz",
        render_mode=HfRawTextRenderMode.JSON_FIELDS,
        fields=("image_id", "questions", "answers"),
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/howard-hou/OCR-VQA",
        license_note="No license declared on the Hugging Face dataset card.",
    ),
    HfRawTextSurfaceConfig(
        name="ocr_vqa_book_metadata_validation",
        dataset_id=OCR_VQA_DATASET_ID,
        config_name="default",
        split="validation",
        output_filename="ocr_vqa/book_metadata_validation.jsonl.gz",
        render_mode=HfRawTextRenderMode.JSON_FIELDS,
        fields=("image_id", "title", "authorName", "genre", "image_url", "set_name"),
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/howard-hou/OCR-VQA",
        license_note="No license declared on the Hugging Face dataset card.",
    ),
    HfRawTextSurfaceConfig(
        name="ocr_vqa_ocr_info_json_validation",
        dataset_id=OCR_VQA_DATASET_ID,
        config_name="default",
        split="validation",
        output_filename="ocr_vqa/ocr_info_json_validation.jsonl.gz",
        render_mode=HfRawTextRenderMode.JSON_FIELDS,
        fields=("image_id", "ocr_info"),
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/howard-hou/OCR-VQA",
        license_note="No license declared on the Hugging Face dataset card.",
    ),
)

raw_web_markup_hf = ExecutorStep(
    name="raw/raw_web_markup/hf_image_text",
    fn=materialize_hf_raw_text,
    config=HfRawTextMaterializationConfig(
        surfaces=RAW_WEB_MARKUP_HF_SURFACES,
        output_path=this_output_path(),
    ),
)

ACTIVE_RAW_WEB_MARKUP_DATASETS: dict[str, RawTextEvaluationDataset] = {
    posixpath.join("svg_stack", "svg_xml_val"): raw_text_dataset(
        raw_web_markup_hf.cd("svg_stack/svg_xml_val.jsonl.gz"),
        tags=_surface_tags("svg_stack", "svg_xml", "val"),
    ),
    posixpath.join("svg_stack", "svg_xml_test"): raw_text_dataset(
        raw_web_markup_hf.cd("svg_stack/svg_xml_test.jsonl.gz"),
        tags=_surface_tags("svg_stack", "svg_xml", "test"),
    ),
    posixpath.join("textocr", "ocr_strings"): raw_text_dataset(
        raw_web_markup_hf.cd("textocr/ocr_strings.jsonl.gz"),
        tags=_surface_tags("textocr", "ocr_strings", "TextOCR"),
    ),
    posixpath.join("textocr", "annotations_json"): raw_text_dataset(
        raw_web_markup_hf.cd("textocr/annotations_json.jsonl.gz"),
        tags=_surface_tags("textocr", "annotations_json", "TextOCR"),
    ),
    posixpath.join("ocr_vqa", "ocr_tokens_validation"): raw_text_dataset(
        raw_web_markup_hf.cd("ocr_vqa/ocr_tokens_validation.jsonl.gz"),
        tags=_surface_tags("ocr_vqa", "ocr_tokens", "validation"),
    ),
    posixpath.join("ocr_vqa", "question_context_validation"): raw_text_dataset(
        raw_web_markup_hf.cd("ocr_vqa/question_context_validation.jsonl.gz"),
        tags=_surface_tags("ocr_vqa", "question_context", "validation"),
    ),
    posixpath.join("ocr_vqa", "book_metadata_validation"): raw_text_dataset(
        raw_web_markup_hf.cd("ocr_vqa/book_metadata_validation.jsonl.gz"),
        tags=_surface_tags("ocr_vqa", "book_metadata", "validation"),
    ),
    posixpath.join("ocr_vqa", "ocr_info_json_validation"): raw_text_dataset(
        raw_web_markup_hf.cd("ocr_vqa/ocr_info_json_validation.jsonl.gz"),
        tags=_surface_tags("ocr_vqa", "ocr_info_json", "validation"),
    ),
}


def prefixed_raw_web_markup_validation_sets(
    datasets: Mapping[str, RawTextEvaluationDataset],
) -> dict[str, RawTextEvaluationDataset]:
    """Prefix raw-web-markup slice names with ``raw_web_markup/``."""

    return {posixpath.join(RAW_WEB_MARKUP_PREFIX, slice_name): dataset for slice_name, dataset in datasets.items()}


def raw_web_markup_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Return raw-text eval slices keyed by ``raw_web_markup/<source>/<surface>``."""

    return prefixed_raw_web_markup_validation_sets(ACTIVE_RAW_WEB_MARKUP_DATASETS)
