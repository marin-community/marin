# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import posixpath

from experiments import exp5056_raw_web_markup_ppl as raw_web_markup
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset


def test_prefixed_raw_web_markup_validation_sets_prefixes_each_slice() -> None:
    dataset = RawTextEvaluationDataset(input_path="raw/web/svg.jsonl.gz")

    prefixed = raw_web_markup.prefixed_raw_web_markup_validation_sets({"svg_stack/svg_xml": dataset})

    assert prefixed == {
        posixpath.join(raw_web_markup.RAW_WEB_MARKUP_PREFIX, "svg_stack", "svg_xml"): dataset,
    }


def test_raw_web_markup_raw_validation_sets_registers_hf_materialized_slices() -> None:
    datasets = raw_web_markup.raw_web_markup_raw_validation_sets()

    expected_keys = {
        "raw_web_markup/svg_stack/svg_xml_val",
        "raw_web_markup/svg_stack/svg_xml_test",
        "raw_web_markup/textocr/ocr_strings",
        "raw_web_markup/textocr/annotations_json",
        "raw_web_markup/ocr_vqa/ocr_tokens_validation",
        "raw_web_markup/ocr_vqa/question_context_validation",
        "raw_web_markup/ocr_vqa/book_metadata_validation",
        "raw_web_markup/ocr_vqa/ocr_info_json_validation",
    }

    assert set(datasets) == expected_keys
    textocr = datasets["raw_web_markup/textocr/ocr_strings"]
    assert textocr.text_key == "text"
    assert textocr.tags == (
        "raw_web_markup",
        "issue:5056",
        "source:textocr",
        "surface:ocr_strings",
        "split:TextOCR",
    )
    assert textocr.input_path.name == "textocr/ocr_strings.jsonl.gz"


def test_raw_web_markup_hf_surfaces_include_sampling_and_license_notes() -> None:
    surface_by_name = {surface.name: surface for surface in raw_web_markup.RAW_WEB_MARKUP_HF_SURFACES}

    assert surface_by_name["textocr_ocr_strings"].dataset_id == "Yesianrohn/OCR-Data"
    assert surface_by_name["textocr_ocr_strings"].license_note == "apache-2.0 on the Hugging Face dataset card."
    assert surface_by_name["textocr_ocr_strings"].max_rows == 2_000
    assert surface_by_name["ocr_vqa_ocr_tokens_validation"].join_separator == " "
    assert surface_by_name["ocr_vqa_ocr_info_json_validation"].fields == ("image_id", "ocr_info")
