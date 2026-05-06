# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import posixpath

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset
from marin.execution.step_spec import StepSpec

from experiments.evals import raw_web_markup_ppl as raw_web_markup


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

    assert isinstance(raw_web_markup.raw_web_markup_hf_step, StepSpec)
    assert isinstance(raw_web_markup.svg_stack_raw, StepSpec)
    assert isinstance(raw_web_markup.textocr_raw, StepSpec)
    assert isinstance(raw_web_markup.ocr_vqa_raw, StepSpec)
    assert surface_by_name["textocr_ocr_strings"].dataset_id == "Yesianrohn/OCR-Data"
    assert surface_by_name["textocr_ocr_strings"].revision == raw_web_markup.TEXTOCR_REVISION
    assert surface_by_name["textocr_ocr_strings"].input_glob == "data/TextOCR-*.parquet"
    assert surface_by_name["textocr_ocr_strings"].license_note == "apache-2.0 on the Hugging Face dataset card."
    assert surface_by_name["textocr_ocr_strings"].max_rows == 2_000
    assert surface_by_name["ocr_vqa_ocr_tokens_validation"].join_separator == " "
    assert surface_by_name["ocr_vqa_ocr_info_json_validation"].fields == ("image_id", "ocr_info")


def test_raw_web_markup_surfaces_have_manifest_policy_and_fingerprints() -> None:
    surface_names = {surface.name for surface in raw_web_markup.RAW_WEB_MARKUP_HF_SURFACES}

    assert set(raw_web_markup.RAW_WEB_MARKUP_SOURCE_MANIFESTS) == surface_names
    assert set(raw_web_markup.RAW_WEB_MARKUP_CONTENT_FINGERPRINTS) == surface_names

    textocr_manifest = raw_web_markup.RAW_WEB_MARKUP_SOURCE_MANIFESTS["textocr_ocr_strings"]
    assert textocr_manifest.policy.eval_only
    assert textocr_manifest.source_metadata["hf_revision"] == raw_web_markup.TEXTOCR_REVISION
    assert textocr_manifest.sample_caps.max_examples == raw_web_markup.RAW_WEB_MARKUP_MAX_ROWS
    assert raw_web_markup.RAW_WEB_MARKUP_CONTENT_FINGERPRINTS["textocr_ocr_strings"] == textocr_manifest.fingerprint()
