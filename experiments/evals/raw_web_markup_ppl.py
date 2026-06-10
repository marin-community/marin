# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registry helpers for raw web, markup, and image-text PPL slices for #5056."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from dataclasses import asdict

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.step_spec import StepSpec
from marin.transform.huggingface.raw_text import (
    HfRawTextRenderMode,
    HfRawTextSurfaceConfig,
    materialize_hf_raw_text,
)

RAW_WEB_MARKUP_PREFIX = "raw_web_markup"
RAW_WEB_MARKUP_ISSUE_TAG = "issue:5056"
RAW_WEB_MARKUP_MAX_ROWS = 2_000
LONG_TAIL_PPL_EPIC_ISSUE = 5005
RAW_WEB_MARKUP_ISSUE = 5056

TEXTOCR_DATASET_ID = "Yesianrohn/OCR-Data"
TEXTOCR_REVISION = "2b1f8aab9fbba3b5be07e2cae9e3e9c43fe5487c"
OCR_VQA_DATASET_ID = "howard-hou/OCR-VQA"
OCR_VQA_REVISION = "88234cc092c5f6d199b5cf3b471e3b490a69c07b"
SVG_STACK_DATASET_ID = "starvector/svg-stack"
SVG_STACK_REVISION = "1d922ec145f5ab6e3ff0e874235aff0d8a9dec91"

SVG_STACK_LICENSE = "No license declared on the Hugging Face dataset card."
TEXTOCR_LICENSE = "apache-2.0 on the Hugging Face dataset card."
OCR_VQA_LICENSE = "No license declared on the Hugging Face dataset card."

svg_stack_raw = download_hf_step(
    "raw/raw_web_markup/starvector_svg_stack",
    hf_dataset_id=SVG_STACK_DATASET_ID,
    revision=SVG_STACK_REVISION,
    hf_urls_glob=["data/val-*.parquet", "data/test-*.parquet", "README.md"],
)

textocr_raw = download_hf_step(
    "raw/raw_web_markup/yesianrohn_ocr_data",
    hf_dataset_id=TEXTOCR_DATASET_ID,
    revision=TEXTOCR_REVISION,
    hf_urls_glob=["data/TextOCR-*.parquet", "README.md"],
)

ocr_vqa_raw = download_hf_step(
    "raw/raw_web_markup/howard_hou_ocr_vqa",
    hf_dataset_id=OCR_VQA_DATASET_ID,
    revision=OCR_VQA_REVISION,
    hf_urls_glob=["data/validation-*.parquet", "README.md"],
)

RAW_WEB_MARKUP_HF_DOWNLOADS: dict[str, StepSpec] = {
    SVG_STACK_DATASET_ID: svg_stack_raw,
    TEXTOCR_DATASET_ID: textocr_raw,
    OCR_VQA_DATASET_ID: ocr_vqa_raw,
}


def _surface_tags(source: str, surface: str, split: str) -> tuple[str, ...]:
    return (RAW_WEB_MARKUP_PREFIX, RAW_WEB_MARKUP_ISSUE_TAG, f"source:{source}", f"surface:{surface}", f"split:{split}")


RAW_WEB_MARKUP_HF_SURFACES: tuple[HfRawTextSurfaceConfig, ...] = (
    HfRawTextSurfaceConfig(
        name="svg_stack_svg_xml_val",
        dataset_id=SVG_STACK_DATASET_ID,
        revision=SVG_STACK_REVISION,
        config_name="default",
        split="val",
        input_glob="data/val-*.parquet",
        output_filename="svg_stack/svg_xml_val.jsonl.gz",
        render_mode=HfRawTextRenderMode.STRING_FIELD,
        field="Svg",
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/starvector/svg-stack",
        license_note=SVG_STACK_LICENSE,
    ),
    HfRawTextSurfaceConfig(
        name="svg_stack_svg_xml_test",
        dataset_id=SVG_STACK_DATASET_ID,
        revision=SVG_STACK_REVISION,
        config_name="default",
        split="test",
        input_glob="data/test-*.parquet",
        output_filename="svg_stack/svg_xml_test.jsonl.gz",
        render_mode=HfRawTextRenderMode.STRING_FIELD,
        field="Svg",
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/starvector/svg-stack",
        license_note=SVG_STACK_LICENSE,
    ),
    HfRawTextSurfaceConfig(
        name="textocr_ocr_strings",
        dataset_id=TEXTOCR_DATASET_ID,
        revision=TEXTOCR_REVISION,
        config_name="default",
        split="TextOCR",
        input_glob="data/TextOCR-*.parquet",
        output_filename="textocr/ocr_strings.jsonl.gz",
        render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
        field="texts",
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/Yesianrohn/OCR-Data",
        license_note=TEXTOCR_LICENSE,
    ),
    HfRawTextSurfaceConfig(
        name="textocr_annotations_json",
        dataset_id=TEXTOCR_DATASET_ID,
        revision=TEXTOCR_REVISION,
        config_name="default",
        split="TextOCR",
        input_glob="data/TextOCR-*.parquet",
        output_filename="textocr/annotations_json.jsonl.gz",
        render_mode=HfRawTextRenderMode.JSON_FIELDS,
        fields=("texts", "bboxes", "polygons", "num_text_regions"),
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/Yesianrohn/OCR-Data",
        license_note=TEXTOCR_LICENSE,
    ),
    HfRawTextSurfaceConfig(
        name="ocr_vqa_ocr_tokens_validation",
        dataset_id=OCR_VQA_DATASET_ID,
        revision=OCR_VQA_REVISION,
        config_name="default",
        split="validation",
        input_glob="data/validation-*.parquet",
        output_filename="ocr_vqa/ocr_tokens_validation.jsonl.gz",
        render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
        field="ocr_tokens",
        join_separator=" ",
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/howard-hou/OCR-VQA",
        license_note=OCR_VQA_LICENSE,
    ),
    HfRawTextSurfaceConfig(
        name="ocr_vqa_question_context_validation",
        dataset_id=OCR_VQA_DATASET_ID,
        revision=OCR_VQA_REVISION,
        config_name="default",
        split="validation",
        input_glob="data/validation-*.parquet",
        output_filename="ocr_vqa/question_context_validation.jsonl.gz",
        render_mode=HfRawTextRenderMode.JSON_FIELDS,
        fields=("image_id", "questions", "answers"),
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/howard-hou/OCR-VQA",
        license_note=OCR_VQA_LICENSE,
    ),
    HfRawTextSurfaceConfig(
        name="ocr_vqa_book_metadata_validation",
        dataset_id=OCR_VQA_DATASET_ID,
        revision=OCR_VQA_REVISION,
        config_name="default",
        split="validation",
        input_glob="data/validation-*.parquet",
        output_filename="ocr_vqa/book_metadata_validation.jsonl.gz",
        render_mode=HfRawTextRenderMode.JSON_FIELDS,
        fields=("image_id", "title", "authorName", "genre", "image_url", "set_name"),
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/howard-hou/OCR-VQA",
        license_note=OCR_VQA_LICENSE,
    ),
    HfRawTextSurfaceConfig(
        name="ocr_vqa_ocr_info_json_validation",
        dataset_id=OCR_VQA_DATASET_ID,
        revision=OCR_VQA_REVISION,
        config_name="default",
        split="validation",
        input_glob="data/validation-*.parquet",
        output_filename="ocr_vqa/ocr_info_json_validation.jsonl.gz",
        render_mode=HfRawTextRenderMode.JSON_FIELDS,
        fields=("image_id", "ocr_info"),
        max_rows=RAW_WEB_MARKUP_MAX_ROWS,
        source_url="https://huggingface.co/datasets/howard-hou/OCR-VQA",
        license_note=OCR_VQA_LICENSE,
    ),
)


def _eval_only_policy(provenance_notes: str) -> IngestionPolicy:
    return IngestionPolicy(
        usage_policy=UsagePolicy.EVAL_ONLY,
        use_policy="Eval-only raw web markup PPL probe. Do not mix into training without explicit follow-up review.",
        requires_sanitization=False,
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
        contamination_risk="high: raw held-out probe text would contaminate this eval if copied into training data",
        provenance_notes=provenance_notes,
    )


def _surface_manifest(surface: HfRawTextSurfaceConfig) -> IngestionSourceManifest:
    return IngestionSourceManifest(
        dataset_key=surface.dataset_id,
        slice_key=f"{RAW_WEB_MARKUP_PREFIX}/{surface.name}",
        source_label=surface.name,
        source_urls=(surface.source_url or f"https://huggingface.co/datasets/{surface.dataset_id}",),
        source_license=surface.license_note,
        source_format="huggingface_parquet",
        surface_form=surface.render_mode.value,
        policy=_eval_only_policy(
            "Public Hugging Face dataset staged from pinned revision "
            f"{surface.revision} for issue #{RAW_WEB_MARKUP_ISSUE}."
        ),
        staging=StagingMetadata(
            transform_name="materialize_hf_raw_text",
            serializer_name=surface.render_mode.value,
            split=surface.split,
            subset=surface.config_name,
            metadata={
                "input_glob": surface.input_glob,
                "output_filename": surface.output_filename,
                "field": surface.field,
                "fields": list(surface.fields),
                "join_separator": surface.join_separator,
            },
        ),
        epic_issue=LONG_TAIL_PPL_EPIC_ISSUE,
        issue_numbers=(RAW_WEB_MARKUP_ISSUE,),
        sample_caps=SampleCapConfig(max_examples=surface.max_rows),
        source_metadata={
            "hf_revision": surface.revision,
            "access": surface.access_note,
        },
    )


RAW_WEB_MARKUP_SOURCE_MANIFESTS: dict[str, IngestionSourceManifest] = {
    surface.name: _surface_manifest(surface) for surface in RAW_WEB_MARKUP_HF_SURFACES
}

RAW_WEB_MARKUP_CONTENT_FINGERPRINTS: dict[str, str] = {
    name: manifest.fingerprint() for name, manifest in RAW_WEB_MARKUP_SOURCE_MANIFESTS.items()
}

raw_web_markup_hf_step = StepSpec(
    name="raw/raw_web_markup/hf_image_text",
    deps=list(RAW_WEB_MARKUP_HF_DOWNLOADS.values()),
    fn=lambda output_path: materialize_hf_raw_text(
        input_paths={dataset_id: step.output_path for dataset_id, step in RAW_WEB_MARKUP_HF_DOWNLOADS.items()},
        output_path=output_path,
        surfaces=RAW_WEB_MARKUP_HF_SURFACES,
        source_manifests=RAW_WEB_MARKUP_SOURCE_MANIFESTS,
        content_fingerprints=RAW_WEB_MARKUP_CONTENT_FINGERPRINTS,
    ),
    hash_attrs={
        "surfaces": [asdict(surface) for surface in RAW_WEB_MARKUP_HF_SURFACES],
        "manifest_fingerprints": RAW_WEB_MARKUP_CONTENT_FINGERPRINTS,
        "raw_downloads": {
            "svg_stack": {"dataset_id": SVG_STACK_DATASET_ID, "revision": SVG_STACK_REVISION},
            "textocr": {"dataset_id": TEXTOCR_DATASET_ID, "revision": TEXTOCR_REVISION},
            "ocr_vqa": {"dataset_id": OCR_VQA_DATASET_ID, "revision": OCR_VQA_REVISION},
        },
        "skip_existing": True,
    },
)
raw_web_markup_hf = raw_web_markup_hf_step.as_executor_step()

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
