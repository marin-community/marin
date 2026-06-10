# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.transform.huggingface.raw_text import (
    HfRawTextRenderMode,
    HfRawTextSurfaceConfig,
    materialize_hf_raw_text,
    render_hf_raw_text,
)


def _read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _manifest(surface: HfRawTextSurfaceConfig) -> IngestionSourceManifest:
    return IngestionSourceManifest(
        dataset_key=surface.dataset_id,
        slice_key=f"raw_web_markup/{surface.name}",
        source_label=surface.name,
        source_urls=(f"https://huggingface.co/datasets/{surface.dataset_id}",),
        source_license="apache-2.0",
        source_format="huggingface_parquet",
        surface_form=surface.render_mode.value,
        policy=IngestionPolicy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only fixture.",
            requires_sanitization=False,
            identity_treatment=IdentityTreatment.PRESERVE,
            secret_redaction=SecretRedaction.NONE,
            contamination_risk="fixture",
            provenance_notes="fixture",
        ),
        staging=StagingMetadata(
            transform_name="materialize_hf_raw_text",
            serializer_name=surface.render_mode.value,
            split=surface.split,
            subset=surface.config_name,
            metadata={"input_glob": surface.input_glob, "output_filename": surface.output_filename},
        ),
        sample_caps=SampleCapConfig(max_examples=surface.max_rows),
        source_metadata={"hf_revision": surface.revision},
    )


def _materialize_kwargs(raw_path: Path, surface: HfRawTextSurfaceConfig) -> dict:
    manifest = _manifest(surface)
    return {
        "input_paths": {surface.dataset_id: str(raw_path)},
        "source_manifests": {surface.name: manifest},
        "content_fingerprints": {surface.name: manifest.fingerprint()},
    }


def test_render_hf_raw_text_modes() -> None:
    row = {
        "Svg": "<svg><text>Hello</text></svg>",
        "ocr_tokens": ["Hello", "world"],
        "metadata": {"title": "Book", "answers": ["A", "B"]},
    }

    assert (
        render_hf_raw_text(
            row,
            HfRawTextSurfaceConfig(
                name="svg",
                dataset_id="test/svg",
                revision="abc123",
                config_name="default",
                split="test",
                input_glob="data/test-*.parquet",
                output_filename="svg.jsonl.gz",
                render_mode=HfRawTextRenderMode.STRING_FIELD,
                field="Svg",
            ),
        )
        == "<svg><text>Hello</text></svg>"
    )
    assert (
        render_hf_raw_text(
            row,
            HfRawTextSurfaceConfig(
                name="ocr",
                dataset_id="test/ocr",
                revision="abc123",
                config_name="default",
                split="test",
                input_glob="data/test-*.parquet",
                output_filename="ocr.jsonl.gz",
                render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
                field="ocr_tokens",
            ),
        )
        == "Hello\nworld"
    )
    assert json.loads(
        render_hf_raw_text(
            row,
            HfRawTextSurfaceConfig(
                name="json",
                dataset_id="test/json",
                revision="abc123",
                config_name="default",
                split="test",
                input_glob="data/test-*.parquet",
                output_filename="json.jsonl.gz",
                render_mode=HfRawTextRenderMode.JSON_FIELDS,
                fields=("metadata.title", "metadata.answers"),
            ),
        )
    ) == {"metadata.answers": ["A", "B"], "metadata.title": "Book"}


def test_materialize_hf_raw_text_reads_pinned_parquet_and_writes_ingestion_metadata(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw"
    _write_parquet(
        raw_path / "data" / "TextOCR-00000-of-00001.parquet",
        [
            {"texts": ["Alpha", "Beta"], "bboxes": [[0.0, 1.0, 2.0, 3.0]], "num_text_regions": 2},
            {"texts": ["Gamma"], "bboxes": [[4.0, 5.0, 6.0, 7.0]], "num_text_regions": 1},
        ],
    )
    surface = HfRawTextSurfaceConfig(
        name="textocr_ocr_strings",
        dataset_id="test/textocr",
        revision="abc123",
        config_name="default",
        split="TextOCR",
        input_glob="data/TextOCR-*.parquet",
        output_filename="textocr/ocr_strings.jsonl.gz",
        render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
        field="texts",
        max_rows=2,
        license_note="apache-2.0",
    )

    result = materialize_hf_raw_text(
        output_path=str(tmp_path / "out"),
        surfaces=(surface,),
        **_materialize_kwargs(raw_path, surface),
    )

    records = _read_jsonl_gz(tmp_path / "out" / "textocr" / "ocr_strings.jsonl.gz")
    assert [record["text"] for record in records] == ["Alpha\nBeta", "Gamma"]
    assert records[0]["metadata"] == {
        "config": "default",
        "split": "TextOCR",
        "row_idx": 0,
        "surface": "textocr_ocr_strings",
        "hf_revision": "abc123",
        "input_glob": "data/TextOCR-*.parquet",
    }

    metadata = json.loads((tmp_path / "out" / "textocr" / "ocr_strings.metadata.json").read_text())
    assert metadata["source_manifest"]["policy"]["eval_only"] is True
    assert metadata["source_manifest"]["source_metadata"]["hf_revision"] == "abc123"
    assert metadata["materialized_output"]["record_count"] == 2
    assert result["surfaces"][0]["metadata_file"] == str(tmp_path / "out" / "textocr" / "ocr_strings.metadata.json")


def test_materialize_hf_raw_text_missing_fingerprint_fails(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw"
    _write_parquet(
        raw_path / "data" / "TextOCR-00000-of-00001.parquet",
        [{"texts": ["Alpha", "Beta"], "bboxes": [[0.0, 1.0, 2.0, 3.0]], "num_text_regions": 2}],
    )
    surface = HfRawTextSurfaceConfig(
        name="textocr_ocr_strings",
        dataset_id="test/textocr",
        revision="abc123",
        config_name="default",
        split="TextOCR",
        input_glob="data/TextOCR-*.parquet",
        output_filename="textocr/ocr_strings.jsonl.gz",
        render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
        field="texts",
        max_rows=2,
    )
    manifest = _manifest(surface)

    with pytest.raises(Exception, match="content_fingerprint mismatch"):
        materialize_hf_raw_text(
            input_paths={surface.dataset_id: str(raw_path)},
            output_path=str(tmp_path / "out"),
            surfaces=(surface,),
            source_manifests={surface.name: manifest},
            content_fingerprints={surface.name: "stale"},
        )


def test_materialize_hf_raw_text_skip_existing_preserves_record_count(tmp_path: Path, write_jsonl_gz) -> None:
    output_file = tmp_path / "textocr" / "ocr_strings.jsonl.gz"
    write_jsonl_gz(
        output_file,
        [
            {"id": "textocr_ocr_strings:0", "text": "Alpha", "source": "test/textocr", "metadata": {"row_idx": 0}},
            {"id": "textocr_ocr_strings:1", "text": "Beta", "source": "test/textocr", "metadata": {"row_idx": 1}},
        ],
    )
    raw_path = tmp_path / "raw"
    surface = HfRawTextSurfaceConfig(
        name="textocr_ocr_strings",
        dataset_id="test/textocr",
        revision="abc123",
        config_name="default",
        split="TextOCR",
        input_glob="data/TextOCR-*.parquet",
        output_filename="textocr/ocr_strings.jsonl.gz",
        render_mode=HfRawTextRenderMode.JOIN_LIST_FIELD,
        field="texts",
        max_rows=2,
    )
    result = materialize_hf_raw_text(
        output_path=str(tmp_path),
        surfaces=(surface,),
        **_materialize_kwargs(raw_path, surface),
    )

    metadata = json.loads((tmp_path / "textocr" / "ocr_strings.metadata.json").read_text())
    assert result["surfaces"] == [
        {
            "name": "textocr_ocr_strings",
            "records": 2,
            "output_file": str(output_file),
            "metadata_file": str(tmp_path / "textocr" / "ocr_strings.metadata.json"),
            "skipped": True,
            "underfilled": False,
        }
    ]
    assert metadata["materialized_output"]["record_count"] == 2
    assert metadata["materialized_output"]["metadata"]["skipped"] is True
