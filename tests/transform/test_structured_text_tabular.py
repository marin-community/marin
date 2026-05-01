# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

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
from marin.transform.structured_text.tabular import (
    TabularStagingConfig,
    chunk_lines_by_bytes,
    serialize_csv_document,
    stage_tabular_source,
)


def test_serialize_csv_document_preserves_precision_and_missing_values():
    header = "x,y,z\n"
    body = [
        "0.1,0.2,0.3\n",
        "3.14159265358979323846,,1e-30\n",
        '4,"with, comma"," spaced "\n',
    ]

    text = serialize_csv_document(header, body)

    assert text == header + "".join(body)
    assert "3.14159265358979323846" in text
    assert ",,1e-30" in text


def test_chunk_lines_by_bytes_reserves_header_budget_without_splitting_rows():
    header = "col_a,col_b\n"
    lines = ["1,2\n", "3,4\n", "x" * 80 + "\n", "5,6\n"]

    chunks = list(chunk_lines_by_bytes(lines, max_bytes_per_chunk=20, header_line=header))

    assert chunks[0] == ["1,2\n", "3,4\n"]
    assert chunks[1] == ["x" * 80 + "\n"]
    assert chunks[2] == ["5,6\n"]


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _read_staged_records(output_path: Path, filename: str = "staged.jsonl.gz") -> list[dict]:
    with gzip.open(output_path / filename, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _manifest(source_label: str = "test:csv") -> IngestionSourceManifest:
    return IngestionSourceManifest(
        dataset_key="tabular/test_csv",
        slice_key="structured_text/test_csv",
        source_label=source_label,
        source_urls=("https://example.com/test.csv",),
        source_license="test-only",
        source_format="raw_csv_files",
        surface_form="byte_preserved_csv_chunks",
        policy=IngestionPolicy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only probe.",
            requires_sanitization=False,
            identity_treatment=IdentityTreatment.PRESERVE,
            secret_redaction=SecretRedaction.NONE,
            contamination_risk="low: local test fixture",
            provenance_notes="Synthetic fixture.",
        ),
        staging=StagingMetadata(
            transform_name="stage_tabular_source",
            preserve_header=True,
            metadata={"output_filename": "staged.jsonl.gz"},
        ),
        sample_caps=SampleCapConfig(max_bytes_per_source=1024, max_bytes_per_document=1024),
    )


def test_stage_tabular_source_preserves_header_rows_and_metadata(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"
    content = "id;quality\n1;5\nid;quality\n2;\n3;3.14159265358979323846\n"
    _write_csv(input_dir / "wine.csv", content)
    manifest = _manifest("test:wine")

    result = stage_tabular_source(
        TabularStagingConfig(
            input_path=str(input_dir),
            output_path=str(output_dir),
            source_label="test:wine",
            file_extensions=(".csv",),
            output_filename="staged.jsonl.gz",
            source_manifest=manifest,
            content_fingerprint=manifest.fingerprint(),
        )
    )

    assert result["record_count"] == 1
    assert result["metadata_file"] == str(output_dir / "metadata.json")

    records = _read_staged_records(output_dir)
    assert records[0]["text"] == content
    assert "id;quality\n2;\n" in records[0]["text"]

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_manifest"]["source_label"] == "test:wine"
    assert metadata["materialized_output"]["metadata"]["source_file_count"] == 1


def test_stage_tabular_source_rejects_non_utf8_input(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"
    input_dir.mkdir()
    (input_dir / "bad.csv").write_bytes(b"id,value\n1,\xff\n")

    with pytest.raises(UnicodeDecodeError):
        stage_tabular_source(
            TabularStagingConfig(
                input_path=str(input_dir),
                output_path=str(output_dir),
                source_label="test:bad",
                file_extensions=(".csv",),
            )
        )


def test_stage_tabular_source_respects_per_source_cap(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"
    header = "a,b\n"
    for index in range(4):
        body = "".join(f"{row},{row}\n" for row in range(100))
        _write_csv(input_dir / f"{index:03d}.csv", header + body)

    result = stage_tabular_source(
        TabularStagingConfig(
            input_path=str(input_dir),
            output_path=str(output_dir),
            source_label="test:cap",
            file_extensions=(".csv",),
            max_bytes_per_source=800,
            max_bytes_per_document=400,
        )
    )

    assert result["record_count"] >= 1
    assert result["bytes_written"] <= 1200
