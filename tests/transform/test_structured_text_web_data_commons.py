# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for WebDataCommons WebTables sample staging."""

import gzip
import json
import tarfile
import zipfile
from io import BytesIO
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
from marin.transform.structured_text.web_data_commons import (
    WebDataCommonsStagingConfig,
    stage_web_data_commons_source,
)


def _read_staged_records(output_path: Path, filename: str = "staged.jsonl.gz") -> list[dict]:
    with gzip.open(output_path / filename, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _manifest(source_label: str, sample_name: str, sample_url: str) -> IngestionSourceManifest:
    return IngestionSourceManifest(
        dataset_key="webdatacommons/webtables",
        slice_key=f"structured_text/web_data_commons/{sample_name.lower()}",
        source_label=source_label,
        source_urls=(sample_url, "https://webdatacommons.org/webtables/englishTables.html"),
        source_license="research sample",
        source_format="zip_of_csv_tables",
        surface_form="byte_preserved_csv_plus_page_metadata",
        policy=IngestionPolicy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only structured-text probe.",
            requires_sanitization=False,
            identity_treatment=IdentityTreatment.PRESERVE,
            secret_redaction=SecretRedaction.NONE,
            contamination_risk="high: held-out eval contamination if reused in training",
            provenance_notes="Public sample archive.",
        ),
        staging=StagingMetadata(
            transform_name="stage_web_data_commons_source",
            serializer_name="csv",
            preserve_header=True,
            metadata={"sample_name": sample_name},
        ),
        sample_caps=SampleCapConfig(max_bytes_per_source=30 * 1024 * 1024, max_bytes_per_document=32 * 1024),
    )


def _write_direct_sample_zip(path: Path) -> None:
    metadata = {
        "arcfile": "common-crawl/parse-output/segment/example.arc.gz",
        "arcfilePosition": 22745259,
        "filename": "22745259_8925078337028777334",
        "uri": "https://example.org/table-source",
    }
    csv_text = '"country","value","missing"\n' '"france","3.14159265358979323846",""\n' '"japan","1e-30","NA"\n'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("sample10/22745259_0_4052170081208609462.csv", csv_text)
        archive.writestr("sample10/22745259_1580946025980243333.json", json.dumps(metadata))


def _write_nested_sample_zip(path: Path) -> None:
    csv_text = '"home","wallpapers-screensavers","arabische\r        les"\n"aswan","0.00001","NA"\n'
    csv_bytes = gzip.compress(csv_text.encode("utf-8"))
    tar_bytes = BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        info = tarfile.TarInfo("93648336_0_9026791465709724805.csv.gz")
        info.size = len(csv_bytes)
        tar.addfile(info, BytesIO(csv_bytes))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "common-crawl_parse-output_segment_1346823845675_1346864613682_245.arc.gz5422388171462217843.tar.gz.tar",
            tar_bytes.getvalue(),
        )


def test_stage_web_data_commons_preserves_direct_csv_and_metadata(tmp_path: Path):
    sample_zip = tmp_path / "sample10.zip"
    output_dir = tmp_path / "staged"
    _write_direct_sample_zip(sample_zip)
    manifest = _manifest("webdatacommons:webtables:sample10", "sample10", str(sample_zip))

    result = stage_web_data_commons_source(
        WebDataCommonsStagingConfig(
            sample_url=str(sample_zip),
            output_path=str(output_dir),
            source_label="webdatacommons:webtables:sample10",
            sample_name="sample10",
            source_manifest=manifest,
            content_fingerprint=manifest.fingerprint(),
        )
    )

    assert result["record_count"] == 1
    assert result["metadata_file"] == str(output_dir / "metadata.json")
    records = _read_staged_records(output_dir)
    record = records[0]
    assert record["text"] == (
        '"country","value","missing"\n' '"france","3.14159265358979323846",""\n' '"japan","1e-30","NA"\n'
    )
    assert record["provenance"]["sample_name"] == "sample10"
    assert record["provenance"]["uri"] == "https://example.org/table-source"
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_manifest"]["source_label"] == "webdatacommons:webtables:sample10"
    assert metadata["materialized_output"]["record_count"] == 1


def test_stage_web_data_commons_reads_nested_sample1k_archives(tmp_path: Path):
    sample_zip = tmp_path / "sample1K.zip"
    output_dir = tmp_path / "staged"
    _write_nested_sample_zip(sample_zip)

    result = stage_web_data_commons_source(
        WebDataCommonsStagingConfig(
            sample_url=str(sample_zip),
            output_path=str(output_dir),
            source_label="webdatacommons:webtables:sample1k",
            sample_name="sample1K",
        )
    )

    assert result["record_count"] == 1
    records = _read_staged_records(output_dir)
    assert records[0]["text"] == '"home","wallpapers-screensavers","arabische\r        les"\n"aswan","0.00001","NA"\n'
    assert records[0]["provenance"]["zip_member"].endswith(".tar.gz.tar")
    assert records[0]["provenance"]["table_member"] == "93648336_0_9026791465709724805.csv.gz"


def test_stage_web_data_commons_respects_text_byte_cap(tmp_path: Path):
    sample_zip = tmp_path / "sample10.zip"
    output_dir = tmp_path / "staged"
    with zipfile.ZipFile(sample_zip, "w") as archive:
        for index in range(10):
            archive.writestr(f"sample10/{index:08d}_0_table.csv", f"a,b\n{index},{'x' * 100}\n")

    result = stage_web_data_commons_source(
        WebDataCommonsStagingConfig(
            sample_url=str(sample_zip),
            output_path=str(output_dir),
            source_label="webdatacommons:webtables:sample10",
            sample_name="sample10",
            max_bytes_per_source=250,
        )
    )

    assert 0 < result["record_count"] < 10
    assert result["text_bytes_written"] <= 250 + 120


def test_stage_web_data_commons_rejects_zip_above_download_cap(tmp_path: Path):
    sample_zip = tmp_path / "sample10.zip"
    _write_direct_sample_zip(sample_zip)

    with pytest.raises(ValueError, match="exceeds max_zip_bytes"):
        stage_web_data_commons_source(
            WebDataCommonsStagingConfig(
                sample_url=str(sample_zip),
                output_path=str(tmp_path / "staged"),
                source_label="webdatacommons:webtables:sample10",
                sample_name="sample10",
                max_zip_bytes=32,
            )
        )
