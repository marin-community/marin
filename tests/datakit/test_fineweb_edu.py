# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FineWeb-Edu canonical pipeline."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.v1.job import create_job_ctx, fray_default_job_ctx

from marin.datakit.canonical.fineweb_edu import (
    download,
    normalize,
)
from marin.datakit.normalize import generate_id, normalize_to_parquet


@pytest.fixture(autouse=True)
def flow_backend_ctx():
    with fray_default_job_ctx(create_job_ctx("sync")):
        yield


def _write_fineweb_edu_parquet(path: Path, records: list[dict]) -> None:
    """Write records matching the FineWeb-Edu schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, str(path))


def test_download_step_identity():
    """Download step has correct name, dataset ID, and revision."""
    dl = download()
    assert dl.name == "raw/fineweb-edu"
    assert dl.hash_attrs["hf_dataset_id"] == "HuggingFaceFW/fineweb-edu"
    assert dl.hash_attrs["revision"] == "87f0914"


def test_normalize_step_identity():
    """Normalize step has correct name and depends on download."""
    step = normalize(subset="data")
    assert step.name == "normalized/fineweb_edu"
    assert step.hash_attrs["text_field"] == "text"
    assert step.hash_attrs["id_field"] == "id"  # default


def test_normalize_step_subset_output_paths():
    """Different subsets produce distinct output paths (via hash_attrs)."""
    data_step = normalize(subset="data")
    sample_step = normalize(subset="sample/10BT")
    assert data_step.name == sample_step.name == "normalized/fineweb_edu"
    # input_path differs in hash_attrs, so output_path differs
    assert data_step.output_path != sample_step.output_path


def test_normalize_end_to_end(tmp_path: Path):
    """Normalize FineWeb-Edu parquet files end to end with local data."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Simulate FineWeb-Edu schema
    records = [
        {
            "id": "fineweb-edu-000001",
            "text": "Mathematics is the study of numbers.",
            "url": "https://example.com/math",
            "dump": "CC-MAIN-2024-10",
            "file_path": "data/train-00000.parquet",
            "language": "en",
            "language_score": 0.98,
            "token_count": 7,
            "score": 3.5,
            "int_score": 4,
        },
        {
            "id": "fineweb-edu-000002",
            "text": "Physics explores the fundamental forces of nature.",
            "url": "https://example.com/physics",
            "dump": "CC-MAIN-2024-10",
            "file_path": "data/train-00000.parquet",
            "language": "en",
            "language_score": 0.99,
            "token_count": 8,
            "score": 4.2,
            "int_score": 4,
        },
    ]
    _write_fineweb_edu_parquet(input_dir / "train-00000.parquet", records)

    normalize_to_parquet(
        input_path=str(input_dir),
        output_path=str(output_dir),
    )

    results = []
    for pf in sorted(output_dir.glob("*.parquet")):
        results.extend(pq.read_table(str(pf)).to_pylist())

    assert len(results) == 2

    for r in results:
        assert r["id"] == generate_id(r["text"])
        assert "source_id" in r
        # Extra FineWeb-Edu columns preserved
        assert "url" in r
        assert "dump" in r
        assert "language" in r
        assert "language_score" in r
        assert "token_count" in r
        assert "score" in r
        assert "int_score" in r

    source_ids = {r["source_id"] for r in results}
    assert source_ids == {"fineweb-edu-000001", "fineweb-edu-000002"}
