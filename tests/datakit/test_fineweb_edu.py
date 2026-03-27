# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FineWeb-Edu canonical pipeline."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.v1.job import create_job_ctx, fray_default_job_ctx

from marin.datakit.canonical.fineweb_edu import (
    HF_DATASET_ID,
    HF_REVISION,
    download,
    normalize,
)
from marin.datakit.normalize import generate_id, normalize_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec


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
    assert dl.hash_attrs["hf_dataset_id"] == HF_DATASET_ID
    assert dl.hash_attrs["revision"] == HF_REVISION


def test_normalize_step_identity():
    """Normalize step has correct name and depends on download."""
    step = normalize(subset="data")
    assert step.name == "fineweb-edu/data/normalize"
    assert step.hash_attrs["text_field"] == "text"
    assert step.hash_attrs["id_field"] == "id"  # default


def test_normalize_step_subset_naming():
    """Different subsets produce distinct step names."""
    data_step = normalize(subset="data")
    sample_step = normalize(subset="sample/10BT")
    assert data_step.name == "fineweb-edu/data/normalize"
    assert sample_step.name == "fineweb-edu/sample/10BT/normalize"
    assert data_step.output_path != sample_step.output_path


def test_normalize_end_to_end(tmp_path: Path):
    """Download → normalize as a StepSpec DAG via StepRunner."""
    raw_dir = tmp_path / "raw"

    # Simulate a pre-existing download by writing FineWeb-Edu parquet locally
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
    _write_fineweb_edu_parquet(raw_dir / "train-00000.parquet", records)

    # Wire as StepSpec DAG: fake download (no-op) → normalize
    fake_download = StepSpec(
        name="test/raw/fineweb-edu",
        fn=lambda output_path: None,  # data already written
        override_output_path=str(raw_dir),
    )

    norm = normalize_step(
        name="test/fineweb-edu/normalize",
        download=fake_download,
        override_output_path=str(tmp_path / "normalized"),
    )

    StepRunner().run([fake_download, norm])

    # Verify normalized output
    output_dir = Path(norm.output_path)
    results = []
    for pf in sorted(output_dir.glob("**/*.parquet")):
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
