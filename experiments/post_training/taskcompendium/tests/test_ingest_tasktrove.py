# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.taskcompendium import ingest_tasktrove

FIXTURE = Path(__file__).parents[4] / "lib/taskcompendium/tests/fixtures/tasktrove/mcq-1961bdb52b5a.tar.gz"
SOURCE = "laion__nemotron-gym-knowledge-mcqa-v2"
ARCHIVE_PATH = "Nemotron-RL-knowledge-mcqa-1961bdb52b5a.tar.gz"


def test_streaming_ingest_records_every_row_and_keeps_verifiers_private(tmp_path, monkeypatch):
    release = tmp_path / "release"
    (release / "tasks").mkdir(parents=True)
    (release / "sft").mkdir()
    (release / "manifest.json").write_text('{"tag":"2026.09.18.3"}')
    archive = FIXTURE.read_bytes()
    rows = [
        {
            "path": ARCHIVE_PATH,
            "source": SOURCE,
            "family": "qa-short-answer",
            "template_id": "c814af4f124d",
            "converter": "nemotron_mcqa",
            "mode": "mcq",
            "route": "rl",
            "tags": ["qa", "mcq", "nemotron"],
            "task_binary": archive,
        },
        {
            "path": "out-of-scope.tar.gz",
            "source": "other",
            "family": "other",
            "template_id": "other",
            "converter": "other",
            "mode": "judge",
            "route": "sft",
            "tags": [],
            "task_binary": b"must not be read by an importer",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), release / "tasks/part-00000.parquet")
    monkeypatch.setattr(ingest_tasktrove, "configure_coreweave_s3", lambda: None)
    monkeypatch.setenv("TASKTROVE_OUTPUT_URI", str(tmp_path / "durable-output"))

    report = ingest_tasktrove.ingest(str(release), tmp_path / "iris-output")

    assert report["input_rows_processed"] == 2
    assert report["counts"]["imported"] == 1
    assert report["counts"]["out_of_scope"] == 1
    assert report["accepted_counts"] == {
        "converter:nemotron_mcqa": 1,
        "family:qa-short-answer": 1,
        "mode:mcq": 1,
        "split:tasks": 1,
    }
    output = tmp_path / "durable-output"
    ledger = pq.read_table(output / "ingestion-ledger.parquet").to_pylist()
    catalog = pq.read_table(output / "private-catalog.parquet").to_pylist()
    assert [row["disposition"] for row in ledger] == ["imported", "out-of-scope"]
    assert len(catalog) == 1
    assert catalog[0]["tags"] == ["qa", "mcq", "nemotron"]
    assert "expected" in catalog[0]["specification_json"]
    assert json.loads(catalog[0]["source_metadata_json"])["source_dataset"] == "nvidia/Nemotron-RL-knowledge-mcqa"
    assert "#manifest-sha256=" in ledger[0]["input_object_pin"]
    assert ledger[1]["route"] == "sft"
    assert report["public_candidate_projection_materialized"] is True
    assert report["public_projection_published"] is False
    assert report["public_candidate_counts"] == {f"source:{SOURCE}": 1, "mode:mcq": 1}
    candidate = json.loads((output / "public-candidates.jsonl").read_text().splitlines()[0])
    assert set(candidate) == set(report["public_allowlist_fields"])
    assert "expected" not in json.dumps(candidate)
    assert (tmp_path / "iris-output/ingestion-summary.json").exists()
