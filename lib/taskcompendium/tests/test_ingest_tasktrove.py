# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.taskcompendium import ingest_tasktrove

SOURCE = "synthetic__mcqa-demo"
ARCHIVE_PATH = "synthetic-mcqa.tar.gz"


def _synthetic_archive() -> bytes:
    manifest = f"""[metadata]
tasktrove_source = "{SOURCE}"
tasktrove_path = "{ARCHIVE_PATH}"
source_dataset = "synthetic/demo"
family = "qa-short-answer"
template_id = "c814af4f124d"
converter = "nemotron_mcqa"
mode = "mcq"
tags = ["qa", "mcq", "synthetic"]
"""
    files = {
        "task.toml": manifest.encode(),
        "instruction.md": (
            b"You are answering a multiple-choice question. Read the question below and write your final "
            b"answer to `/app/answer.txt`.\n\n"
            b"The verifier extracts a single letter (A/B/C/...) from your answer file using a regex pattern; "
            b"the simplest valid output is a file containing exactly\n`Answer: X` (where X is your chosen letter).\n\n"
            b"---\n\nAnswer the following multiple choice question. The last line of your response "
            b"should be in the following format: 'Answer: A/B/C/D/E' (e.g. 'Answer: D').\n\n"
            b"Synthetic question: Which label is correct?\nA: Alpha\nB: Bravo\nC: Charlie\nD: Delta\nE: Echo"
        ),
        "tests/verifier.toml": b'mode = "mcq"\nexpected = "D"\noptions = 5\n',
    }
    archive_data = io.BytesIO()
    with tarfile.open(fileobj=archive_data, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return archive_data.getvalue()


def test_streaming_ingest_records_every_row_and_keeps_verifiers_private(tmp_path, monkeypatch):
    release = tmp_path / "release"
    (release / "tasks").mkdir(parents=True)
    (release / "sft").mkdir()
    (release / "manifest.json").write_text('{"tag":"2026.09.18.3"}')
    rows = [
        {
            "path": ARCHIVE_PATH,
            "source": SOURCE,
            "family": "qa-short-answer",
            "template_id": "c814af4f124d",
            "converter": "nemotron_mcqa",
            "mode": "mcq",
            "route": "rl",
            "tags": ["qa", "mcq", "synthetic"],
            "task_binary": _synthetic_archive(),
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
        {
            "path": "malformed.tar.gz",
            "source": SOURCE,
            "family": "qa-short-answer",
            "template_id": "c814af4f124d",
            "converter": "nemotron_mcqa",
            "mode": "mcq",
            "route": "rl",
            "tags": ["qa", "mcq", "broken"],
            "task_binary": b"not a tar archive",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), release / "tasks/part-00000.parquet")
    monkeypatch.setattr(ingest_tasktrove, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(ingest_tasktrove, "PUBLIC_CANDIDATE_COHORTS", {SOURCE: "mcq"})
    report = ingest_tasktrove.ingest(str(release), str(tmp_path / "durable-output"), tmp_path / "iris-output")

    assert report["input_rows_processed"] == 3
    assert report["counts"]["imported"] == 1
    assert report["counts"]["out_of_scope"] == 1
    assert report["counts"]["rejected"] == 1
    assert report["accepted_counts"] == {
        "converter:nemotron_mcqa": 1,
        "family:qa-short-answer": 1,
        "mode:mcq": 1,
        "split:tasks": 1,
    }
    output = tmp_path / "durable-output"
    ledger = pq.read_table(output / "ingestion-ledger.parquet").to_pylist()
    catalog = pq.read_table(output / "private-catalog.parquet").to_pylist()
    assert [row["disposition"] for row in ledger] == ["imported", "out-of-scope", "rejected"]
    assert len(catalog) == 1
    assert catalog[0]["tags"] == ["qa", "mcq", "synthetic"]
    assert "expected" in catalog[0]["specification_json"]
    assert json.loads(catalog[0]["source_metadata_json"])["source_dataset"] == "synthetic/demo"
    assert "#manifest-sha256=" in ledger[0]["input_object_pin"]
    assert ledger[1]["route"] == "sft"
    assert ledger[0]["tags"] == ["qa", "mcq", "synthetic"]
    assert ledger[1]["tags"] == []
    assert ledger[2]["tags"] == ["qa", "mcq", "broken"]
    assert ledger[2]["mode"] == "mcq"
    assert "Invalid TaskTrove archive container" in ledger[2]["reason"]
    assert report["public_candidate_projection_materialized"] is True
    assert report["public_projection_published"] is False
    assert report["public_candidate_counts"] == {f"source:{SOURCE}": 1, "mode:mcq": 1}
    candidate = json.loads((output / "public-candidates.jsonl").read_text().splitlines()[0])
    assert set(candidate) == set(report["public_allowlist_fields"])
    assert candidate["record_version"] == 1
    proof = json.loads((output / "candidate-proof.jsonl").read_text().splitlines()[0])
    assert set(proof) == set(report["public_candidate_proof_fields"])
    assert proof["candidate_id"] == candidate["id"]
    assert proof["archive_sha256"] == catalog[0]["archive_sha256"]
    assert proof["disposition"] == "imported"
    assert report["public_candidate_proof_count"] == 1
    assert "expected" not in json.dumps(candidate)
    assert (tmp_path / "iris-output/ingestion-summary.json").exists()
