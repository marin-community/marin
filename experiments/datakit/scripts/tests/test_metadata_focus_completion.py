# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import generate_id

from experiments.datakit.scripts.metadata_cluster_selection import _digest
from experiments.datakit.scripts.metadata_focus_completion import reconcile_focus_source
from experiments.datakit.scripts.metadata_focus_provider import (
    FOCUS_SOURCE,
    JUSTEXT_REVISION,
    MODEL_SETTINGS,
    RUNTIME_DEPENDENCIES,
    focus_provider_evidence,
)


def _fixture(tmp_path):
    targets = {}
    candidates = []
    for index, text in enumerate(["first original", "second original"]):
        doc_id = generate_id(text)
        filename = f"crawl/warc/{index}.warc.gz"
        target = {"normalized_id": doc_id, "record_id": f"uuid-{index}", "url": f"https://original/{index}"}
        targets[filename] = [target]
        candidates.append(
            {
                "id": doc_id,
                "text": text,
                "source_id": target["record_id"],
                "url": target["url"],
                "warc_filename": filename,
                "warc_bytes_sha256": hashlib.sha256(f"response-{index}".encode()).hexdigest(),
            }
        )
    ledger = tmp_path / "selected.parquet"
    rows = [
        {
            "id": candidate["id"],
            "source_id": json.dumps(candidate["source_id"]),
            "normalized_row": index,
            "normalized_shard": "normalized",
            "cluster_id": 22,
            "marin_source": FOCUS_SOURCE,
        }
        for index, candidate in enumerate([candidates[0], candidates[0], candidates[1]])
    ]
    pq.write_table(pa.Table.from_pylist(rows), ledger)
    ordered = hashlib.sha256()
    for row in rows:
        ordered.update(json.dumps([row["id"], row["normalized_row"]], separators=(",", ":")).encode() + b"\n")
    selected_marker = tmp_path / "selected-complete.json"
    selected_marker.write_text(
        json.dumps(
            {
                "manifest": {"marin_source": FOCUS_SOURCE, "normalized": "normalized"},
                "target_cluster": 22,
                "selected_occurrences": 3,
                "ordered_occurrences_sha256": ordered.hexdigest(),
                "output": str(ledger),
            }
        )
    )
    source = {
        "marin_source": FOCUS_SOURCE,
        "blocked_reason": None,
        "selected_occurrences": 3,
        "provider_paths": ["https://data.commoncrawl.org/" + name for name in sorted(targets)],
        "original_provider_evidence": focus_provider_evidence(targets),
        "ledgers": [
            {
                "output": str(ledger),
                "selected_occurrences": 3,
                "completion_marker": str(selected_marker),
                "ordered_occurrences_sha256": ordered.hexdigest(),
            }
        ],
    }
    source["manifest_sha256"] = _digest(source)
    plan: dict = {"sources": [source], "target_cluster": 22, "source_roots": {FOCUS_SOURCE: str(tmp_path / "source")}}
    plan["manifest_sha256"] = _digest(plan)
    replay = tmp_path / "replay"
    for index, candidate in enumerate(candidates):
        destination = replay / f"file={index:06d}"
        destination.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([candidate]), destination / "candidates.parquet")
        marker = {
            "binding": {
                "warc_filename": candidate["warc_filename"],
                "targets": targets[candidate["warc_filename"]],
                "justext_revision": JUSTEXT_REVISION,
            },
            "runtime_dependencies": RUNTIME_DEPENDENCIES,
            "model_settings": MODEL_SETTINGS,
            "unique_candidates": 1,
            "text_sha256": {candidate["id"]: hashlib.sha256(candidate["text"].encode()).hexdigest()},
        }
        (destination / "_complete.json").write_text(json.dumps(marker))
    return plan, targets, replay


def test_focus_completion_requires_all_original_warcs_and_expands_selected_occurrences(tmp_path):
    plan, targets, replay = _fixture(tmp_path)
    final_marker = replay / "file=000001/_complete.json"
    complete_bytes = final_marker.read_bytes()
    final_marker.unlink()
    with pytest.raises(FileNotFoundError):
        reconcile_focus_source(plan, targets, str(replay))
    root = Path(plan["source_roots"][FOCUS_SOURCE])
    assert not (root / "reconstructed/_complete.json").exists()
    final_marker.write_bytes(complete_bytes)
    report = reconcile_focus_source(plan, targets, str(replay))
    recovered = pq.read_table(report["outputs"][0]["path"]).to_pylist()
    assert sorted(row["text"] for row in recovered) == ["first original", "first original", "second original"]
    assert report["provider_files"] == 2 and report["reconstructed_occurrences"] == 3


def test_focus_completion_rechecks_canonical_text_after_original_replay(tmp_path):
    plan, targets, replay = _fixture(tmp_path)
    output = replay / "file=000000/candidates.parquet"
    table = pq.ParquetFile(output).read()
    table = table.set_column(table.schema.get_field_index("text"), "text", pa.array(["changed after WARC replay"]))
    pq.write_table(table, output)
    with pytest.raises(ValueError, match="canonical text"):
        reconcile_focus_source(plan, targets, str(replay))
    assert not (Path(plan["source_roots"][FOCUS_SOURCE]) / "reconstructed/_complete.json").exists()
