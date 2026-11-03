# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import generate_id

from experiments.datakit.scripts.metadata_cluster_external_completion import complete_common_corpus_source
from experiments.datakit.scripts.metadata_cluster_external_reconstruction import (
    aggregate_common_corpus_scan,
    common_corpus_recipe,
    reconstruct_common_corpus_selection,
    run_common_corpus_scan,
)
from experiments.datakit.scripts.metadata_cluster_reconstruction import aggregate_reconstruction
from experiments.datakit.scripts.metadata_cluster_selection import _digest, _write_json
from experiments.datakit.store.metadata import METADATA_SCHEMA


def test_original_common_corpus_scan_preserves_multiple_texts_per_identifier_and_selected_multiplicity(tmp_path):
    files = []
    for index, rows in enumerate(
        [
            [
                {"identifier": "patent", "text": "first patent", "language": "English", "open_type": "Open Government"},
                {"identifier": "patent", "text": "second patent", "language": "English", "open_type": "Open Government"},
                {"identifier": "patent", "text": "first patent", "language": "English", "open_type": "Closed"},
            ],
            [
                {"identifier": "patent", "text": "first patent", "language": "English", "open_type": "Open Culture"},
                {"identifier": "patent", "text": "first patent", "language": "French", "open_type": "Open Culture"},
            ],
        ]
    ):
        path = str(tmp_path / f"provider-{index}.parquet")
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
        files.append(path)
    manifest = {"files": files, "hf_dataset_id": "PleIAs/common_corpus", "revision": "pinned"}
    selected = [
        {
            "id": generate_id(text),
            "source_id": json.dumps("patent"),
            "normalized_shard": "normalized.parquet",
            "normalized_row": index,
        }
        for index, text in enumerate(["first patent", "second patent", "first patent"])
    ]
    reports = [
        run_common_corpus_scan(
            manifest, selected, str(tmp_path / "scan"), task_index=index, task_count=2, workers=1, batch_size=1
        )
        for index in range(2)
    ]
    summary = aggregate_common_corpus_scan(manifest, selected, reports)
    assert summary["complete_original_file_coverage"] and summary["original_files"] == 2
    assert summary["original_rows"] == 5
    assert summary["matched_selected_occurrences"] == 3
    assert summary["ambiguous_provider_targets"] == 1
    with pytest.raises(ValueError, match="cover every task"):
        aggregate_common_corpus_scan(manifest, selected, reports[:1])
    reconstructed = reconstruct_common_corpus_selection(manifest, selected, reports, str(tmp_path / "reconstructed"))
    rows = pq.ParquetFile(reconstructed["output"]).read().to_pylist()
    assert [(row["id"], row["text"], row["normalized_row"]) for row in rows] == [
        (generate_id(text), text, index) for index, text in enumerate(["first patent", "second patent", "first patent"])
    ]
    assert [row["matching_provider_rows"] for row in rows] == [2, 1, 2]
    resumed = run_common_corpus_scan(
        manifest, selected, str(tmp_path / "scan"), task_index=0, task_count=2, workers=1, batch_size=1
    )
    assert resumed["completed"] == reports[0]["completed"]

    ledger_path = str(tmp_path / "selected.parquet")
    ledger_rows = [{**row, "marin_source": "common_corpus/english", "cluster_id": 22} for row in selected]
    pq.write_table(pa.Table.from_pylist(ledger_rows, schema=METADATA_SCHEMA), ledger_path)
    digest = hashlib.sha256()
    for row in selected:
        digest.update(json.dumps([row["id"], row["normalized_row"]], separators=(",", ":")).encode() + b"\n")
    ledger_marker = {
        "manifest": {"marin_source": "common_corpus/english", "normalized": "normalized.parquet"},
        "target_cluster": 22,
        "selected_occurrences": 3,
        "ordered_occurrences_sha256": digest.hexdigest(),
        "output": ledger_path,
    }
    marker_path = str(tmp_path / "selected-complete.json")
    _write_json(marker_path, ledger_marker)
    source = {
        "marin_source": "common_corpus/english",
        "blocked_reason": None,
        "provider_paths": files,
        "recipe": json.loads(json.dumps(dataclasses.asdict(common_corpus_recipe()))),
        "selected_occurrences": 3,
        "ledgers": [
            {
                "completion_marker": marker_path,
                **{key: value for key, value in ledger_marker.items() if key != "manifest"},
            }
        ],
    }
    source["manifest_sha256"] = _digest(source)
    plan = {
        "sources": [source],
        "source_roots": {"common_corpus/english": str(tmp_path / "source-completion")},
        "target_cluster": 22,
        "selected_occurrences": 3,
        "output_prefix": str(tmp_path / "cluster-completion"),
    }
    plan["manifest_sha256"] = _digest(plan)
    completion = complete_common_corpus_source(plan, manifest, selected, reports)
    assert completion["reconstructed_occurrences"] == 3
    assert completion["historical_occurrence_provenance"] == "not_established"
    cluster = aggregate_reconstruction(plan)
    assert cluster["complete_cluster"] and cluster["reconstructed_occurrences"] == 3
