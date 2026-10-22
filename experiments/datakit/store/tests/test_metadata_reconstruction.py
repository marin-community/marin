# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import sys
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit import partition_filename
from marin.datakit.normalize import generate_id
from zephyr.shard_keys import deterministic_hash

from experiments.datakit.scripts.metadata_release_trial import main, run_trial, verify_reconstruction
from experiments.datakit.scripts.metadata_source_replay import SourceReplay, run_source_replay


def test_reconstruction_requires_selected_occurrence_counts(tmp_path):
    text = "A repeated document"
    doc_id = generate_id(text)
    selected = tmp_path / "selected.parquet"
    replay = tmp_path / "replay.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": doc_id}, {"id": doc_id}]), selected)
    pq.write_table(pa.Table.from_pylist([{"id": doc_id, "text": text}]), replay)
    outputs = [{"path": str(selected)}]
    report = verify_reconstruction(outputs, [str(replay)], max_rows=64)
    assert report["status"] == "incomplete"
    assert report["missing_occurrences"] == 1
    pq.write_table(pa.Table.from_pylist([{"id": doc_id, "text": text}] * 2), replay)
    assert verify_reconstruction(outputs, [str(replay)], max_rows=64)["status"] == "passed"


def test_reconstruction_rejects_changed_text_with_unchanged_id(tmp_path):
    doc_id = generate_id("original text")
    selected = tmp_path / "selected.parquet"
    replay = tmp_path / "replay.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": doc_id}]), selected)
    pq.write_table(pa.Table.from_pylist([{"id": doc_id, "text": "changed text"}]), replay)
    with pytest.raises(ValueError, match="incorrect normalized-text hash"):
        verify_reconstruction([{"path": str(selected)}], [str(replay)], max_rows=64)


def test_trial_reports_unexamined_sources_as_incomplete(tmp_path):
    report = run_trial(
        {"expected_sources": ["not-sampled"], "recipe_commit": "test", "store_uri": "test", "shards": []},
        str(tmp_path),
        max_rows=64,
        replay_max_rows=64,
    )
    assert report["status"] == "incomplete"
    assert report["missing_sources"] == ["not-sampled"]


def test_trial_counts_reused_replay_once_across_source_shards(tmp_path):
    text = "Same document in two source shards"
    doc_id = generate_id(text)
    replay = tmp_path / "replay.parquet"
    pq.write_table(pa.table({"id": [doc_id], "text": [text]}), replay)
    entries = []
    for index in range(2):
        selection = {"marin_source": "source", "dedup": "", "bucket_edges": None}
        tables = {
            "normalized": {"id": [doc_id], "text": [text]},
            "decontam": {"id": [doc_id], "contaminated": [False]},
            "cluster": {"id": [doc_id], "cluster_40": [3]},
            "quality": {"id": [doc_id], "quality_bucket": [2], "score": [0.5]},
            "exact_dedup": {"id": pa.array([], type=pa.string())},
        }
        for name, columns in tables.items():
            path = tmp_path / f"{name}-{index}.parquet"
            pq.write_table(pa.table(columns), path)
            selection[name] = str(path)
        entries.append({"selection": selection, "replay_normalized": [str(replay)]})
    manifest = {
        "expected_sources": ["source"],
        "recipe_commit": "test",
        "store_uri": "test",
        "cluster_view": 40,
        "shards": entries,
    }
    report = run_trial(manifest, str(tmp_path / "out"), max_rows=64, replay_max_rows=64)
    assert report["status"] == "incomplete"
    assert report["reconstruction"]["source"]["missing_occurrences"] == 1
    pq.write_table(pa.table({"id": [doc_id] * 2, "text": [text] * 2}), replay)
    report = run_trial(manifest, str(tmp_path / "complete"), max_rows=64, replay_max_rows=64)
    assert report["status"] == "passed"
    assert report["reconstructed_sources"] == ["source"]
    assert report["reconstruction"]["source"]["selected_occurrences"] == 2


@pytest.mark.parametrize("input_format", ["jsonl.gz", "parquet"])
def test_real_input_replay_finds_repartitioned_normalized_hashes(tmp_path, input_format):
    rows = [{"content": "   "}, {"content": "A     B", "provider": 42}, {"content": "日本語"}]
    input_path = tmp_path / f"raw.{input_format}"
    if input_format == "parquet":
        pq.write_table(pa.Table.from_pylist(rows), input_path)
    else:
        with gzip.open(input_path, "wt") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    normalized_dir = tmp_path / "normalized" / "outputs" / "main"
    normalized_dir.mkdir(parents=True)
    expected_ids = [generate_id("A  B"), generate_id("日本語")]
    for partition in range(3):
        ids = sorted(doc_id for doc_id in expected_ids if deterministic_hash(doc_id) % 3 == partition)
        pq.write_table(
            pa.table({"id": pa.array(ids, type=pa.string())}), normalized_dir / partition_filename(partition, 3)
        )
    recipe = SourceReplay(
        source="fixture",
        input_paths=[str(input_path)],
        normalized_output=str(tmp_path / "normalized"),
        num_partitions=3,
        normalization_parameters={"text_field": "content", "id_field": "provider", "max_whitespace_run_chars": 2},
        input_provenance="fixture staged input",
        recipe_commit="fixture",
    )
    report = run_source_replay(recipe, max_rows=3, max_projected_bytes=1024 * 1024)
    assert report["status"] == "passed"
    assert report["matched_ids"] == 2
    assert [row["input_row"] for row in report["samples"]] == [1, 2]
    assert all("text" not in row for row in report["samples"])
    wrong_recipe = replace(
        recipe, normalization_parameters={**recipe.normalization_parameters, "max_whitespace_run_chars": 3}
    )
    report = run_source_replay(wrong_recipe, max_rows=3, max_projected_bytes=1024 * 1024)
    assert report["status"] == "incomplete"
    assert report["missing_ids"] == [generate_id("A   B")]


def test_trial_cli_persists_incomplete_report_in_new_directory(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "expected_sources": ["not-sampled"],
                "recipe_commit": "test",
                "store_uri": "test",
                "shards": [],
            }
        )
    )
    output = tmp_path / "new-output"
    monkeypatch.setattr(sys, "argv", ["metadata_release_trial", "--manifest", str(manifest), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "incomplete"
    assert report["missing_sources"] == ["not-sampled"]
