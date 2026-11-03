# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.scripts.metadata_cluster_census import inventory_shards, run_census
from experiments.datakit.scripts.metadata_cluster_selection import (
    aggregate_selection_reports,
    census_target_counts,
    run_selection_task,
    verify_selection_outputs,
)
from experiments.datakit.scripts.metadata_cluster_verification import (
    aggregate_verification_reports,
    verify_selection_task,
)


def test_cluster_selection_prunes_only_census_zero_shards_and_resumes(tmp_path):
    prefix = tmp_path / "marin"
    source_key = "normalized/books/outputs/main"
    filenames = [f"part-{index:05d}-of-00002.parquet" for index in range(2)]
    source = {
        "marin_source": "books",
        "source_key": source_key,
        "fuzzy_exempt": False,
        "artifact_roots": {"cluster": "cluster", "decontam": "decontam", "quality": "quality", "content_type": "types"},
        "cluster_shards": [str(prefix / "cluster" / filename) for filename in filenames],
    }
    inventory = {
        "store_uri": str(prefix / "datakit" / "store"),
        "store_config": {"cluster_view": 2},
        "bucket_edges": {"default": [0.5]},
        "sources": [source],
        "records": {
            "cluster": {"record": {"result": {"output_dir": "cluster"}}},
            "decontam": {"record": {"result": {"main_output_dir": "decontam/main"}}},
            "exact": {"record": {"result": {"sources": {source_key: {"attr_dir": "exact/books"}}}}},
            "fuzzy": {"record": {"result": {"sources": {source_key: {"attr_dir": "fuzzy/books"}}}}},
        },
    }
    first = filenames[0]
    ids = ["repeat", "repeat", "exact"]
    tables = {
        source_key: {"id": ids},
        "cluster": {"id": ids, "cluster_2": [1, 1, 1]},
        "decontam/main": {"id": ids, "contaminated": [False] * 3},
        "quality": {"id": ids, "score": [0.7] * 3},
        "types": {"id": ids, "content_type": ["prose"] * 3},
        "exact/books": {"id": ["exact"]},
    }
    for directory, data in tables.items():
        path = prefix / directory / first
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(data), path)
    pq.write_table(pa.table({"id": ["other"], "cluster_2": [0]}), prefix / "cluster" / filenames[1])
    shards = inventory_shards(inventory, None)
    census = run_census(
        shards,
        str(tmp_path / "census.json"),
        inventory_digest=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        cluster_view=2,
        task_index=0,
        task_count=1,
        workers=1,
        batch_size=1,
        checkpoint_interval=0,
        sampled=False,
    )
    counts = census_target_counts(inventory, inventory, [census], 1)
    output = str(tmp_path / "selection")
    options = dict(
        exact_catalog_root="exact",
        fuzzy_catalog_root="fuzzy",
        target_cluster=1,
        task_count=2,
        workers=1,
        batch_size=1,
        checkpoint_interval=0,
        census_counts=counts,
    )
    reports = [run_selection_task(inventory, output, task_index=index, **options) for index in range(2)]
    aggregate_options = {
        key: value
        for key, value in options.items()
        if key not in {"task_count", "workers", "batch_size", "checkpoint_interval"}
    }
    result = aggregate_selection_reports(inventory, reports, **aggregate_options)
    assert result["complete_cluster"]
    assert result["selected_occurrences"] == 2
    assert result["quality_counts"] == {"1": 2}
    assert result["census_zero_shards"] == 1
    assert result["sources"]["books"]["selected_occurrences"] == 2
    with pytest.raises(ValueError, match="cover every task"):
        aggregate_selection_reports(inventory, reports[:1], **aggregate_options)
    assert reports[0]["completed"][0]["status"] == "selected"
    assert reports[1]["completed"][0]["status"] == "census_zero"
    selected = reports[0]["completed"][0]
    rows = pq.ParquetFile(selected["output"]).read().to_pylist()
    assert [(row["id"], row["normalized_row"]) for row in rows] == [("repeat", 0), ("repeat", 1)]
    verification_options = {key: value for key, value in aggregate_options.items() if key != "census_counts"}
    assert verify_selection_outputs(inventory, reports, **verification_options) == {
        "verified_shards": 1,
        "verified_occurrences": 2,
    }
    proofs = [
        verify_selection_task(
            inventory,
            report,
            str(tmp_path / "verification" / f"task-{index}.json"),
            task_index=index,
            task_count=2,
            **verification_options,
        )
        for index, report in enumerate(reports)
    ]
    verified = aggregate_verification_reports(inventory, reports, proofs, result)
    assert verified["verified_occurrences"] == 2 and verified["verified_shards"] == 1
    assert verified["complete"]
    assert json.loads((tmp_path / "verification" / "task-0.json").read_text())["verified_occurrences"] == 2
    with pytest.raises(ValueError, match="cover every selection task"):
        aggregate_verification_reports(inventory, reports, proofs[:1], result)
    with pytest.raises(ValueError, match="cover every selection task"):
        aggregate_verification_reports(inventory, reports, [proofs[0], proofs[0]], result)
    (prefix / source_key / first).unlink()
    resumed = [run_selection_task(inventory, output, task_index=index, **options) for index in range(2)]
    assert aggregate_selection_reports(inventory, resumed, **aggregate_options)["selected_occurrences"] == 2
    assert resumed[0]["completed"] == reports[0]["completed"]
    table = pq.ParquetFile(selected["output"]).read()
    table = table.set_column(table.schema.get_field_index("id"), "id", pa.array(["wrong", "repeat"]))
    pq.write_table(table, selected["output"])
    with pytest.raises(ValueError, match="occurrence hash"):
        verify_selection_outputs(inventory, reports, **verification_options)
