# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.scripts.metadata_cluster_census import (
    aggregate_census_reports,
    inventory_shards,
    run_census,
)


def test_census_partitions_complete_shards_and_resumes_without_double_counting(tmp_path):
    inventory = {
        "store_uri": str(tmp_path / "store"),
        "store_config": {"cluster_view": 40},
        "sources": [],
        "records": {},
    }
    for source, values in (("books", [22, 22, 27]), ("code", [27, 3, 27, 22])):
        path = tmp_path / f"{source}.parquet"
        pq.write_table(
            pa.table({"cluster_40": values, "id": [str(i) for i in range(len(values))]}), path, row_group_size=2
        )
        inventory["sources"].append(
            {"marin_source": source, "artifact_roots": {"cluster": source}, "cluster_shards": [str(path)]}
        )
        inventory["records"][source] = {"record": {"result": {"output_dir": str(tmp_path)}}}
    shards = inventory_shards(inventory, None)
    options = dict(
        inventory_digest=hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        cluster_view=40,
        task_count=2,
        workers=2,
        batch_size=1,
        checkpoint_interval=0,
        sampled=False,
    )
    reports = [
        run_census(shards, str(tmp_path / f"task-{index}.json"), task_index=index, **options) for index in range(2)
    ]
    result = aggregate_census_reports(inventory, reports)
    assert result["source_cluster_counts"] == {"books": {"22": 2, "27": 1}, "code": {"3": 1, "22": 1, "27": 2}}
    assert result["total_rows"] == 7
    assert result["complete"]
    assert result["completed_shards"] == 2
    assert result["count_scope"] == "preselection_cluster_assignments"
    with pytest.raises(ValueError, match="cover every task"):
        aggregate_census_reports(inventory, reports[:1])
    for shard in shards:
        (tmp_path / f"{shard.source}.parquet").unlink()
    resumed = [
        run_census(shards, str(tmp_path / f"task-{index}.json"), task_index=index, **options) for index in range(2)
    ]
    assert aggregate_census_reports(inventory, resumed)["total_rows"] == 7
