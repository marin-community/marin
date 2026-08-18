# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import read_record

from experiments.datakit import parquet_rewrite


def test_rewrite_train_runs_in_order_and_resumes_from_artifact_records(tmp_path, monkeypatch):
    completed: list[tuple[str, ...]] = []

    def fake_migration(source_globs, **_kwargs):
        completed.append(source_globs)
        return {"files_rewritten": 1}

    monkeypatch.setattr(parquet_rewrite, "run_migration", fake_migration)
    prefixes = (
        parquet_rewrite.RewritePrefix("first", ("s3://bucket/first/*.parquet",)),
        parquet_rewrite.RewritePrefix("second", ("s3://bucket/second/*.parquet",)),
        parquet_rewrite.RewritePrefix("third", ("s3://bucket/third/*.parquet",)),
    )

    result = parquet_rewrite.run_rewrite_train(prefixes, artifact_prefix=str(tmp_path))

    assert completed == [prefix.source_globs for prefix in prefixes]
    assert result.source_globs == prefixes[-1].source_globs
    assert result.counters == {"files_rewritten": 1}
    assert all(read_record(f"{tmp_path}/{prefix.name}/{parquet_rewrite.REWRITE_VERSION}") for prefix in prefixes)

    parquet_rewrite.run_rewrite_train(prefixes, artifact_prefix=str(tmp_path))
    assert completed == [prefix.source_globs for prefix in prefixes]


def test_inventory_manifest_preserves_fixed_steps_and_artifacts(tmp_path):
    manifest_path = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.table(
            {
                "step_index": [0, 0, 1],
                "step_name": ["rollup", "rollup", "large"],
                "step_files": [3, 3, 4],
                "step_bytes": [30, 30, 40],
                "artifact_root": ["marin/a", "marin/b", "marin/c"],
                "artifact_files": [1, 2, 4],
                "artifact_bytes": [10, 20, 40],
                "source_glob": ["s3://bucket/a/*.parquet", "s3://bucket/b/*.parquet", "s3://bucket/c/*.parquet"],
                "directory_files": [1, 2, 4],
                "directory_bytes": [10, 20, 40],
            }
        ),
        manifest_path,
    )

    rows = parquet_rewrite.read_inventory_manifest(str(manifest_path))
    prefixes = parquet_rewrite.inventory_rewrite_prefixes(rows)

    assert prefixes == (
        parquet_rewrite.RewritePrefix("rollup", ("s3://bucket/a/*.parquet", "s3://bucket/b/*.parquet"), 3, 30),
        parquet_rewrite.RewritePrefix("large", ("s3://bucket/c/*.parquet",), 4, 40),
    )
