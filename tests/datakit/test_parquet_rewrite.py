# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.artifact import read_record

from experiments.datakit import parquet_rewrite


def test_rewrite_train_runs_in_order_and_resumes_from_artifact_records(tmp_path, monkeypatch):
    completed: list[str] = []

    def fake_migration(source_glob, *, workers, worker_cpu, worker_ram, options):
        completed.append(source_glob)
        return {"files_rewritten": 1}

    monkeypatch.setattr(parquet_rewrite, "run_migration", fake_migration)
    prefixes = (
        parquet_rewrite.RewritePrefix("first", "s3://bucket/first/*.parquet"),
        parquet_rewrite.RewritePrefix("second", "s3://bucket/second/*.parquet"),
        parquet_rewrite.RewritePrefix("third", "s3://bucket/third/*.parquet"),
    )

    result = parquet_rewrite.run_rewrite_train(prefixes, artifact_prefix=str(tmp_path))

    assert completed == [prefix.source_glob for prefix in prefixes]
    assert result.source_glob == prefixes[-1].source_glob
    assert result.counters == {"files_rewritten": 1}
    assert all(read_record(f"{tmp_path}/{prefix.name}/{parquet_rewrite.REWRITE_VERSION}") for prefix in prefixes)

    parquet_rewrite.run_rewrite_train(prefixes, artifact_prefix=str(tmp_path))
    assert completed == [prefix.source_glob for prefix in prefixes]
