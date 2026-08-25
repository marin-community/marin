# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path

import polars as pl
import pytest

import marin.datakit.download.code_alchemy_swh_download as swh


def test_balanced_slices_are_contiguous_bounded_and_near_equal():
    slices = swh.balanced_slices(1_000_001, 500_000)

    assert slices == [(0, 333_334), (333_334, 333_334), (666_668, 333_333)]
    assert max(size for _, size in slices) <= 500_000
    assert swh.balanced_slices(0, 10) == []
    with pytest.raises(ValueError, match="positive"):
        swh.balanced_slices(1, 0)


def test_task_resource_defaults_match_worker_pool():
    cfg = swh.CodeAlchemySwhDownloadConfig()

    assert cfg.build_task_resources.preemptible == cfg.worker_resources.preemptible
    assert cfg.download_task_resources.preemptible == cfg.worker_resources.preemptible
    assert cfg.build_task_resources.cpu <= cfg.worker_resources.cpu
    assert cfg.download_task_resources.cpu <= cfg.worker_resources.cpu


def test_prefix_discovery_refuses_partial_stack_join(monkeypatch):
    class FakeStoragePath:
        def __init__(self, path: str):
            self.path = path

        def glob(self):
            if "blob_prefix=00" in self.path:
                return [self.path.replace("*.parquet", "part-00000.parquet")]
            return []

    monkeypatch.setattr(swh, "PREFIX_COUNT", 2)
    monkeypatch.setattr(swh, "StoragePath", FakeStoragePath)

    with pytest.raises(FileNotFoundError, match="expected all 2 prefixes"):
        swh._prefix_inputs("s3://bucket/missing-ids")


def test_planner_balances_each_prefix_and_uses_contracted_paths(monkeypatch, tmp_path: Path):
    inputs = {
        "00": (str(tmp_path / "00.parquet"),),
        "ff": (str(tmp_path / "ff.parquet"),),
    }
    counts = {"00": 1_000_001, "ff": 1}
    monkeypatch.setattr(swh, "_prefix_inputs", lambda _: inputs)
    monkeypatch.setattr(swh, "_count_prefix_rows", lambda prefix, paths: (prefix, counts[prefix]))
    monkeypatch.setattr(swh, "rust_crate_fingerprint", lambda: "abc123")

    class EmptyOutputGlob:
        def __init__(self, _path: str):
            pass

        def glob(self):
            return []

    monkeypatch.setattr(swh, "StoragePath", EmptyOutputGlob)
    cfg = swh.CodeAlchemySwhDownloadConfig(
        input_path="s3://bucket/root/missing-ids",
        output_path="s3://bucket/root/downloaded-gzip",
        ids_per_task=500_000,
        metadata_workers=2,
    )

    tasks = swh.list_download_shards(cfg)

    assert [(task.prefix, task.row_start, task.row_count) for task in tasks] == [
        ("00", 0, 333_334),
        ("00", 333_334, 333_334),
        ("00", 666_668, 333_333),
        ("ff", 0, 1),
    ]
    assert tasks[0].binary_path == "s3://bucket/root/downloaded-gzip/binary/abc123/swh_downloader"
    assert tasks[0].output_path == (
        "s3://bucket/root/downloaded-gzip/data/blob_prefix=00/part-00000.parquet"
    )
    assert tasks[0].metrics_path == (
        "s3://bucket/root/downloaded-gzip/.metrics/blob_prefix=00/part-00000.json"
    )


def test_download_shard_normalizes_schema_accounts_failures_and_resumes(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "missing.parquet"
    output_path = tmp_path / "out" / "data" / "blob_prefix=00" / "part-00000.parquet"
    failure_path = tmp_path / "out" / "failures" / "blob_prefix=00" / "part-00000.tsv"
    metrics_path = tmp_path / "out" / ".metrics" / "blob_prefix=00" / "part-00000.json"
    for path in (output_path, failure_path, metrics_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    ids = ["00" + "1" * 38, "00" + "2" * 38, "00" + "3" * 38]
    pl.DataFrame({"blob_id": ids}).write_parquet(input_path)
    fake_binary = tmp_path / "swh_downloader"
    fake_binary.touch()
    monkeypatch.setattr(swh, "_local_rust_binary", lambda *_: fake_binary)
    calls = []

    def fake_run(command, *, check, env, **kwargs):
        calls.append(command)
        rust_output = Path(env["OUTPUT_DIR"])
        pl.DataFrame({"blob_id": [ids[0]], "content": [b"\x1f\x8bgzip"]}).write_parquet(
            rust_output / "ids.parquet"
        )
        (rust_output / "ids.failures.tsv").write_text(
            f"{ids[1]}\tnot_found\n{ids[2]}\tretryable_failure\n"
        )
        Path(env["METRICS_PATH"]).write_text(
            json.dumps(
                {
                    "success": 1,
                    "not_found": 1,
                    "retryable_failure": 1,
                    "permanent_failure": 0,
                    "classified": 3,
                    "retry_events": 6,
                    "throttled": 2,
                    "bytes": 6,
                    "connection_opens": 4,
                    "connection_errors": 1,
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(swh.subprocess, "run", fake_run)
    task = swh.DownloadShardTask(
        prefix="00",
        shard_index=0,
        input_paths=(str(input_path),),
        row_start=0,
        row_count=3,
        output_path=str(output_path),
        failure_path=str(failure_path),
        metrics_path=str(metrics_path),
        binary_path="s3://bucket/binary",
        crate_fingerprint="fingerprint",
        fleet=64,
        pipeline=8,
        max_attempts=6,
        attempt_timeout_seconds=15,
        connect_timeout_seconds=5,
        tokio_workers=2,
        row_group_mb=16,
    )

    result = swh.download_swh_shard(task)

    assert result.success == 1
    assert result.not_found == 1
    assert result.retryable_failure == 1
    assert pl.read_parquet(output_path).schema == {"blob_id": pl.String, "source_gzip": pl.Binary}
    assert "not_found" in failure_path.read_text()
    assert json.loads(metrics_path.read_text())["status"] == "complete"
    assert len(calls) == 1

    retried = swh.download_swh_shard(task)
    assert retried.reused is False
    assert len(calls) == 2

    totals = swh._aggregate_download_results([task], str(tmp_path / "out"))
    assert totals == {
        "input_rows": 3,
        "success": 1,
        "not_found": 1,
        "retryable_failure": 1,
        "permanent_failure": 0,
        "retry_events": 6,
        "bytes": 6,
    }
    assert json.loads((tmp_path / "out" / ".metrics" / "aggregate-download.json").read_text()) == totals
