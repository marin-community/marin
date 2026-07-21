# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pytest

from experiments.grug.moe.standalone.cuda_module_probe import (
    read_probe_events,
    summarize_events,
    task_index_from_environment,
    upload_probe_artifacts,
)


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_summary_pairs_loads_and_counts_recovery(tmp_path: Path) -> None:
    _write_events(
        tmp_path / "probe-0-1.ndjson",
        [
            {
                "event": "load_enter",
                "sequence": 1,
                "api": "cuModuleLoadFatBinary",
                "effective_profile": "trace",
                "sha256": "abc",
                "in_flight": 2,
            },
            {
                "event": "load_exit",
                "sequence": 1,
                "result": 0,
                "pre_sync_result": 0,
                "attempts": [{"name": "original", "result": 1}, {"name": "same_pointer", "result": 0}],
            },
        ],
    )

    summary = summarize_events(read_probe_events(tmp_path))

    assert summary.load_count == 1
    assert summary.profiles == {"trace": 1}
    assert summary.apis == {"cuModuleLoadFatBinary": 1}
    assert summary.original_results == {"1": 1}
    assert summary.recovery_stages == {"same_pointer": 1}
    assert summary.sync_results == {"0": 1}
    assert summary.hashes == {"abc": 1}
    assert summary.maximum_in_flight == 2


def test_summary_rejects_unmatched_exit(tmp_path: Path) -> None:
    _write_events(tmp_path / "probe-0-1.ndjson", [{"event": "load_exit", "sequence": 4, "attempts": []}])

    with pytest.raises(ValueError, match="without load_enter"):
        summarize_events(read_probe_events(tmp_path))


def test_upload_writes_compressed_events_and_task_zero_cubins(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_events(
        log_dir / "probe-0-1.ndjson",
        [
            {
                "event": "load_enter",
                "sequence": 1,
                "api": "cuModuleLoadFatBinary",
                "effective_profile": "trace",
                "in_flight": 1,
            },
            {"event": "load_exit", "sequence": 1, "result": 0, "pre_sync_result": 0, "attempts": []},
        ],
    )
    (log_dir / "abc.cubin").write_bytes(b"cubin")
    destination = tmp_path / "uploaded"

    upload_probe_artifacts(log_dir, str(destination), task_index=0)

    task_dir = destination / "task-0"
    assert json.loads((task_dir / "summary.json").read_text())["load_count"] == 1
    with gzip.open(task_dir / "events.ndjson.gz", "rt") as events:
        assert [json.loads(line)["event"] for line in events] == ["load_enter", "load_exit"]
    assert (task_dir / "cubins" / "abc.cubin").read_bytes() == b"cubin"


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"IRIS_TASK_INDEX": "3"}, 3),
        ({"IRIS_TASK_ID": "/user/job/15:0"}, 15),
    ],
)
def test_task_index_from_environment(environment: dict[str, str], expected: int) -> None:
    assert task_index_from_environment(environment) == expected
