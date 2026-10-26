# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise TTL publication and restore through an external preparation CLI fixture.

Native AgentX format compatibility is checked in its isolated pinned environment.
These tests cover the storage and process boundary without adding that dependency.
"""

import json
import os
import sys
from pathlib import Path

from experiments.evaluation.agentx.recipe import load_series
from experiments.evaluation.agentx.workload import prepare_workload


def test_workload_restores_without_preparation_and_rebuilds_expired_or_changed_inputs(tmp_path, monkeypatch):
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    producer = bin_directory / "uvx"
    fixture = Path(__file__).with_name("fixtures") / "prepare.py"
    producer.write_text(f"#!{sys.executable}\n" + fixture.read_text())
    producer.chmod(0o755)
    events = tmp_path / "events"
    monkeypatch.setenv("PATH", f"{bin_directory}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("AGENTX_FIXTURE_EVENTS", str(events))
    series = load_series()
    series_path = tmp_path / "series.yaml"
    series_path.write_text(json.dumps(series))
    store = str(tmp_path / "store")

    first = prepare_workload(tmp_path / "first", series_path=series_path, cache_root=store)
    pointer = Path(first["archive_uri"]).parent / "workload.json"
    for path in (Path(first["archive_uri"]), pointer):
        os.utime(path, (1, 1))
    restored = prepare_workload(tmp_path / "restored", series_path=series_path, cache_root=store)
    assert restored["cache_status"] == "hit"
    assert restored["archive_sha256"] == first["archive_sha256"]
    assert (tmp_path / "restored/bundle/native/dataset.dat").read_bytes() == b"42" * 100
    assert events.read_text().splitlines().count("build") == 1
    assert Path(first["archive_uri"]).stat().st_mtime > 1
    assert pointer.stat().st_mtime > 1

    # Serving and measurement updates reuse derived data.
    series.update({"duration": 900, "gpu_memory_utilization": 0.9})
    series_path.write_text(json.dumps(series))
    measurement = prepare_workload(tmp_path / "measurement", series_path=series_path, cache_root=store)
    assert measurement["archive_uri"] == first["archive_uri"]
    assert measurement["cache_status"] == "hit"

    # TTL can expire data before its pointer. Only this fixture's exact blob is removed.
    Path(first["archive_uri"]).unlink()
    rebuilt = prepare_workload(tmp_path / "rebuilt", series_path=series_path, cache_root=store)
    assert rebuilt["cache_status"] == "built"
    assert rebuilt["key"] == first["key"]
    assert (tmp_path / "rebuilt/bundle/native/dataset.dat").read_bytes() == b"42" * 100
    assert events.read_text().splitlines().count("build") == 2

    series["random_seed"] = 43
    series_path.write_text(json.dumps(series))
    changed = prepare_workload(tmp_path / "changed", series_path=series_path, cache_root=store)
    assert changed["key"] != first["key"]
    assert (tmp_path / "changed/bundle/native/dataset.dat").read_bytes() == b"43" * 100
    assert Path(rebuilt["archive_uri"]).exists()
