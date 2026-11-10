# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import fsspec
from skyrl_train.io import io
from omegaconf import OmegaConf

from skyrl_train.utils.tracking import Tracking


def test_metrics_cleanup_uses_object_store_listing_paths(monkeypatch):
    export_root = "s3://tracker-test/exports"
    filesystem = fsspec.filesystem("memory")
    list_directory = filesystem.ls

    def list_objects(path, detail=True, **kwargs):
        entries = list_directory(path, detail=detail, **kwargs)
        return entries if detail else [entry.removeprefix("s3://") for entry in entries]

    monkeypatch.setattr(filesystem, "ls", list_objects)
    resolve = io.filesystem_and_path
    monkeypatch.setattr(
        io, "filesystem_and_path", lambda uri: (filesystem, uri) if uri.startswith("s3://") else resolve(uri)
    )
    for step in (0, 3, 8):
        io.write_bytes_atomic(f"{export_root}/metrics/step-{step:06d}.json", json.dumps({"old": 1}).encode())
    config = OmegaConf.create({"trainer": {"export_path": export_root}})
    with Tracking("project", "name", "files", config) as tracker:
        tracker.start(resumed_step=3)
        assert io.exists(f"{export_root}/metrics/step-000000.json")
        assert io.exists(f"{export_root}/metrics/step-000003.json")
        assert not io.exists(f"{export_root}/metrics/step-000008.json")
        tracker.start(resumed_step=None)
        tracker.log({"loss": 0.25}, step=0)
    assert not io.exists(f"{export_root}/metrics/step-000003.json")
    assert json.loads(io.read_bytes(f"{export_root}/metrics/step-000000.json")) == {
        "step": 0,
        "loss": 0.25,
    }
