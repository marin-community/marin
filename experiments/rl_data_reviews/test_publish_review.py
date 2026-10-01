# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace


def test_archive_retains_harbor_evidence_and_runtime_identity(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    publisher = importlib.import_module("publish_review")
    files = {
        "tasks/0000/execution-000/review-f634561b1d8dd2b88908/agent/trajectory.json": '{"steps": [1, 2]}',
        "tasks/0000/execution-000/review-f634561b1d8dd2b88908/agent/offline-tooling.json": '{"return_code": 0}',
        "tasks/0001/execution-000/trial/agent/trajectory.json": '{"steps": [3]}',
        "tasks/0000/attempt.json": '{"verification": {"status": "verified"}}',
        "run.json": '{"config": {"model": {"name": "solver", "parameters": {"temperature": 0.7}}}}',
        "quality-review.schema.json": '{"type": "object"}',
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    uploaded = {}

    def remote_sql(*args, **kwargs):
        payload = json.loads(kwargs["input"])
        for artifact in json.loads(payload["parameters"]["items"]):
            uploaded[artifact["path"]] = artifact
        return SimpleNamespace(stdout='{"rows": []}')

    monkeypatch.setattr(publisher.subprocess, "run", remote_sql)
    collection = {"reviews": [{"evidence": [{"snapshot_path": "tasks/0000/attempt.json"}]}]}
    publication = publisher.ReviewPublication(tmp_path, "source", collection, {}, {}, None)

    publisher.archive_evidence(publication, "review-id")

    assert set(uploaded) == set(files)
    for relative, content in files.items():
        assert uploaded[relative]["content"] == content
        assert uploaded[relative]["sha256"] == hashlib.sha256(content.encode()).hexdigest()
