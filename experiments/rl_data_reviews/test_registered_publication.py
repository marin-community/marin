# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("reviewed_registration", ["evaluated-contract", "earlier-contract", None])
def test_publication_binds_review_to_registration_and_archives_run(tmp_path, monkeypatch, reviewed_registration):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    publisher = importlib.import_module("publish_review")
    atlas_id = "Registered releases:example"
    subject = {
        "id": "source:example",
        "level": "source",
        "repository": "example/tasks",
        "dataset_revision": "a" * 40,
        "source_id": atlas_id,
        "task_id": None,
        "task_path": None,
        "row_index": None,
    }
    collection = {
        "schema_version": "0.3.0",
        "created_at": "2026-10-01T00:00:00Z",
        "subjects": [subject],
        "reviews": [],
        "tag_assignments": [],
        "source_mappings": [],
        "execution_provenance": {
            "marinskyrl_commit": "b" * 40,
            "marinskyrl_dirty": False,
            "marinskyrl_python_tree_sha256": "c" * 64,
            "harbor_commit": None,
        },
    }
    payload = {
        "origin": "Registered releases",
        "dataset_revision": "a" * 40,
        "registration_revision": "evaluated-contract",
    }
    source = {"source_id": atlas_id, "revision": "a" * 40}
    if reviewed_registration is not None:
        source["registration_revision"] = reviewed_registration
    run = {"config": {"source": source}}
    (tmp_path / "quality-reviews.json").write_text(json.dumps(collection))
    (tmp_path / "quality-review.schema.json").write_text(
        (Path(__file__).parent / "quality-review.schema.json").read_text()
    )
    (tmp_path / "run.json").write_text(json.dumps(run))
    artifacts = []

    def remote_sql(*_args, **kwargs):
        parameters = json.loads(kwargs["input"])["parameters"]
        if "items" in parameters:
            artifacts.extend(json.loads(parameters["items"]))
            response = {}
        else:
            response = {"rows": [{"payload": payload}]}
        return SimpleNamespace(stdout=json.dumps(response))

    monkeypatch.setattr("subprocess.run", remote_sql)
    if reviewed_registration != "evaluated-contract":
        with pytest.raises(ValueError, match="population or execution contract"):
            publisher.validated_publication(tmp_path, atlas_id)
        assert artifacts == []
        return
    publication = publisher.validated_publication(tmp_path, atlas_id)
    publisher.archive_evidence(publication, "review1")
    assert publication.subject["dataset_revision"] == "a" * 40
    assert {item["path"] for item in artifacts} == {"run.json"}
    assert json.loads(artifacts[0]["content"]) == run
