# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.reassessment import reassessment_input, retained_sources
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source


def test_reassessment_adopts_incomplete_curated_source_without_producer_record(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "cache"))
    source = number_source(tmp_path / "unavailable-original-input")
    retained = tmp_path / "data" / "numbers-abcdef" / "2026.10.07.1"
    retained.mkdir(parents=True)
    manifest = {
        "source": source.name,
        "source_dataset": source.dataset.name,
        "source_revision": source.dataset.revision,
        "recipe_revision": source.version,
        "mode": "full",
        "status": "incomplete",
        "raw_population_count": 30,
        "processed_rows": 30,
        "unprocessed_rows": 0,
        "verification": {"status": "inconclusive"},
        "datasets": {name: str(retained / name) for name in ("download", "normalize", "review", "verify", "final")},
    }
    (retained / "manifest.json").write_text(json.dumps(manifest))
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"sources": [{"source": source.name, "manifest": str(retained / "manifest.json")}]}))
    (root,) = retained_sources(str(inventory)).values()
    client = LocalClient()
    try:
        with set_current_client(client):
            (adopted,) = run(reassessment_input(source, root))
            assert adopted.status == "incomplete"
            assert adopted.result.outputs["final"] == manifest["datasets"]["final"]
            assert adopted.result.evidence["manifest"] == str(retained / "manifest.json")
            assert json.loads((tmp_path / "cache" / adopted.path).joinpath("manifest.json").read_text()) == manifest
            # A changed declaration must not receive grades from the earlier export.
            changed = tmp_path / "data" / "numbers-changed" / "2026.10.07.1"
            changed.mkdir(parents=True)
            (changed / "manifest.json").write_text(json.dumps({**manifest, "source_revision": "another-pin"}))
            with pytest.raises(RuntimeError):
                run(reassessment_input(source, str(changed)))
            assert not (StoragePath(reassessment_input(source, str(changed)).path()) / "manifest.json").exists()
    finally:
        client.shutdown()
