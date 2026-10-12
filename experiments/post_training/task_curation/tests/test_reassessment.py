# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.campaign import error_chain
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
    telemetry = '{"status":"completed","source_wall_seconds":12.0}'
    (retained / "telemetry.json").write_text(telemetry)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"sources": [{"source": source.name, "manifest": str(retained / "manifest.json")}]}))
    (root,) = retained_sources(str(inventory)).values()
    client = LocalClient()
    try:
        with set_current_client(client):
            (adopted,) = run(reassessment_input(source, root))
            assert adopted.status == "incomplete"
            assert adopted.result.outputs["final"] == manifest["datasets"]["final"]
            assert adopted.result.evidence["manifest"] == str(StoragePath(adopted.path) / "manifest.json")
            assert (StoragePath(adopted.path) / "telemetry.json").read_text() == telemetry
            restored = json.loads((StoragePath(adopted.path) / "manifest.json").read_text())
            assert restored["datasets"] == manifest["datasets"]
            assert restored["telemetry"] == str(StoragePath(adopted.path) / "telemetry.json")
            # A changed declaration must not receive grades from the earlier export.
            changed = tmp_path / "data" / "numbers-changed" / "2026.10.07.1"
            changed.mkdir(parents=True)
            (changed / "manifest.json").write_text(json.dumps({**manifest, "source_revision": "another-pin"}))
            with pytest.raises(RuntimeError) as failed:
                run(reassessment_input(source, str(changed)))
            assert "Retained curation pin differs" in error_chain(failed.value)
            assert not (StoragePath(reassessment_input(source, str(changed)).path()) / "manifest.json").exists()
    finally:
        client.shutdown()
