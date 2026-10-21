# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local source snapshots exercise the same selector as remote Parquet sources."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

from experiments.post_training.snowball_curriculum import v0
from experiments.post_training.snowball_curriculum.selection import load_sources
from experiments.post_training.snowball_curriculum.v0 import run


def test_v0_exports_admitted_source_selection_and_shortfall(tmp_path, monkeypatch):
    records = []
    for i in range(4):
        task = TaskSpec(
            id=str(i),
            context=ConversationInput(events=(TextMessage(role="user", content=f"Task {i}"),)),
            environment_requirements=EnvironmentRequirements(),
            answer_type=AnswerType.TEXT,
            answer_format=PlainText(),
            grader=NoGrader(reason="Fixture carries an opaque contract"),
            source=Source(dataset="fixture", revision="pinned", row=str(i), importer_revision="1"),
        )
        records.append({"task_json": task.model_dump_json()})
    (tmp_path / "tasks.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "population": "admitted",
                "sources": [
                    {
                        "source_id": "extension",
                        "intended_use": "train",
                        "task_paths": ["tasks.jsonl"],
                        "catalog_snapshot_id": "frozen-snapshot",
                    }
                ],
            }
        )
    )
    sources = load_sources(str(manifest), population="admitted")
    output = tmp_path / "selected"
    with monkeypatch.context() as local_cwd:
        local_cwd.chdir(tmp_path)
        run(sources=sources, recipe={"extensions": {"extension": "reasoning_gym"}}, output="selected", target=10)
    report = json.loads((output / "selection.json").read_text())
    exported = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
    assert report["selected_count"] == 4
    assert report["shortfall"] == 6
    assert {row["task_id"] for row in exported} == {"0", "1", "2", "3"}
    assert all(row["population"] == "admitted" for row in exported)
    # The source snapshot can be replayed without consulting a mutable catalog pointer.
    assert load_sources(str(output / "sources.json"), population="admitted") == sources
    with pytest.raises(FileExistsError):
        run(sources=sources, recipe={"extensions": {"extension": "reasoning_gym"}}, output=str(output), target=10)
    with pytest.raises(ValueError, match="converted"):
        load_sources(str(manifest), population="converted")
    frozen_recipe = {
        "extensions": {"extension": "reasoning_gym"},
        "catalog_uri": "s3://frozen/catalog",
        "catalog_snapshot_id": "frozen-snapshot",
        "migration_receipt": "s3://frozen/receipt.json",
    }
    recipe_path = tmp_path / "frozen-recipe.json"
    recipe_path.write_text(json.dumps(frozen_recipe))
    holdouts = tmp_path / "holdouts.json"
    holdouts.write_text(json.dumps({"population": "converted", "sources": []}))
    replay_output = tmp_path / "replayed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v0",
            "--sources",
            str(manifest),
            "--recipe",
            str(recipe_path),
            "--holdout-sources",
            str(holdouts),
            "--target",
            "10",
            "--output",
            str(replay_output),
        ],
    )
    v0.main()
    replayed_recipe = json.loads((replay_output / "recipe.json").read_text())
    assert {key: replayed_recipe[key] for key in frozen_recipe} == frozen_recipe
    assert json.loads((replay_output / "selection.json").read_text())["selected_count"] == 4
    # A missing top-level pin is recovered from the per-source snapshot; a conflicting pin fails before export.
    frozen_recipe.pop("catalog_snapshot_id")
    recipe_path.write_text(json.dumps(frozen_recipe))
    monkeypatch.setattr(sys, "argv", [*sys.argv[:-1], str(tmp_path / "inferred")])
    v0.main()
    assert json.loads((tmp_path / "inferred" / "recipe.json").read_text())["catalog_snapshot_id"] == "frozen-snapshot"
    frozen_recipe["catalog_snapshot_id"] = "different-snapshot"
    recipe_path.write_text(json.dumps(frozen_recipe))
    with pytest.raises(ValueError, match="snapshots differ"):
        v0.main()


@pytest.mark.parametrize("changed_manifest", [False, True])
def test_catalog_sources_resolves_imported_archives_through_verified_migration(tmp_path, monkeypatch, changed_manifest):
    destination = tmp_path / "migrated"
    final = destination / "version" / "final"
    final.mkdir(parents=True)
    parquet = final / "part-00000.parquet"
    pq.write_table(pa.table({"task_id": ["admitted-task"]}), parquet)
    original = "gs://old-region/expired/source/"
    archived = {"intended_use": "train", "datasets": {"final": "gs://old-region/expired/source/version/final"}}
    manifest = destination / ".verified-copy-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": original,
                "destination": str(destination),
                "files": [{"path": "version/final/part-00000.parquet"}],
            }
        )
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "prefixes": [
                    {
                        "source": original,
                        "destination": str(destination),
                        "url": str(manifest),
                        "ready": True,
                        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )

    class ImportedCatalog:
        """An imported catalog has archived evidence and no original attempt paths."""

        def __init__(self, uri):
            self.uri = uri

        def current(self):
            return SimpleNamespace(snapshot_id="frozen-snapshot")

        def sources(self, _snapshot_id):
            return [
                SimpleNamespace(id="selected", assessment_key="assessment"),
                SimpleNamespace(id="unrequested", assessment_key="unavailable"),
            ]

        def assessment(self, key):
            if key == "unavailable":
                raise FileNotFoundError("Unrequested source evidence is unavailable")
            return SimpleNamespace(curation_uri="finestore://archived/evidence")

        def document(self, _uri):
            return archived

    monkeypatch.setattr(v0, "CurationCatalog", ImportedCatalog)
    if changed_manifest:
        manifest.write_text(manifest.read_text() + "\n")
        with pytest.raises(ValueError, match="Migration manifest changed"):
            v0.catalog_sources("local-catalog", source_ids={"selected"}, migration_receipt=str(receipt))
        return
    snapshot, sources = v0.catalog_sources("local-catalog", source_ids={"selected"}, migration_receipt=str(receipt))
    assert snapshot == "frozen-snapshot"
    assert len(sources) == 1
    selected = sources[0]
    assert pq.read_table(selected.task_paths[0]).to_pylist() == [{"task_id": "admitted-task"}]
    assert selected.manifest_sha256 == canonical_sha256(archived)
    assert selected.catalog_snapshot_id == snapshot
    assert selected.assessment_curation_uri == "finestore://archived/evidence"
