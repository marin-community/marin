# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend census manifest composition from sealed measurement rows."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.backend_census_manifest import (
    MANIFEST_KIND,
    compose_manifest,
    load_rows,
    seal_manifest,
)

PROFILE = {"cpus": 4, "memory_mb": 8192, "storage_mb": 10240}


def _write_population(root: Path, source_slug: str, tasks: int) -> str:
    population = root / source_slug / "census" / "population"
    (population / "specs").mkdir(parents=True, exist_ok=True)
    table = pa.table({"task_spec": pa.array([json.dumps({"id": f"{source_slug}-{i}"}) for i in range(tasks)])})
    pq.write_table(table, population / "specs" / "part-000000.parquet")
    return str(population)


def _actor_artifact(root: Path, source_id: str, measured: int) -> dict:
    artifact = root / "actor-census" / hashlib.sha256(source_id.encode()).hexdigest()[:20]
    artifact.mkdir(parents=True, exist_ok=True)
    record = {"result": {"actor_census_input": {"dataset_id": "upstream/data", "measured_tasks": measured}}}
    uri = artifact / ".artifact.json"
    uri.write_text(json.dumps(record))
    payload = uri.read_bytes()
    return {"uri": str(uri), "sha256": hashlib.sha256(payload).hexdigest()}


def _row(root: Path, source_id: str, tasks: int = 10, sealed_inputs: bool = False) -> dict:
    slug = source_id.replace(":", "_").replace("/", "_")
    population = _write_population(root, slug, tasks)
    row = {
        "source_id": source_id,
        "actor_artifact": _actor_artifact(root, source_id, tasks),
        "declaration": {
            "id": source_id,
            "actor_census_input": {"dataset_id": "upstream/data", "measured_tasks": tasks},
            "grading_environment": {"assessed_tasks": tasks},
            "integrity_contracts": {"groups": []},
        },
        "input": {"dataset_id": "upstream/data", "measured_tasks": tasks},
        "population_path": population,
    }
    if sealed_inputs:
        shard = f"{population}/specs/part-000000.parquet"
        row["inputs"] = {shard: hashlib.sha256(Path(shard).read_bytes()).hexdigest()}
    return row


def _write_rows(tmp_path: Path, source_ids: list[str], sealed_inputs: bool = False) -> Path:
    path = tmp_path / "inputs.json"
    path.write_text(
        json.dumps({"rows": [_row(tmp_path, sid, sealed_inputs=sealed_inputs) for sid in source_ids]}, indent=1)
    )
    return path


def _defaults() -> dict:
    return {"agent": dict(PROFILE), "grader": dict(PROFILE)}


def test_compose_covers_exactly_the_staged_targets_with_spec_paths(tmp_path):
    rows = load_rows([_write_rows(tmp_path, ["Alpha:one", "Beta:two"], sealed_inputs=True)])
    manifest = compose_manifest(
        rows,
        snapshot_id="assembly-test-v1",
        catalog_root="s3://catalog/production",
        source_revision="a" * 40,
        runtime_defaults=_defaults(),
    )
    assert manifest["kind"] == MANIFEST_KIND
    assert manifest["source_ids"] == ["Alpha:one", "Beta:two"]
    assert [entry["source_id"] for entry in manifest["inputs"]] == ["Alpha:one", "Beta:two"]
    assert all(entry["paths"] and entry["paths"][0].endswith(".parquet") for entry in manifest["inputs"])
    assert all(entry["column"] == "task_spec" and entry["expected_tasks"] == 10 for entry in manifest["inputs"])
    assert manifest["grading_environments"]["Alpha:one"]["assessed_tasks"] == 10


def test_compose_rejects_duplicate_rows_and_mismatched_bindings(tmp_path):
    with pytest.raises(ValueError, match="Duplicate measurement row across batches"):
        load_rows([_write_rows(tmp_path, ["Alpha:one", "Alpha:one"])])
    single = load_rows([_write_rows(tmp_path, ["Alpha:one"])])
    single[0].declaration["actor_census_input"]["measured_tasks"] = 999
    with pytest.raises(ValueError, match="differs from its declaration"):
        compose_manifest(
            single,
            snapshot_id="s",
            catalog_root="c",
            source_revision="a" * 40,
            runtime_defaults=_defaults(),
        )


def test_seal_manifest_is_write_once_with_readback(tmp_path):
    rows = load_rows([_write_rows(tmp_path, ["Alpha:one"])])
    manifest = compose_manifest(
        rows, snapshot_id="s", catalog_root="c", source_revision="a" * 40, runtime_defaults=_defaults()
    )
    output = str(tmp_path / "sealed" / "manifest.json")
    reference = seal_manifest(manifest, output)
    assert StoragePath(reference.uri).exists()
    again = seal_manifest(manifest, output)
    assert again == reference
    with pytest.raises(ValueError, match="use a new output path"):
        seal_manifest({**manifest, "snapshot_id": "other"}, output)
