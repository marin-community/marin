# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose the backend census manifest from sealed per-batch measurement rows.

``backend_live_qualification`` consumes a sealed census manifest (kind
``atlas-backend-census-inputs``) whose rows name the converted populations of
exactly the release targets. This module is that manifest's maintained composer:
it verifies each sealed measurement row's actor-census readback and declaration
binding, collects the population's spec shards, and seals the manifest
write-once. Batch rows arrive as the measurement lanes' sealed inputs documents;
nothing here re-measures or re-declares a source.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

LOGGER = logging.getLogger(__name__)
MANIFEST_KIND = "atlas-backend-census-inputs"
SPECS_SUBDIR = "specs"


class MeasurementRow(BaseModel):
    """One sealed measurement-lane row: a target's declaration, census and population."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    actor_artifact: dict[str, str]
    """{uri, sha256} of the sealed actor-census artifact record backing the row."""
    declaration: dict
    input: dict
    population_path: str | None = None
    inputs: dict[str, str] | None = None
    """The measured spec shards as {uri: sha256}, sealed by the measurement lane."""


def _spec_shards(row: MeasurementRow) -> list[str]:
    if row.inputs:
        shards = sorted(row.inputs)
        for shard in shards:
            if ".." in PurePosixPath(shard).parts or not PurePosixPath(shard).suffix == ".parquet":
                raise ValueError(f"Unsafe sealed shard path: {shard}")
        return shards
    if row.population_path is None:
        raise ValueError(f"Row carries neither sealed inputs nor a population path: {row.source_id}")
    shards = sorted(str(path) for path in (StoragePath(row.population_path) / f"{SPECS_SUBDIR}/*.parquet").glob())
    if not shards:
        raise ValueError(f"Population carries no {SPECS_SUBDIR} shards: {row.population_path}")
    for shard in shards:
        if ".." in PurePosixPath(shard).parts:
            raise ValueError(f"Unsafe population shard path: {shard}")
    return shards


def _verify_actor_artifact(row: MeasurementRow) -> None:
    uri = row.actor_artifact["uri"]
    payload = StoragePath(uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != row.actor_artifact["sha256"]:
        raise ValueError(f"Actor artifact digest mismatch: {row.source_id} ({uri})")
    census = json.loads(payload)
    if census.get("actor_census_input", census.get("result", {})) is None:
        raise ValueError(f"Actor artifact carries no census binding: {row.source_id}")


def compose_manifest(
    rows: list[MeasurementRow],
    *,
    snapshot_id: str,
    catalog_root: str,
    source_revision: str,
    runtime_defaults: dict[str, dict],
    grading_environments: dict[str, dict] | None = None,
    integrity_contracts: dict[str, dict] | None = None,
) -> dict:
    """Verify sealed rows and compose the census manifest over exactly their targets."""
    if not rows:
        raise ValueError("No measurement rows staged")
    if set(runtime_defaults) != {"agent", "grader"}:
        raise ValueError(f"runtime_defaults must carry exactly agent and grader profiles: {sorted(runtime_defaults)}")
    inputs = []
    source_ids = []
    grading = dict(grading_environments or {})
    contracts = dict(integrity_contracts or {})
    for row in rows:
        source_id = canonical_source_id(row.source_id)
        declaration_id = canonical_source_id(row.declaration.get("id") or row.declaration.get("source_id") or "")
        if declaration_id != source_id:
            raise ValueError(f"Row declaration binds another source: {row.source_id} vs {declaration_id!r}")
        _verify_actor_artifact(row)
        declared_input = row.declaration.get("actor_census_input") or {}
        if (
            declared_input.get("dataset_id"),
            declared_input.get("dataset_revision"),
            declared_input.get("measured_tasks"),
        ) != (row.input.get("dataset_id"), row.input.get("dataset_revision"), row.input.get("measured_tasks")):
            raise ValueError(f"Row census binding differs from its declaration: {row.source_id}")
        inputs.append(
            {
                "source_id": source_id,
                "paths": _spec_shards(row),
                "column": "task_spec",
                "expected_tasks": row.input["measured_tasks"],
            }
        )
        source_ids.append(source_id)
        if row.declaration.get("grading_environment") is not None:
            grading[source_id] = row.declaration["grading_environment"]
        if row.declaration.get("integrity_contracts") is not None:
            contracts[source_id] = row.declaration["integrity_contracts"]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(f"Duplicate measurement rows: {sorted({s for s in source_ids if source_ids.count(s) > 1})}")
    return {
        "kind": MANIFEST_KIND,
        "snapshot_id": snapshot_id,
        "catalog_root": catalog_root,
        "source_revision": source_revision,
        "runtime_defaults": runtime_defaults,
        "source_ids": sorted(source_ids),
        "inputs": sorted(inputs, key=lambda entry: entry["source_id"]),
        "grading_environments": grading,
        "integrity_contracts": contracts,
    }


def seal_manifest(manifest: dict, output: str) -> ArtifactRef:
    """Seal the manifest write-once with byte-identical readback."""
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(output)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"sealed manifest {output} exists with different content; use a new output path")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    if target.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {output}")
    return ArtifactRef(uri=str(target), sha256=hashlib.sha256(raw).hexdigest())


def load_rows(paths: list[Path], receipts: list[Path] | None = None) -> list[MeasurementRow]:
    """Read sealed batch inputs documents into verified rows.

    ``receipts`` supplies the batch validation receipts whose ``sources`` rows
    carry the sealed spec-shard maps; they join on the canonical source id.
    """
    sealed: dict[str, dict[str, str]] = {}
    for path in receipts or []:
        document = json.loads(StoragePath(str(path)).read_text())
        for raw in document.get("sources") or []:
            shards = raw.get("inputs")
            if shards:
                sealed[canonical_source_id(raw["source_id"])] = shards
    rows: dict[str, MeasurementRow] = {}
    for path in paths:
        document = json.loads(StoragePath(str(path)).read_text())
        batch = document.get("rows")
        if batch is None:
            raise ValueError(f"Sealed inputs document carries no rows: {path}")
        for raw in batch:
            key = canonical_source_id(raw["source_id"])
            if key in rows:
                raise ValueError(f"Duplicate measurement row across batches: {key}")
            if key in sealed:
                raw = {**raw, "inputs": sealed[key]}
            rows[key] = MeasurementRow.model_validate(raw)
    missing = set(sealed) - set(rows)
    if missing:
        raise ValueError(f"Receipt shards name sources the inputs documents lack: {sorted(missing)}")
    return sorted(rows.values(), key=lambda row: canonical_source_id(row.source_id))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, nargs="+", required=True, help="Sealed batch inputs documents")
    parser.add_argument("--receipts", type=Path, nargs="*", help="Batch validation receipts carrying sealed shard maps")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--runtime-defaults", required=True, help="A batch survey document carrying the profiles")
    parser.add_argument("--output", required=True, help="Write-once sealed manifest path")
    args = parser.parse_args()
    configure_coreweave_s3()
    rows = load_rows(args.rows, args.receipts)
    defaults = json.loads(StoragePath(args.runtime_defaults).read_text())["runtime_defaults"]
    manifest = compose_manifest(
        rows,
        snapshot_id=args.snapshot_id,
        catalog_root=args.catalog_root,
        source_revision=args.source_revision,
        runtime_defaults=defaults,
    )
    reference = seal_manifest(manifest, args.output)
    summary = json.dumps({"manifest": reference.model_dump(mode="json"), "sources": len(manifest["source_ids"])})
    print(f"CENSUS-MANIFEST-SEALED {summary}", flush=True)


if __name__ == "__main__":
    main()
