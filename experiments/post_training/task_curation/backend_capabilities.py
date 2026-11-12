# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Survey backend validation and join a complete converted-spec census for Atlas.

This command never creates machines, snapshots, or provider clients. Inputs are
an explicit local JSON manifest; published object-store shards should first be
staged with fsutil near their storage region.
"""

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.curation import CurationCatalog
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rolloutengine.machines import _machine_spec
from rolloutengine.models import MachineRuntimeSpec
from shellbox.machine import UnsupportedMachineSpec
from shellbox.requirements import Requirement, Support
from shellbox.survey import CapabilityCell, Checker, capability_matrix, production_checkers
from taskcompendium.backend_requirements import (
    BackendFit,
    BackendSourceMap,
    BackendSurvey,
    RuntimeProfile,
    backend_schema_sha256,
    execution_resource_shape,
    task_demands,
)
from taskcompendium.models import EnvironmentRequirements

SUPPORT_ORDER = (Support.SUPPORTED, Support.PARTIAL, Support.UNSUPPORTED)
RESOURCE_FIELDS = ("cpus", "memory_mb", "storage_mb", "gpus")


def production_matrix(checkers: dict[str, Checker]) -> list[CapabilityCell]:
    """Combine factory evidence with production setup and cleanup orchestration."""
    return [
        (
            cell.model_copy(
                update={
                    "support": Support.SUPPORTED,
                    "evidence": (*cell.evidence, "lib/rolloutengine/src/rolloutengine/cleanup.py"),
                    "note": "Rolloutengine bounds cleanup independently of factory creation.",
                }
            )
            if cell.requirement == Requirement.CLEANUP_TIMEOUT
            else (
                cell.model_copy(
                    update={
                        "evidence": (*cell.evidence, "lib/rolloutengine/src/rolloutengine/machines.py:_prepare_machine"),
                        "note": (
                            "Rolloutengine runs setup as root through sh; "
                            "the image or simulator must support each command."
                        ),
                    }
                )
                if cell.requirement == Requirement.SETUP
                else cell
            )
        )
        for cell in capability_matrix(checkers)
    ]


def combined_support(values: Iterable[Support]) -> Support:
    return max(values, key=SUPPORT_ORDER.index, default=Support.SUPPORTED)


def source_map(
    source_id: str,
    documents: Iterable[Mapping],
    matrix: list[CapabilityCell],
    checkers: dict[str, Checker],
    inputs: dict[str, str],
    runtime_defaults: dict[str, RuntimeProfile],
    grading_environment: dict | None = None,
    integrity_contracts: dict | None = None,
) -> BackendSourceMap:
    """Count every converted task, validating combined demands in each machine slot."""
    cells = {(cell.backend, cell.requirement): cell for cell in matrix}
    demand_counts = defaultdict(Counter)
    ranges = defaultdict(dict)
    task_counts = defaultdict(Counter)
    role_counts = {backend: defaultdict(Counter) for backend in checkers}
    blockers = defaultdict(Counter)
    errors = defaultdict(Counter)
    assignments = defaultdict(Counter)
    count = 0
    for document in documents:
        count += 1
        task = document.get("task", document)
        published_runtime = document.get("runtime") or {}
        runtime = {
            slot: {
                **runtime_defaults[role].model_dump(mode="json"),
                **(
                    execution_resource_shape(task, published_runtime.get(slot))
                    if role == "agent"
                    else (published_runtime.get(slot) or {})
                ),
            }
            for role, slot in (("agent", "task_machine"), ("grader", "verifier_machine"))
        }
        demands = task_demands({"task": task, "runtime": runtime})
        for role, needs in demands.items():
            declared = task.get("environment_requirements") if role == "agent" else task["grader"]["environment"]
            assigned = "local" if role == "grader" and declared.get("packages_lock") else "daytona"
            assignments[role][assigned] += 1
            demand_counts[role].update(needs)
            shape = runtime.get("task_machine" if role == "agent" else "verifier_machine")
            if shape:
                for name in RESOURCE_FIELDS:
                    value = shape.get(name)
                    if value is not None:
                        low, high = ranges[role].get(name, (value, value))
                        ranges[role][name] = (min(low, value), max(high, value))
        for backend, check in checkers.items():
            statuses = []
            for role, needs in demands.items():
                status = combined_support(cells[backend, need].support for need in needs)
                for need in needs:
                    if cells[backend, need].support != Support.SUPPORTED:
                        blockers[backend][f"{role}:{need.value}"] += 1
                declared = task.get("environment_requirements") if role == "agent" else task["grader"]["environment"]
                shape = runtime.get("task_machine" if role == "agent" else "verifier_machine")
                requirements = EnvironmentRequirements.model_validate(declared)
                settings = MachineRuntimeSpec.model_validate({**shape, "backend": backend})
                try:
                    spec = _machine_spec(requirements, settings)
                    if backend == "local":
                        spec = replace(spec, memory_mb=None)
                    check(requirements, spec)
                except UnsupportedMachineSpec as error:
                    status = Support.UNSUPPORTED
                    errors[backend][f"{role}: {error}"] += 1
                role_counts[backend][role][status] += 1
                statuses.append(status)
            task_counts[backend][combined_support(statuses)] += 1
    if not count:
        raise ValueError(f"Converted-spec census is empty: {source_id}")
    return BackendSourceMap(
        source_id=source_id,
        basis="converted_spec_census",
        task_count=count,
        inputs=inputs,
        demands=dict(demand_counts),
        resource_ranges=dict(ranges),
        role_backend_assignments=dict(assignments),
        grading_environment=grading_environment,
        integrity_contracts=integrity_contracts,
        backends={
            backend: BackendFit(
                support=combined_support(task_counts[backend]),
                task_counts=dict(task_counts[backend]),
                roles=dict(role_counts[backend]),
                blockers=dict(blockers[backend]),
                validation_errors=dict(errors[backend]),
            )
            for backend in checkers
        },
    )


def converted_specs(paths: list[Path], column: str) -> Iterable[dict]:
    """Stream converted spec JSON without retaining private task payloads."""
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=[column]):
            for encoded in batch.column(0).to_pylist():
                if encoded is None:
                    raise ValueError(f"Null converted spec in {path}")
                yield json.loads(encoded)


def build_survey(manifest: dict, checkers: dict[str, Checker], revision: str) -> BackendSurvey:
    """Bind a census to its catalog inventory; unavailable sources remain unassessed."""
    matrix = production_matrix(checkers)
    runtime_defaults = {
        role: RuntimeProfile.model_validate(manifest["runtime_defaults"][role]) for role in ("agent", "grader")
    }
    source_ids = manifest["source_ids"]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Duplicate inventory source IDs")
    selected = {}
    for entry in manifest["inputs"]:
        source_id = entry["source_id"]
        if source_id not in source_ids or source_id in selected:
            raise ValueError(f"Unknown or duplicate input source: {source_id}")
        paths = [Path(path) for path in entry["paths"]]
        digests = {}
        for index, path in enumerate(paths):
            with path.open("rb") as stream:
                digests[f"{index}:{path.name}"] = hashlib.file_digest(stream, "sha256").hexdigest()
        measured = source_map(
            source_id,
            converted_specs(paths, entry["column"]),
            matrix,
            checkers,
            digests,
            runtime_defaults,
            manifest.get("grading_environments", {}).get(source_id),
            manifest.get("integrity_contracts", {}).get(source_id),
        )
        if measured.task_count != entry["expected_tasks"]:
            raise ValueError(f"Converted population differs from manifest: {source_id}")
        selected[source_id] = measured
    limits = {
        backend: {
            "max_concurrent_sandboxes": None,
            "snapshot_cap": None,
            "evidence": "Not measured by offline factory validation.",
        }
        for backend in checkers
    }
    for backend, observation in manifest.get("deployment_limits", {}).items():
        if backend not in checkers or not observation.get("evidence") or not observation.get("observed_at"):
            raise ValueError("Deployment limits require a backend, timestamp and evidence")
        limits[backend] = observation
    return BackendSurvey(
        catalog_root=manifest["catalog_root"],
        snapshot_id=manifest["snapshot_id"],
        generated_at=datetime.now(UTC).isoformat(),
        git_revision=revision,
        schema_sha256=backend_schema_sha256(),
        matrix=matrix,
        deployment_limits=limits,
        runtime_defaults=runtime_defaults,
        sources=[
            selected.get(source_id)
            or BackendSourceMap(
                source_id=source_id,
                basis="unassessed",
                task_count=0,
                inputs={},
                demands={},
                resource_ranges={},
                backends={},
                role_backend_assignments={},
                grading_environment=manifest.get("grading_environments", {}).get(source_id),
                integrity_contracts=manifest.get("integrity_contracts", {}).get(source_id),
            )
            for source_id in source_ids
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-root", help="Bind to the current published catalog instead of an archived inventory.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.catalog_root is not None:
        configure_coreweave_s3()
        catalog = CurationCatalog(args.catalog_root)
        snapshot = catalog.current()
        sources = list(catalog.sources(snapshot.snapshot_id))
        manifest = {
            **manifest,
            "catalog_root": catalog.root,
            "snapshot_id": snapshot.snapshot_id,
            "source_ids": [source.id for source in sources],
            "grading_environments": {
                source.id: source.declaration.grading_environment.model_dump(mode="json")
                for source in sources
                if source.declaration.grading_environment is not None
            },
        }
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    survey = build_survey(manifest, production_checkers(), revision)
    args.output.write_text(survey.model_dump_json(indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "matrix_cells": len(survey.matrix),
                "sources": len(survey.sources),
                "tasks": sum(source.task_count for source in survey.sources),
            }
        )
    )


if __name__ == "__main__":
    main()
