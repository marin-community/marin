# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Census: measure each admitted population and seal the population census receipt.

One maintained module replaces the per-release census workers (see INVENTORY.md).
Every release-identifying value arrives as typed :class:`CensusInputs`; worker source
carries no release names or load IDs. Per source the stage retains the SampleInventory
panel, computes oracle/static counts through the maintained classifiers, and prepares
the frozen public inputs the curriculum corpus consumes.

The small slice (``only``) is the per-stage preflight path (DESIGN.md §6): one real
source through the real code path into ``slice_output_root``, never the release
receipt root. Receipts are sealed write-once with byte-identical readback.
"""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from finestore.rl.curation import ActorCensusInput, SourceDeclaration
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.oracle_filter import has_oracle_solution

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.admission_inventory import inventory_source
from experiments.post_training.task_curation.calibration_inventory import task_blockers
from experiments.post_training.task_curation.curriculum_backfill import FrozenCoverageSource, prepare_source
from experiments.post_training.task_curation.datasets.retained_releases import retained_release_receipts
from experiments.post_training.task_curation.lexical_artifacts import SourceTextPopulation
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.release_stages.sizing import StageResources, census_sizing
from experiments.post_training.task_curation.sources import all_sources
from experiments.post_training.task_curation.static_retention import STATIC_REASONS
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

INPUTS_KIND = "atlas-population-census-inputs"
RECEIPT_SUFFIX = "admission-population-census"
# The actorless exemption the static classifier applies before the STATIC_REASONS gate.
ACTORLESS_EXEMPTION = "requires_calibration_solution_and_empty_liveness"
CLASSIFIER = "calibration_inventory.task_blockers with actorless exemption; static_retention.STATIC_REASONS"
_PARQUET_BATCH_ROWS = 16


@dataclass(frozen=True)
class CensusSourceRow:
    """One admitted source's census inputs: its frozen declaration and population pin."""

    source_id: str
    declaration: dict
    population_path: str


@dataclass(frozen=True)
class CensusInputs:
    """Typed inputs for one census run; every value is a graph output."""

    release_id: str
    source_revision: str
    inputs: ArtifactRef
    output_root: str
    region_prefix: str
    expected_sources: int
    slice_output_root: str | None = None

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision:
            raise ValueError("release_id and source_revision are required")
        if not self.output_root or not self.region_prefix:
            raise ValueError("output_root and region_prefix are required")
        if self.expected_sources <= 0:
            raise ValueError(f"expected_sources must be positive, got {self.expected_sources}")
        if not self.region_prefix.endswith("/"):
            raise ValueError("region_prefix must end with '/' so the pin check is unambiguous")


def seal(path: str, value: dict) -> ArtifactRef:
    """Write-once seal with byte-identical readback; returns the artifact reference."""
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(path)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"sealed artifact {path} exists with different content; use a new output root")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    if target.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {path}")
    return ArtifactRef(uri=str(target), sha256=hashlib.sha256(raw).hexdigest())


def read_census_rows(inputs: ArtifactRef, revision: str) -> list[CensusSourceRow]:
    """Digest-verify the sealed census-inputs document and return its rows."""
    payload = StoragePath(inputs.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != inputs.sha256:
        raise ValueError(f"census inputs digest mismatch: {inputs.uri}")
    document = json.loads(payload)
    if document.get("kind") != INPUTS_KIND:
        raise ValueError(f"census inputs kind is {document.get('kind')!r}, expected {INPUTS_KIND!r}")
    if document.get("source_revision") != revision:
        raise ValueError(f"census inputs pin revision {document.get('source_revision')!r}, expected {revision!r}")
    rows = [CensusSourceRow(**row) for row in document["rows"]]
    if not rows:
        raise ValueError("census inputs carry no rows")
    if len({row.source_id for row in rows}) != len(rows):
        raise ValueError("census inputs repeat a source_id")
    return sorted(rows, key=lambda row: row.source_id)


def resolve_source(source_id: str, declaration: dict, sources_fn: Callable[[], dict]) -> Any:
    """Resolve the declared source and assert the row carries its exact declaration."""
    matches = [s for s in sources_fn().values() if canonical_source_id(s.info.id) == source_id]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} sources resolve to {source_id!r}; census needs exactly one")
    source = matches[0]
    if source.declaration.model_dump(mode="json") != declaration:
        raise ValueError(f"declared census row for {source_id} does not match the current declaration")
    return source


def population_documents(population: SourceTextPopulation):
    """Stream the frozen population's task-spec documents in a stable order."""
    for path in sorted((StoragePath(population.specs_output_dir) / "*.parquet").glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(batch_size=_PARQUET_BATCH_ROWS, columns=["task_spec"]):
                for row in batch.to_pylist():
                    yield json.loads(row["task_spec"])


def declared_binding(declaration: SourceDeclaration, source_id: str) -> ActorCensusInput:
    """The source's declared census input: its own binding, or the retained lineage's.

    A retained release declares through its retained lineage, not a live actor
    census binding; the retained receipt's own actor census is the declared input.
    """
    if declaration.actor_census_input is not None:
        return declaration.actor_census_input
    retained = retained_release_receipts().get(source_id)
    if retained is None or retained.source_id != source_id:
        raise ValueError(f"declared source lacks its exact input census: {source_id}")
    return retained.actor.input


def measure_source(
    row: CensusSourceRow,
    output_root: str,
    revision: str,
    region_prefix: str,
    *,
    sources_fn: Callable[[], dict] = all_sources,
    inventory_fn: Callable[..., Any] = inventory_source,
) -> dict:
    """Census one source: inventory panel, oracle/static counts, public preparation."""
    configure_coreweave_s3()
    source = resolve_source(row.source_id, row.declaration, sources_fn)
    declaration = source.declaration
    population = SourceTextPopulation.raw_load(row.population_path)
    if canonical_source_id(population.source_id) != canonical_source_id(row.source_id):
        raise ValueError(f"population {population.source_id!r} does not bind {row.source_id!r}")
    for pinned in (population.path, population.specs_output_dir):
        if not pinned.startswith(region_prefix):
            raise ValueError(f"population path {pinned} is outside the release region prefix {region_prefix}")
    inventory = inventory_fn(
        declaration, f"{output_root}/inventories", None, binding=declared_binding(declaration, row.source_id)
    )
    reasons: Counter[str] = Counter()
    partitions: Counter[tuple[str, ...]] = Counter()
    count = oracle_count = blocked_count = 0
    identifiers: set[str] = set()
    for document in population_documents(population):
        task = TaskSpec.model_validate(document)
        if task.id in identifiers:
            raise ValueError(f"population repeats task id {task.id!r} for {row.source_id}")
        identifiers.add(task.id)
        oracle = has_oracle_solution(task)
        count += 1
        oracle_count += int(oracle)
        if not oracle:
            blockers = task_blockers(task) - {ACTORLESS_EXEMPTION}
            unclassified = blockers - STATIC_REASONS.keys()
            if unclassified:
                raise ValueError(f"unclassified static blockers for {row.source_id}: {sorted(unclassified)}")
            blocked_count += bool(blockers)
            reasons.update(blockers)
            partitions[tuple(sorted(blockers))] += 1
    binding = declared_binding(declaration, row.source_id)
    if count != inventory.retained_tasks or count != binding.measured_tasks:
        raise ValueError(
            f"census count {count} disagrees with the inventory ({inventory.retained_tasks})"
            f" and the declared binding ({binding.measured_tasks}) for {row.source_id}"
        )
    if oracle_count != inventory.oracle_tasks:
        raise ValueError(f"oracle count {oracle_count} disagrees with the inventory for {row.source_id}")
    audit = {
        "source_id": row.source_id,
        "input": binding.model_dump(mode="json"),
        "source_revision": revision,
        "tasks": count,
        "scanned_task_specs_sha256": inventory.scanned_task_specs_sha256,
        "oracle_tasks": oracle_count,
        "unique_static_blocked_tasks": None if oracle_count else blocked_count,
        "overlapping_reason_counts": dict(reasons),
        "unique_reason_partitions": [
            {"reasons": list(key), "tasks": value} for key, value in sorted(partitions.items())
        ],
        "classifier": CLASSIFIER,
        "source_commands": 0,
        "provider_creations": 0,
        "model_requests": 0,
        "executability": "unmeasured",
    }
    key = canonical_sha256(row.source_id)[:20]
    static_ref = seal(f"{output_root}/static-counts/{key}.json", audit)
    population_evidence = str(StoragePath(population.path) / ".artifact.json")
    frozen = FrozenCoverageSource(
        declaration=declaration,
        specs_dir=population.specs_output_dir,
        task_column="task_spec",
        raw_count=count,
        task_specs_sha256=population.task_specs_sha256,
        task_spec_scan_sha256=inventory.scanned_task_specs_sha256,
        input_record_sha256=file_sha256(StoragePath(population_evidence)),
        intended_use=declaration.intended_use,
        input_evidence=population_evidence,
    )
    prepared = prepare_source(frozen, f"{output_root}/public-preparation")
    prepared_root = StoragePath(prepared["root"])
    result = {
        "source_id": row.source_id,
        "declaration": declaration.model_dump(mode="json"),
        "inventory": inventory.model_dump(mode="json"),
        "inventory_reference": {
            "uri": inventory.evidence_url,
            "sha256": file_sha256(StoragePath(inventory.evidence_url)),
        },
        "static_audit": {"uri": static_ref.uri, "sha256": static_ref.sha256},
        "static_blocked_tasks": audit["unique_static_blocked_tasks"],
        "static_reason_counts": dict(reasons),
        "oracle_tasks": oracle_count,
        "population": population.model_dump(mode="json"),
        "public_preparation": {
            "root": str(prepared_root),
            "manifest": {
                "uri": str(prepared_root / "manifest.json"),
                "sha256": file_sha256(prepared_root / "manifest.json"),
            },
            "public_inputs": {
                "uri": str(prepared_root / "public-inputs.json"),
                "sha256": file_sha256(prepared_root / "public-inputs.json"),
            },
        },
        "old_task_specs_or_count_bindings_inherited": False,
        "qa_bindings_inherited": False,
        "source_commands": 0,
        "model_requests": 0,
    }
    result_ref = seal(f"{output_root}/results/{key}.json", result)
    return {**result, "result_ref": {"uri": result_ref.uri, "sha256": result_ref.sha256}}


def _resources(total_tasks: int) -> StageResources:
    return census_sizing(total_tasks)


def run_census(
    inputs: CensusInputs,
    *,
    dry_run: bool,
    only: str | None = None,
    sources_fn: Callable[[], dict] = all_sources,
    inventory_fn: Callable[..., Any] = inventory_source,
    feeds: tuple | None = None,
) -> dict:
    """Run one census pass; ``dry_run`` measures a real slice without sealing a receipt."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "census", {"mode": mode, **({"only": only} if only else {})}, alert_feeds)
    try:
        inputs.validate()
        rows = read_census_rows(inputs.inputs, inputs.source_revision)
        if dry_run:
            if only is None:
                raise ValueError("dry-run census requires --only: the preflight measures one real source")
            if inputs.slice_output_root is None:
                raise ValueError("dry-run census requires slice_output_root in the typed inputs")
            rows = [row for row in rows if row.source_id == only]
            if not rows:
                raise ValueError(f"census inputs carry no row for {only!r}")
            output_root = inputs.slice_output_root
        else:
            if only is not None:
                raise ValueError("write-mode census measures every row; use dry-run for a single-source slice")
            if len(rows) != inputs.expected_sources:
                raise ValueError(f"census inputs carry {len(rows)} rows, expected {inputs.expected_sources}")
            output_root = inputs.output_root
        measured = [
            measure_source(
                row,
                output_root,
                inputs.source_revision,
                inputs.region_prefix,
                sources_fn=sources_fn,
                inventory_fn=inventory_fn,
            )
            for row in rows
        ]
        proof: dict[str, Any] = {
            "kind": f"atlas-{inputs.release_id}-{RECEIPT_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "inputs": {"uri": inputs.inputs.uri, "sha256": inputs.inputs.sha256},
            "region_prefix": inputs.region_prefix,
            "sources": len(measured),
            "tasks": sum(entry["inventory"]["retained_tasks"] for entry in measured),
            "oracle_tasks": sum(entry["oracle_tasks"] for entry in measured),
            "dry_run": dry_run,
            "sql_writes": 0,
            "model_requests": 0,
            "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        if not dry_run:
            sizing = _resources(proof["tasks"])
            proof["resources"] = sizing.as_resources()
            proof["results"] = measured
            receipt = seal(f"{inputs.output_root}/receipt.json", proof)
            proof["receipt"] = {"uri": receipt.uri, "sha256": receipt.sha256}
            alerts.stage_finished(
                inputs.release_id,
                "census",
                {"mode": mode, "tasks": proof["tasks"], "sources": proof["sources"], **proof["receipt"]},
                alert_feeds,
            )
            return proof
        alerts.stage_finished(
            inputs.release_id, "census", {"mode": mode, "only": only, "tasks": proof["tasks"]}, alert_feeds
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "census", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON CensusInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    parser.add_argument("--only", help="Measure a single source (dry-run preflight slice)")
    args = parser.parse_args()
    document = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    document["inputs"] = ArtifactRef(**document["inputs"])
    inputs = CensusInputs(**document)
    proof = run_census(inputs, dry_run=args.mode == "dry-run", only=args.only)
    label = "CENSUS-PREFLIGHT-SLICE" if args.mode == "dry-run" else "CENSUS-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
