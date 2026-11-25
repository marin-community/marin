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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.curation import ActorCensusInput, OracleFiltrationEvidence, SourceDeclaration
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
    # Explicit per-source worker count, sized by the graph from StageResources.
    workers: int
    slice_output_root: str | None = None

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision:
            raise ValueError("release_id and source_revision are required")
        if not self.output_root or not self.region_prefix:
            raise ValueError("output_root and region_prefix are required")
        if self.expected_sources <= 0:
            raise ValueError(f"expected_sources must be positive, got {self.expected_sources}")
        if self.workers <= 0:
            raise ValueError(f"workers must be positive, got {self.workers}")
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


def _oracle_empty_evidence(source) -> OracleFiltrationEvidence | None:
    """The 0-admission oracle filtration the source's declaration binds, if any."""
    evidence = source.info.oracle_filtration
    if evidence is not None and evidence.admitted_count == 0:
        return evidence
    return None


def _measure_oracle_empty(
    row: CensusSourceRow,
    source,
    evidence: OracleFiltrationEvidence,
    output_root: str,
    revision: str,
) -> dict:
    """The census of a 0-admission oracle-filtered source.

    The filtration retained nothing, so the release ships the excluded, zero-task
    declaration exactly as the prior releases carried it (the prior corpus omitted
    these sources): there is no population to pin, no inventory to draw, and no
    specs to scan — the oracle receipt is the census evidence.
    """
    declaration = source.declaration
    result = {
        "source_id": row.source_id,
        "declaration": declaration.model_dump(mode="json"),
        "oracle_empty": {
            "original_count": evidence.original_count,
            "exclusion_counts": dict(evidence.exclusion_counts),
            "filtered_release": {
                "repo": evidence.filtered_release.repo,
                "revision": evidence.filtered_release.revision,
            },
            "evidence": evidence.validation.uri,
        },
        "inventory": None,
        "inventory_reference": None,
        "static_audit": None,
        "static_blocked_tasks": None,
        "static_reason_counts": {},
        "oracle_tasks": 0,
        "population": None,
        "public_preparation": None,
        "old_task_specs_or_count_bindings_inherited": False,
        "qa_bindings_inherited": False,
        "source_commands": 0,
        "model_requests": 0,
    }
    key = canonical_sha256(row.source_id)[:20]
    result_ref = seal(f"{output_root}/results/{key}.json", result)
    return {**result, "result_ref": {"uri": result_ref.uri, "sha256": result_ref.sha256}}


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
    empty_oracle = _oracle_empty_evidence(source)
    if empty_oracle is not None:
        return _measure_oracle_empty(row, source, empty_oracle, output_root, revision)
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


class _WorkerRegistry:
    """The worker's source registry, built once per worker process.

    The census runs at a frozen source revision: the receipts behind the registry
    are write-once artifacts, so every source a worker measures resolves against
    the same registry. One build per worker is faithful to the serial semantics
    and keeps the per-source resolve cost from repeating 321 times.
    """

    def __init__(self, fn: Callable[[], dict]) -> None:
        self._fn = fn
        self._registry: dict | None = None

    def __call__(self) -> dict:
        if self._registry is None:
            self._registry = self._fn()
        return self._registry

    def __reduce__(self):
        # A fresh cache per worker process, even if the parent built one already.
        return (_WorkerRegistry, (self._fn,))


def _initialize_census_worker() -> None:
    """One core per census worker: the job's cores are the workers, not their threads."""
    configure_coreweave_s3()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _measure_all(
    rows: list[CensusSourceRow],
    output_root: str,
    revision: str,
    region_prefix: str,
    *,
    workers: int,
    sources_fn: Callable[[], dict] = all_sources,
    inventory_fn: Callable[..., Any] = inventory_source,
) -> list[dict]:
    """Measure every row, fanned out across ``workers`` child processes.

    The declared row order is the receipt order: ``pool.map`` preserves it. Each
    child measures whole sources independently — the per-source seals land at
    source-keyed paths, so a parallel run writes exactly what a serial run writes.
    Custom ``sources_fn``/``inventory_fn`` injections must be picklable; a serial
    run (``workers=1``) is the path for unpicklable test doubles.
    """
    if workers == 1:
        return [
            measure_source(
                row,
                output_root,
                revision,
                region_prefix,
                sources_fn=sources_fn,
                inventory_fn=inventory_fn,
            )
            for row in rows
        ]
    target = partial(
        measure_source,
        output_root=output_root,
        revision=revision,
        region_prefix=region_prefix,
        sources_fn=_WorkerRegistry(sources_fn),
        inventory_fn=inventory_fn,
    )
    # One source per dispatch: the per-source cost is heavy-tailed (the largest
    # source dominates any larger chunk), so single-source dispatches balance the tail.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=_initialize_census_worker
    ) as pool:
        return list(pool.map(target, rows, chunksize=1))


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
        measured = _measure_all(
            rows,
            output_root,
            inputs.source_revision,
            inputs.region_prefix,
            workers=inputs.workers,
            sources_fn=sources_fn,
            inventory_fn=inventory_fn,
        )
        proof: dict[str, Any] = {
            "kind": f"atlas-{inputs.release_id}-{RECEIPT_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "inputs": {"uri": inputs.inputs.uri, "sha256": inputs.inputs.sha256},
            "region_prefix": inputs.region_prefix,
            "sources": len(measured),
            "tasks": sum((entry.get("inventory") or {}).get("retained_tasks", 0) for entry in measured),
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
