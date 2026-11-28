# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Catalog staging: assemble and verify the staged inputs for the catalog load.

One maintained module replaces the hand-assembled staging-refs step inside the
per-release catalog-load launchers (see INVENTORY.md). It verifies that every
staged input — declarations, publication receipt, history-staging receipt, prior
QC browse completion, fresh QC index and derived contracts — is sealed, digest
intact, and coherent in its typed load and snapshot IDs, then seals the catalog
plan the load consumes. The deeper load-time validation stays in
``catalog_load.py``; this stage refuses inconsistent staging before any SQL stage
runs (DESIGN.md §8b discipline).
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from finestore.rl.curation import CurationCatalog
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.history_staging import (
    DEFAULT_APPLET_ID,
    DEFAULT_SERVICE_URL,
)
from experiments.post_training.task_curation.release_stages.manifest import (
    ArtifactRef,
    read_ref,
    sealed_contract_rows,
    verify_population_coverage,
)
from experiments.post_training.task_curation.source_status import served_snapshot_sources
from infra.marina.apps.rl_data_catalog.server.schema import SCHEMA_SQL
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

CATALOG_STAGE_SUFFIX = "catalog-staging"
REQUIRED_STAGING_REFS = (
    "declarations",
    "publication",
    "history_receipt",
    "prior_qc_completion",
    "fresh_qc_index",
    "contracts",
)


@dataclass(frozen=True)
class CatalogStageInputs:
    """Typed inputs for one catalog-staging run; every value is a graph output."""

    release_id: str
    prior_load_id: str
    target_load_id: str
    snapshot_id: str
    source_revision: str
    catalog_root: str
    output_root: str
    expected_source_count: int
    expected_browse_source_count: int
    expected_contract_count: int
    expected_contract_tasks: int
    declarations: ArtifactRef
    publication: ArtifactRef
    history_receipt: ArtifactRef
    prior_qc_completion: ArtifactRef
    fresh_qc_index: ArtifactRef
    contracts: ArtifactRef
    migrations: tuple[str, ...] = ()
    service_url: str = DEFAULT_SERVICE_URL
    applet_id: str = DEFAULT_APPLET_ID

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        for name in ("prior_load_id", "target_load_id"):
            value = getattr(self, name)
            if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a 32-hex load ID, got {value!r}")
        if not self.release_id or not self.snapshot_id or not self.source_revision:
            raise ValueError("release_id, snapshot_id and source_revision are required")
        if not self.catalog_root or not self.output_root:
            raise ValueError("catalog_root and output_root are required")
        if self.expected_source_count <= 0 or self.expected_browse_source_count < 0:
            raise ValueError("source count must be positive and browse count nonnegative")
        if self.expected_contract_count <= 0 or self.expected_contract_tasks <= 0:
            raise ValueError("expected contract counts must be positive")


def verify_staged_load_ids(inputs: CatalogStageInputs, staged: dict[str, dict]) -> None:
    """Every staged input must name this release's loads and pointer state."""
    history = staged["history_receipt"]
    if history["load_id"] != inputs.target_load_id or history["prior_load_id"] != inputs.prior_load_id:
        raise ValueError(
            f"history receipt targets {history['load_id']!r} from {history['prior_load_id']!r};"
            f" expected {inputs.target_load_id!r} from {inputs.prior_load_id!r}"
        )
    if history.get("selected_pointer_changed"):
        raise ValueError("history receipt reports a changed selection pointer; staging must not build on it")
    browse = staged["prior_qc_completion"]
    if browse["status"] != "PASS" or browse["source_count"] != inputs.expected_browse_source_count:
        raise ValueError(
            f"prior QC browse evidence is {browse.get('status')!r} over {browse.get('source_count')} sources"
        )
    if browse["target_load_id"] != inputs.target_load_id or browse["source_load_id"] != inputs.prior_load_id:
        raise ValueError("prior QC browse evidence names different loads than this release")
    fresh_qc = staged["fresh_qc_index"]
    if fresh_qc["load_id"] != inputs.target_load_id or fresh_qc["snapshot_id"] != inputs.snapshot_id:
        raise ValueError("fresh QC index names different loads than this release")


def verify_staged_snapshot(inputs: CatalogStageInputs, staged: dict[str, dict]) -> list[str]:
    """The catalog snapshot must exist with the declared population at the planned ID."""
    catalog = CurationCatalog(inputs.catalog_root)
    snapshot = catalog.current()
    if snapshot.snapshot_id != inputs.snapshot_id:
        raise ValueError(f"catalog snapshot is {snapshot.snapshot_id!r}, expected {inputs.snapshot_id!r}")
    # The snapshot keeps original source ids; declarations and surveys use canonical ones.
    sources = [canonical_source_id(row.id) for row in served_snapshot_sources(catalog, inputs.snapshot_id)]
    if len(sources) != inputs.expected_source_count:
        raise ValueError(f"snapshot holds {len(sources)} sources, expected {inputs.expected_source_count}")
    declarations = read_ref(inputs.declarations)
    if set(declarations) != set(sources):
        raise ValueError(
            f"declarations and snapshot disagree (declarations {len(declarations)}, snapshot {len(sources)})"
        )
    publication = read_ref(inputs.publication)
    backend_ids = {row["source_id"] for row in read_ref(ArtifactRef(**publication["backend"]))["sources"]}
    identity_ids = set(read_ref(ArtifactRef(**publication["identity"]))["fingerprints"])
    verify_population_coverage(declarations, identity_ids, backend_ids)
    del staged
    return sorted(sources)


def verify_staged_contracts(inputs: CatalogStageInputs, staged: dict[str, dict]) -> int:
    """The derived contracts must cover the planned measured-task total.

    The typed receipt validation itself belongs to the load stage; this plan stage
    refuses a staged document whose totals already disagree.
    """
    rows = sealed_contract_rows(staged["contracts"])
    if len(rows) != inputs.expected_contract_count:
        raise ValueError(f"{len(rows)} derived contracts, expected {inputs.expected_contract_count}")
    measured = sum(row["input"]["measured_tasks"] for row in rows)
    if measured != inputs.expected_contract_tasks:
        raise ValueError(f"derived contracts measure {measured} tasks, expected {inputs.expected_contract_tasks}")
    return measured


def verify_migrations(inputs: CatalogStageInputs) -> None:
    """Every planned migration must be part of the in-repo schema."""
    for position, migration in enumerate(inputs.migrations):
        if migration + ";" not in SCHEMA_SQL:
            raise ValueError(f"migration {position} is not part of the in-repo schema: {migration!r}")


def run_catalog_stage(
    inputs: CatalogStageInputs,
    *,
    dry_run: bool,
    feeds: tuple | None = None,
) -> dict:
    """Assemble and verify the staged catalog inputs; ``dry_run`` writes nothing."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "catalog_stage", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        staged = {
            name: read_ref(getattr(inputs, name))
            for name in (
                "declarations",
                "publication",
                "history_receipt",
                "prior_qc_completion",
                "fresh_qc_index",
                "contracts",
            )
        }
        verify_staged_load_ids(inputs, staged)
        source_ids = verify_staged_snapshot(inputs, staged)
        measured = verify_staged_contracts(inputs, staged)
        verify_migrations(inputs)
        plan = {
            "kind": f"atlas-{inputs.release_id}-{CATALOG_STAGE_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "catalog_root": inputs.catalog_root,
            "snapshot_id": inputs.snapshot_id,
            "prior_load_id": inputs.prior_load_id,
            "target_load_id": inputs.target_load_id,
            "source_count": len(source_ids),
            "browse_source_count": inputs.expected_browse_source_count,
            "contract_count": inputs.expected_contract_count,
            "contract_tasks": measured,
            "migrations": list(inputs.migrations),
            "refs": {
                name: {"uri": getattr(inputs, name).uri, "sha256": getattr(inputs, name).sha256}
                for name in REQUIRED_STAGING_REFS
            },
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if dry_run:
            alerts.stage_finished(
                inputs.release_id, "catalog_stage", {"mode": mode, "source_count": len(source_ids)}, alert_feeds
            )
            return plan
        raw = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
        path = StoragePath(f"{inputs.output_root}/catalog-plan.json")
        if path.exists():
            raise ValueError(f"catalog plan already exists at {path}; bump the release prefix instead of overwriting")
        path.write_bytes(raw, auto_mkdir=True)
        if path.read_bytes() != raw:
            raise ValueError("catalog plan readback mismatch")
        plan["receipt"] = {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        plan["completed_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        alerts.stage_finished(
            inputs.release_id,
            "catalog_stage",
            {"mode": mode, "source_count": len(source_ids), **plan["receipt"]},
            alert_feeds,
        )
        return plan
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "catalog_stage", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON CatalogStageInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    for name in REQUIRED_STAGING_REFS:
        document[name] = ArtifactRef(**document[name])
    inputs = CatalogStageInputs(**document)
    proof = run_catalog_stage(inputs, dry_run=args.mode == "dry-run")
    label = "CATALOG-STAGE-DRY-RUN" if args.mode == "dry-run" else "CATALOG-STAGE-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
