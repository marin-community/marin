# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Catalog load: validate the sealed publication and select the target load.

One maintained module replaces the per-release ``*-catalog-load-worker-v1.py`` copies
(see INVENTORY.md). The r39 worker's shape is preserved — sealed-reference
verification, dry-run validation with zero SQL, resume-aware selection — but every
release-identifying value arrives as typed :class:`CatalogLoadInputs` from the
manifest, and the dry run checks the typed target IDs before any write
(DESIGN.md §8b).

The ``retain_prior_assessments`` monkeypatch preserves the r39 mechanism for routing
``load_catalog``'s internal history copy through the paced client; flagged for the
r39 builder's review — the long-term fix is an explicit retain parameter on
``load_catalog``.
"""

import argparse
import gzip
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from finestore.rl.curation import CurationCatalog
from marina.client import AppletQuery
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.backend_requirements import backend_schema_sha256

import experiments.post_training.task_curation.release_stages.alerts as alerts
import infra.marina.apps.rl_data_catalog.load as load_module
from experiments.post_training.task_curation.catalog_identity import (
    canonical_backend_survey,
    canonical_identity_survey,
    canonical_snapshot_source,
)
from experiments.post_training.task_curation.derived_contract_census import DerivedContractReceipt
from experiments.post_training.task_curation.difficulty.seal import verify_snapshot
from experiments.post_training.task_curation.release_stages.admission_check import review_admissions
from experiments.post_training.task_curation.release_stages.history_staging import (
    DEFAULT_APPLET_ID,
    DEFAULT_SERVICE_URL,
    Query,
    history_staging_sizing,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref, sealed_contract_rows
from experiments.post_training.task_curation.release_stages.paced_query import RetryControl, open_paced_query
from experiments.post_training.task_curation.source_status import served_snapshot_sources
from infra.marina.apps.rl_data_catalog.load import CatalogDisplayPolicy
from infra.marina.apps.rl_data_catalog.load_derived_contract import load_contracts
from infra.marina.apps.rl_data_catalog.load_difficulty import publish_coverage
from infra.marina.apps.rl_data_catalog.load_difficulty_seal import load_difficulty_seal
from infra.marina.apps.rl_data_catalog.load_reviews import load_content_reviews
from infra.marina.apps.rl_data_catalog.parallel_history import retain_prior_assessments_parallel
from infra.marina.apps.rl_data_catalog.server.schema import SCHEMA_SQL

CATALOG_DISPLAY_POLICY = CatalogDisplayPolicy.PREVIEW
QC_PANEL_RECORDS = 100
# Signature of the history-copy hook load_catalog calls internally.
RetainCopy = Callable[[AppletQuery, Mapping[str, str], str, str], dict[str, int]]


@dataclass(frozen=True)
class CatalogLoadInputs:
    """Typed inputs for one catalog-load run; every value is a graph output."""

    release_id: str
    prior_load_id: str
    target_load_id: str
    snapshot_id: str
    source_revision: str
    catalog_root: str
    receipt_root: str
    publication: ArtifactRef
    staging: dict[str, ArtifactRef]
    expected_source_count: int
    expected_contract_count: int
    expected_contract_tasks: int
    expected_selected_panels: int
    expected_browse_source_count: int
    measured_module: str
    measured_revision: str
    measured_sha256: str
    measured_schema_sha256: str
    previously_available: tuple[str, ...] = ()
    migrations: tuple[str, ...] = ()
    service_url: str = DEFAULT_SERVICE_URL
    applet_id: str = DEFAULT_APPLET_ID
    difficulty_snapshot: ArtifactRef | None = None

    def validate(self) -> None:
        """Refuse malformed inputs before any query or read runs."""
        for name in ("prior_load_id", "target_load_id"):
            value = getattr(self, name)
            if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a 32-hex load ID, got {value!r}")
        if not self.release_id or not self.snapshot_id or not self.source_revision:
            raise ValueError("release_id, snapshot_id and source_revision are required")
        for required in ("prior_qc_completion", "fresh_qc_index", "contracts", "history_receipt"):
            if required not in self.staging:
                raise ValueError(f"staging refs must include {required!r}")
        if self.expected_source_count <= 0 or self.expected_selected_panels <= 0:
            raise ValueError("expected counts must be positive")


def read_catalog_state(catalog_root: str, inputs: CatalogLoadInputs, publication: dict) -> list:
    """Verify the sealed publication aligns with the catalog snapshot; no SQL."""
    catalog = CurationCatalog(catalog_root)
    snapshot = catalog.current()
    if snapshot.snapshot_id != inputs.snapshot_id:
        raise ValueError(f"catalog snapshot is {snapshot.snapshot_id!r}, expected {inputs.snapshot_id!r}")
    backend_document = read_ref(ArtifactRef(**publication["backend"]))
    identity_document = read_ref(ArtifactRef(**publication["identity"]))
    snapshot_sources = [canonical_snapshot_source(row) for row in served_snapshot_sources(catalog, inputs.snapshot_id)]
    source_ids = {row.id for row in snapshot_sources}
    fingerprint_ids = set(identity_document["fingerprints"])
    backend_ids = {row["source_id"] for row in backend_document["sources"]}
    if not (source_ids == fingerprint_ids == backend_ids):
        raise ValueError(
            "catalog snapshot, identity fingerprints and backend sources disagree"
            f" (snapshot {len(source_ids)}, identities {len(fingerprint_ids)}, backend {len(backend_ids)})"
        )
    if len(snapshot_sources) != inputs.expected_source_count:
        raise ValueError(f"snapshot has {len(snapshot_sources)} sources, expected {inputs.expected_source_count}")
    return snapshot_sources


def verify_measured_dependency(inputs: CatalogLoadInputs) -> None:
    """The composed surveys were measured under this schema and module revision."""
    if backend_schema_sha256() != inputs.measured_schema_sha256:
        raise ValueError(
            f"backend requirement schema {backend_schema_sha256()} differs from the measured"
            f" {inputs.measured_schema_sha256}; re-measure before loading"
        )
    model_sha256 = hashlib.sha256(Path(inputs.measured_module).resolve().read_bytes()).hexdigest()
    if model_sha256 != inputs.measured_sha256:
        raise ValueError(f"measured module digest {model_sha256} differs from the recorded {inputs.measured_sha256}")


def run_dry_run(inputs: CatalogLoadInputs) -> dict:
    """Validate every sealed reference and the catalog alignment with zero SQL.

    Coordinator design input (2026-11-23, after the catalog-load v1/v2 cluster
    failures): the dry run also runs the load-time admission gate over every source
    to be admitted and reports ALL blocked sources at once, so a blocked source
    fails the dry run — never the cluster job.
    """
    publication = read_ref(inputs.publication)
    if inputs.difficulty_snapshot is not None:
        verify_snapshot(inputs.difficulty_snapshot)
    contract_rows = sealed_contract_rows(read_ref(inputs.staging["contracts"]))
    if len(contract_rows) != inputs.expected_contract_count:
        raise ValueError(f"{len(contract_rows)} derived contracts, expected {inputs.expected_contract_count}")
    measured = sum(row["input"]["measured_tasks"] for row in contract_rows)
    if measured != inputs.expected_contract_tasks:
        raise ValueError(f"derived contracts measure {measured} tasks, expected {inputs.expected_contract_tasks}")
    verify_measured_dependency(inputs)
    for position, migration in enumerate(inputs.migrations):
        if migration + ";" not in SCHEMA_SQL:
            raise ValueError(f"migration {position} is not part of the in-repo schema: {migration!r}")
    snapshot_sources = read_catalog_state(inputs.catalog_root, inputs, publication)
    backend_document = read_ref(ArtifactRef(**publication["backend"]))
    survey = canonical_backend_survey(backend_document)
    review_admissions(
        {row.id: row.declaration.model_dump(mode="json") for row in snapshot_sources},
        survey,
        set(inputs.previously_available),
    )
    return {
        "kind": "catalog-load-dry-run",
        "release_id": inputs.release_id,
        "target_load_id": inputs.target_load_id,
        "snapshot_id": inputs.snapshot_id,
        "publication": {"uri": inputs.publication.uri, "sha256": inputs.publication.sha256},
        "staging_refs": {name: {"uri": ref.uri, "sha256": ref.sha256} for name, ref in inputs.staging.items()},
        "source_count": inputs.expected_source_count,
        "contracts": len(contract_rows),
        "sql_writes": 0,
    }


def verify_history_receipt(inputs: CatalogLoadInputs) -> dict:
    """The history-stage receipt must name this release's loads and stay pointer-clean."""
    history = read_ref(inputs.staging["history_receipt"])
    if history["load_id"] != inputs.target_load_id or history["prior_load_id"] != inputs.prior_load_id:
        raise ValueError(
            f"history receipt targets {history['load_id']!r} from {history['prior_load_id']!r};"
            f" expected {inputs.target_load_id!r} from {inputs.prior_load_id!r}"
        )
    if history.get("selected_pointer_changed"):
        raise ValueError("history receipt reports a changed selection pointer; catalog load must not build on it")
    return history


def run_catalog_load(
    inputs: CatalogLoadInputs,
    *,
    dry_run: bool,
    query: Query | None = None,
    query_factory: Callable[[], Query] | None = None,
    retain: RetainCopy | None = None,
    catalog: CurationCatalog | None = None,
    contracts_loader: Callable[..., dict] | None = None,
    reviews_loader: Callable[..., dict] | None = None,
    coverage_publisher: Callable[..., dict] | None = None,
    feeds: tuple[Path, ...] | None = None,
) -> dict:
    """Run one catalog load; ``dry_run`` validates with zero SQL (DESIGN.md §8b)."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    alerts.stage_started(inputs.release_id, "catalog_load", {"mode": "dry-run" if dry_run else "write"}, alert_feeds)
    try:
        inputs.validate()
        if dry_run:
            proof = run_dry_run(inputs)
            alerts.stage_finished(
                inputs.release_id,
                "catalog_load",
                {"mode": "dry-run", "target_load_id": inputs.target_load_id},
                alert_feeds,
            )
            return proof
        return _run_write(
            inputs,
            query,
            query_factory,
            retain,
            catalog,
            contracts_loader,
            reviews_loader,
            coverage_publisher,
            alert_feeds,
        )
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "catalog_load", {"error": str(error)[:500]}, alert_feeds)
        raise


def _run_write(
    inputs, query, query_factory, retain, catalog, contracts_loader, reviews_loader, coverage_publisher, alert_feeds
):
    control = RetryControl()
    query = query if query is not None else open_paced_query(inputs.service_url, inputs.applet_id, control)
    query_factory = (
        query_factory
        if query_factory is not None
        else (lambda: open_paced_query(inputs.service_url, inputs.applet_id, control))
    )
    selection = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
    if selection not in ([{"load_id": inputs.prior_load_id}], [{"load_id": inputs.target_load_id}]):
        raise ValueError(f"another catalog load was selected: {selection!r}")
    catalog = catalog if catalog is not None else CurationCatalog(inputs.catalog_root)
    if catalog.current().snapshot_id != inputs.snapshot_id:
        raise ValueError(f"catalog snapshot moved to {catalog.current().snapshot_id!r}")
    publication = read_ref(inputs.publication)
    backend = canonical_backend_survey(read_ref(ArtifactRef(**publication["backend"])))
    identity = canonical_identity_survey(read_ref(ArtifactRef(**publication["identity"])))
    sources = [canonical_snapshot_source(row) for row in served_snapshot_sources(catalog, inputs.snapshot_id)]
    if len(sources) != inputs.expected_source_count or {row.id for row in sources} != set(identity.fingerprints):
        raise ValueError("catalog sources and identity fingerprints disagree at write time")
    browse = read_ref(inputs.staging["prior_qc_completion"])
    if browse["status"] != "PASS" or browse["source_count"] != inputs.expected_browse_source_count:
        raise ValueError(
            f"prior QC browse evidence is {browse.get('status')!r} over {browse.get('source_count')} sources"
        )
    if browse["target_load_id"] != inputs.target_load_id or browse["source_load_id"] != inputs.prior_load_id:
        raise ValueError("prior QC browse evidence names different loads than this release")

    for migration in inputs.migrations:
        if migration + ";" not in SCHEMA_SQL:
            raise ValueError(f"migration is not part of the in-repo schema: {migration!r}")
        query(migration, {})

    contract_rows = sealed_contract_rows(read_ref(inputs.staging["contracts"]))
    contracts = [DerivedContractReceipt.model_validate(row) for row in contract_rows]
    if len(contracts) != inputs.expected_contract_count:
        raise ValueError(f"{len(contracts)} derived contracts, expected {inputs.expected_contract_count}")
    if sum(row.input.measured_tasks for row in contracts) != inputs.expected_contract_tasks:
        raise ValueError("derived contract measured tasks disagree with the release plan")
    contracts_loader = contracts_loader if contracts_loader is not None else load_contracts
    result_contracts = contracts_loader(contracts, query)

    history = verify_history_receipt(inputs)
    counts = query("SELECT COUNT(*) AS n FROM atlas_assessments WHERE load_id=:load", {"load": inputs.target_load_id})[
        "rows"
    ][0]["n"]
    headers = int(history["prior_assessment_headers"])
    if not headers <= counts <= headers + 1:
        raise ValueError(f"target load holds {counts} assessments, outside the staged history range")

    # Route load_catalog's internal history copy through the paced client; an explicit
    # retain parameter on load_catalog would remove this patch (see module docstring).
    # The cast is confined to this monkeypatch boundary; the closure matches
    # historical_assessments.retain_prior_assessments at runtime.
    sizing = history_staging_sizing(int(history["prior_item_count"]))
    paced_retain: RetainCopy
    if retain is not None:
        paced_retain = retain
    else:
        paced_retain = _default_retain(query_factory, control, sizing)
    original_retain = load_module.retain_prior_assessments
    load_module.retain_prior_assessments = cast(Any, paced_retain)
    try:
        result = load_module.load_catalog(
            catalog,
            query,
            backend,
            identity,
            display_policy=CATALOG_DISPLAY_POLICY,
            resume_load_id=inputs.target_load_id,
        )
    finally:
        load_module.retain_prior_assessments = original_retain
    selected = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
    if selected != [{"load_id": inputs.target_load_id}]:
        raise ValueError(f"serving pointer is {selected!r}, expected the target load")

    fresh_qc = read_ref(inputs.staging["fresh_qc_index"])
    if fresh_qc["load_id"] != inputs.target_load_id or fresh_qc["snapshot_id"] != inputs.snapshot_id:
        raise ValueError("fresh QC index names different loads than this release")
    reviews_loader = reviews_loader if reviews_loader is not None else load_content_reviews
    qc_results = [_stage_qc_panel(inputs, panel, reviews_loader, query) for panel in fresh_qc["receipts"]]
    selected_panels = query(
        "SELECT COUNT(*) AS n FROM atlas_review_current WHERE load_id=:load", {"load": inputs.target_load_id}
    )["rows"][0]["n"]
    if selected_panels != inputs.expected_selected_panels:
        raise ValueError(f"{selected_panels} selected review panels, expected {inputs.expected_selected_panels}")
    sealed_difficulty = None
    if inputs.difficulty_snapshot is not None:
        sealed_difficulty = load_difficulty_seal(inputs.difficulty_snapshot, query, inputs.target_load_id)
    coverage_publisher = coverage_publisher if coverage_publisher is not None else publish_coverage
    coverage_result = (
        coverage_publisher(query, inputs.target_load_id, sealed_snapshot_id=sealed_difficulty["snapshot_id"])
        if sealed_difficulty is not None
        else coverage_publisher(query, inputs.target_load_id)
    )

    proof = {
        "kind": "catalog-load",
        "release_id": inputs.release_id,
        "source_revision": inputs.source_revision,
        "publication": {"uri": inputs.publication.uri, "sha256": inputs.publication.sha256},
        "prior_load_id": inputs.prior_load_id,
        "load_id": inputs.target_load_id,
        "snapshot_id": inputs.snapshot_id,
        "contracts": result_contracts,
        "fresh_qc_results": qc_results,
        "stale_pin_panels": sorted(row["source_id"] for row in qc_results if not row["current_pin"]),
        "selected_review_panels": selected_panels,
        "difficulty_coverage": coverage_result,
        "sealed_difficulty": sealed_difficulty,
        "result": result,
        "measured_dependency": {
            "module": inputs.measured_module,
            "revision": inputs.measured_revision,
            "sha256": inputs.measured_sha256,
            "schema_sha256": inputs.measured_schema_sha256,
        },
        "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    configure_coreweave_s3()
    receipt = _write_receipt(inputs, proof)
    alerts.stage_finished(
        inputs.release_id,
        "catalog_load",
        {"mode": "write", "target_load_id": inputs.target_load_id, **receipt},
        alert_feeds,
    )
    return {**proof, "receipt": receipt}


def _default_retain(query_factory: Callable[[], Query], control: RetryControl, sizing) -> RetainCopy:
    def retain(
        query: AppletQuery,
        columns: Mapping[str, str],
        prior_load_id: str,
        load_id: str,
    ) -> dict[str, int]:
        return retain_prior_assessments_parallel(
            query,
            query_factory,
            columns,
            prior_load_id,
            load_id,
            workers=sizing.workers,
            item_batch_rows=sizing.item_batch_rows,
            stop=control.stop,
        )

    return retain


def _stage_qc_panel(inputs: CatalogLoadInputs, panel: dict, reviews_loader: Callable[..., dict], query: Query) -> dict:
    if panel["status"] != "complete" or panel["records"] != QC_PANEL_RECORDS:
        raise ValueError(f"fresh QC panel {panel['source_id']} is {panel['status']!r} with {panel['records']} records")
    raw = StoragePath(panel["reference"]["uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != panel["reference"]["sha256"]:
        raise ValueError(f"fresh QC panel digest mismatch for {panel['source_id']}")
    staged = json.loads(gzip.decompress(raw))
    if not len(staged["items"]) == len(staged["raw"]) == QC_PANEL_RECORDS:
        raise ValueError(f"fresh QC panel {panel['source_id']} is not a complete panel")
    # A panel the prior load served at an older population pin replays as stale: the
    # loader records its pin currency, matching the admission gate's sample_pin_stale.
    loaded = reviews_loader(staged, inputs.target_load_id, panel["source_id"], query)
    return {
        "source_id": panel["source_id"],
        "reference": panel["reference"],
        "counts": loaded,
        "review": staged["receipt"],
        "current_pin": bool(staged["receipt"]["current_pin"]),
    }


def _write_receipt(inputs: CatalogLoadInputs, proof: dict) -> dict:
    path = StoragePath(f"{inputs.receipt_root}/catalog-load/receipt.json")
    if path.exists():
        raise ValueError(
            f"catalog-load receipt already exists at {path}; bump the release prefix instead of overwriting"
        )
    raw = (json.dumps(proof, indent=2) + "\n").encode()
    path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError(f"receipt readback mismatch at {path}")
    return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON CatalogLoadInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["publication"] = ArtifactRef(**document["publication"])
    document["staging"] = {name: ArtifactRef(**ref) for name, ref in document["staging"].items()}
    inputs = CatalogLoadInputs(**document)
    proof = run_catalog_load(inputs, dry_run=args.mode == "dry-run")
    label = "CATALOG-LOAD-DRY-RUN" if args.mode == "dry-run" else "CATALOG-LOAD-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
