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
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
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
import infra.marina.apps.rl_data_catalog.load_reviews as load_reviews_module
from experiments.post_training.task_curation.backend_capabilities import verify_backend_blocker_reasons
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
from experiments.post_training.task_curation.release_stages.sizing import catalog_replay_sizing
from experiments.post_training.task_curation.source_status import served_snapshot_sources
from infra.marina.apps.rl_data_catalog.load import CatalogDisplayPolicy
from infra.marina.apps.rl_data_catalog.load_derived_contract import load_contracts
from infra.marina.apps.rl_data_catalog.load_difficulty import publish_coverage
from infra.marina.apps.rl_data_catalog.load_difficulty_seal import load_difficulty_seal
from infra.marina.apps.rl_data_catalog.load_reviews import (
    PreparedReviewPanel,
    finish_content_review,
    insert_review_segment,
    load_content_reviews,
    panel_row_groups,
    prepare_content_review,
)
from infra.marina.apps.rl_data_catalog.load_reviews import ensure_tables as load_review_ensure_tables
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
        if self.expected_source_count <= 0 or self.expected_selected_panels < 0:
            raise ValueError("source count must be positive and selected panel count nonnegative")


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
    verify_backend_blocker_reasons(survey)
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
    verify_backend_blocker_reasons(backend)
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
    qc_results: list[dict] = []
    if reviews_loader is not None:
        # An injected loader keeps the serial per-panel contract exactly (tests).
        qc_results = [_stage_qc_panel(inputs, panel, reviews_loader, query) for panel in fresh_qc["receipts"]]
    elif fresh_qc["receipts"]:
        staged_panels = [_validate_staged_panel(inputs, panel) for panel in fresh_qc["receipts"]]
        total = _replay_record_total(staged_panels)
        progress = _ReplayProgress(total)
        if catalog_replay_sizing(total).workers == 1:
            # One paced client: replay panel by panel, in the declared receipt order.
            for panel, staged in zip(fresh_qc["receipts"], staged_panels, strict=True):
                loaded = load_content_reviews(staged, inputs.target_load_id, panel["source_id"], query, progress)
                qc_results.append(
                    {
                        "source_id": panel["source_id"],
                        "reference": panel["reference"],
                        "counts": loaded,
                        "review": staged["receipt"],
                        "current_pin": bool(staged["receipt"]["current_pin"]),
                    }
                )
        else:
            qc_results = _replay_qc_panels_parallel(
                inputs, fresh_qc["receipts"], staged_panels, total, progress, query, query_factory, control
            )
        progress.finish()
    orphaned_panels = _drop_orphaned_review_panels(query, inputs.target_load_id)
    selected = {
        row["source_id"]
        for row in query(
            "SELECT source_id FROM atlas_review_current WHERE load_id=:load", {"load": inputs.target_load_id}
        )["rows"]
    }
    unselected = sorted({result["source_id"] for result in qc_results} - selected)
    if unselected:
        raise ValueError(f"fresh QC panels are not selected after the replay: {unselected[:5]}")
    selected_panels = len(selected)
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
        # The plan's hand-counted expectation is recorded, not enforced: it predates the
        # release's removals. The enforced invariant is above (every fresh panel selected,
        # no panel for a source outside the load).
        "plan_expected_selected_panels": inputs.expected_selected_panels,
        "orphaned_review_panels_dropped": orphaned_panels,
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


def _drop_orphaned_review_panels(query: Query, load_id: str) -> list[str]:
    """Drop selected review panels whose source is not in the load; return their sources.

    History staging carries the prior load's selections, including the release's planned
    removals; a removed source's panel must not survive into the new load.
    """
    orphans = sorted(
        row["source_id"]
        for row in query(
            "SELECT r.source_id FROM atlas_review_current r WHERE r.load_id=:load AND NOT EXISTS"
            " (SELECT 1 FROM atlas_sources s WHERE s.load_id=:load AND s.id=r.source_id)",
            {"load": load_id},
        )["rows"]
    )
    if orphans:
        query(
            "DELETE FROM atlas_review_current r WHERE r.load_id=:load AND NOT EXISTS"
            " (SELECT 1 FROM atlas_sources s WHERE s.load_id=:load AND s.id=r.source_id)",
            {"load": load_id},
        )
    return orphans


def _validate_staged_panel(inputs: CatalogLoadInputs, panel: dict) -> dict:
    """Digest-verify a fresh QC panel's staged document and check its declared scope."""
    if panel["status"] != "complete":
        raise ValueError(f"fresh QC panel {panel['source_id']} is {panel['status']!r}")
    raw = StoragePath(panel["reference"]["uri"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != panel["reference"]["sha256"]:
        raise ValueError(f"fresh QC panel digest mismatch for {panel['source_id']}")
    staged = json.loads(gzip.decompress(raw))
    # A SAMPLE panel holds its drawn sample (QC_PANEL_RECORDS, or the whole population of a
    # smaller source); a FULL panel holds every retained task.
    scope = staged["assessment"]["scope"]
    measured = staged["receipt"]["panel"]
    expected = measured["sample_size"] if scope == "sample" else measured["retained_tasks"] if scope == "full" else None
    if expected is None or (scope == "sample" and expected > QC_PANEL_RECORDS):
        raise ValueError(f"fresh QC panel {panel['source_id']} has scope {scope!r} of size {expected}")
    if not panel["records"] == len(staged["items"]) == len(staged["raw"]) == expected:
        raise ValueError(f"fresh QC panel {panel['source_id']} is not a complete panel")
    return staged


def _stage_qc_panel(inputs: CatalogLoadInputs, panel: dict, reviews_loader: Callable[..., dict], query: Query) -> dict:
    staged = _validate_staged_panel(inputs, panel)
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


class _ReplayClient(threading.local):
    """One paced SQL client per replay worker, built lazily from the factory."""

    query: Query | None = None


_REPLAY_LOG_INTERVAL_S = 30.0
# Rows per replay segment: small enough that the five monster panels cannot bound
# the makespan (the largest holds ~122k records), large enough that the per-batch
# verify statements do not dominate the statement budget.
_REPLAY_SEGMENT_ROWS = 20_000


class _ReplayProgress:
    """The replay's record progress: done/total, percent, rate and ETA at a fixed cadence.

    ``total`` is computed before the replay starts from the staged panels' row groups
    (the rows at their pins); ``done`` accumulates the rows whose insert-and-verify
    completed, across panels and threads.
    """

    def __init__(
        self, total: int, interval_s: float = _REPLAY_LOG_INTERVAL_S, log: Callable[[str], None] = logging.info
    ) -> None:
        if total <= 0:
            raise ValueError(f"the replay progress needs a positive record total, got {total}")
        self.total = total
        self.done = 0
        self._interval_s = interval_s
        self._log = log
        self._started = time.monotonic()
        self._last_logged = self._started
        self._lock = threading.Lock()

    def __call__(self, table: str, rows: int) -> None:
        with self._lock:
            self.done += rows
            due = time.monotonic() - self._last_logged >= self._interval_s
            if not due:
                return
            self._last_logged = time.monotonic()
            done, total, elapsed = self.done, self.total, time.monotonic() - self._started
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else float("inf")
        self._log(
            f"PROGRESS replay done={done}/{total} ({100 * done / total:.1f}%,"
            f" {rate:.0f} records/s, eta {duration_iso(remaining)})"
        )

    def finish(self) -> None:
        """The final line: the whole total processed, with the wall clock."""
        with self._lock:
            done, total, elapsed = self.done, self.total, time.monotonic() - self._started
        rate = done / elapsed if elapsed > 0 else 0.0
        self._log(f"PROGRESS replay done={done}/{total} (100.0%, {rate:.0f} records/s, wall {duration_iso(elapsed)})")


def duration_iso(seconds: float) -> str:
    """A bounded HH:MM:SS rendering for progress lines (ceil so an ETA never reads zero early)."""
    if seconds == float("inf"):
        return "?"
    total = math.ceil(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _replay_record_total(staged_panels: list[dict]) -> int:
    """The records the replay will process: every panel's tables at their pins.

    The staged documents are the source load's panel rows sealed at their pins, and
    the row groups are the loader's own insert plan, so this total matches what the
    loader's progress callback reports.
    """
    total = 0
    for staged in staged_panels:
        for _table, rows, _columns in panel_row_groups(staged):
            total += len(rows)
    if total <= 0:
        raise ValueError("the fresh QC panels carry no records to replay")
    return total


def _segment_rows(rows: list[dict]) -> list[list[dict]]:
    """Split one table's rows into page-range segments of at most _REPLAY_SEGMENT_ROWS."""
    return [rows[start : start + _REPLAY_SEGMENT_ROWS] for start in range(0, len(rows), _REPLAY_SEGMENT_ROWS)] or [[]]


def _replay_qc_panels_parallel(
    inputs: CatalogLoadInputs,
    panels: list[dict],
    staged_panels: list[dict],
    total: int,
    progress: _ReplayProgress,
    query: Query,
    query_factory: Callable[[], Query],
    control: RetryControl,
) -> list[dict]:
    """Replay the panels with page-range segments fanned out over paced workers.

    Each panel's insertable groups split into CopySegments; the segments of all
    panels share one worker pool, so no single panel bounds the makespan. The
    per-panel preamble (validation, chunk metadata) runs before its segments and
    the cohort summaries plus the presentation pointer publish after them — the
    ordering that matters. One failure sets the shared stop event so the peers
    halt instead of retrying against a broken budget.
    """
    workers = catalog_replay_sizing(total).workers
    load_review_ensure_tables(query, load_reviews_module.REVIEW_TABLES)
    clients = _ReplayClient()

    def client() -> Query:
        if clients.query is None:
            clients.query = query_factory()
        return clients.query

    def stop_if_halted() -> None:
        if control.stop.is_set():
            raise RuntimeError("the QC replay stopped after another client failed")

    prepared: list[PreparedReviewPanel | None] = [None] * len(panels)

    def prepare(index: int) -> int:
        stop_if_halted()
        try:
            prepared[index] = prepare_content_review(
                staged_panels[index], inputs.target_load_id, panels[index]["source_id"], client(), ensure_tables_=False
            )
        except BaseException:
            control.stop.set()
            raise
        return index

    segments: list[tuple[int, str, str, list[dict]]] = []
    for index, staged in enumerate(staged_panels):
        for table, rows, columns in panel_row_groups(staged):
            for piece in _segment_rows(rows):
                segments.append((index, table, columns, piece))

    def run_segment(segment: tuple[int, str, str, list[dict]]) -> tuple[int, str, int]:
        panel_index, table, columns, rows = segment
        stop_if_halted()
        panel = prepared[panel_index]
        if panel is None:
            raise RuntimeError(f"panel {panels[panel_index]['source_id']} never prepared")
        try:
            count = insert_review_segment(client(), inputs.target_load_id, panel.key, table, columns, rows, progress)
        except BaseException:
            control.stop.set()
            raise
        return panel_index, table, count

    def finish(panel_index: int) -> dict:
        stop_if_halted()
        panel = prepared[panel_index]
        if panel is None:
            raise RuntimeError(f"panel {panels[panel_index]['source_id']} never prepared")
        try:
            finish_content_review(client(), panel, progress)
        except BaseException:
            control.stop.set()
            raise
        return {
            "source_id": panels[panel_index]["source_id"],
            "reference": panels[panel_index]["reference"],
            "counts": counts[panel_index],
            "review": panel.staged["receipt"],
            "current_pin": bool(panel.staged["receipt"]["current_pin"]),
        }

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="atlas-replay") as executor:
        for _ in executor.map(prepare, range(len(panels))):
            pass
        counts = [{table: 0 for table, _rows, _columns in panel_row_groups(staged)} for staged in staged_panels]
        for panel_index, table, count in executor.map(run_segment, segments):
            counts[panel_index][table] += count
        results: list[dict] = []
        for result in executor.map(finish, range(len(panels))):
            results.append(result)
    return results


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
    if document.get("difficulty_snapshot") is not None:
        document["difficulty_snapshot"] = ArtifactRef(**document["difficulty_snapshot"])
    inputs = CatalogLoadInputs(**document)
    proof = run_catalog_load(inputs, dry_run=args.mode == "dry-run")
    label = "CATALOG-LOAD-DRY-RUN" if args.mode == "dry-run" else "CATALOG-LOAD-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
