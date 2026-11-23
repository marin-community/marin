# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""History staging: copy the prior release's retained history onto the target load.

One maintained module replaces the per-release ``*-history-stage-worker-v1.py`` copies
(see INVENTORY.md). All release-identifying values arrive as typed
:class:`HistoryStagingInputs`; worker source carries no load IDs or snapshot names.

Per the coordinator's amendment (DESIGN.md §8b), this SQL-writing stage runs a
read-only dry run that plans every write and checks the typed target IDs before any
write happens — the mechanical fix for the r39 failure where a ``sed``ed worker
asserted against the wrong load.
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.paced_query import RetryControl, open_paced_query
from experiments.post_training.task_curation.release_stages.sizing import HistoryStagingSizing, history_staging_sizing
from infra.marina.apps.rl_data_catalog.load import REVIEW_COMPLETENESS_VERSION, TABLE_COLUMNS, sql_column_names
from infra.marina.apps.rl_data_catalog.parallel_history import retain_prior_assessments_parallel

Query = Callable[[str, dict], dict]

DEFAULT_SERVICE_URL = "https://marina.oa.dev"
DEFAULT_APPLET_ID = "c9742a15-fb46-4d56-bc88-aa916351ebd3"


@dataclass(frozen=True)
class HistoryStagingInputs:
    """Typed inputs for one history-staging run; every value is a graph output."""

    release_id: str
    prior_load_id: str
    target_load_id: str
    snapshot_id: str
    source_revision: str
    catalog_root: str
    receipt_root: str
    expected_prior_sources: int
    service_url: str = DEFAULT_SERVICE_URL
    applet_id: str = DEFAULT_APPLET_ID

    def validate(self) -> None:
        """Refuse malformed inputs before any query runs."""
        for name in ("prior_load_id", "target_load_id"):
            value = getattr(self, name)
            if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a 32-hex load ID, got {value!r}")
        if not self.release_id or not self.snapshot_id or not self.source_revision:
            raise ValueError("release_id, snapshot_id and source_revision are required")
        if self.expected_prior_sources <= 0:
            raise ValueError(f"expected_prior_sources must be positive, got {self.expected_prior_sources}")


@dataclass(frozen=True)
class PriorState:
    """Read-only state of the prior load, verified before anything is planned."""

    latest_load_id: str
    source_count: int
    assessment_key_count: int
    header_count: int
    item_total: int
    populations_match_summaries: bool


@dataclass(frozen=True)
class HistoryStagingPlan:
    """Every write the run would perform, computed without touching the database."""

    load_row: dict[str, str]
    prior_load_id: str
    target_load_id: str
    snapshot_id: str
    workers: int
    item_batch_rows: int
    planned_at: str


def read_prior_state(query: Query, inputs: HistoryStagingInputs) -> PriorState:
    """Verify the prior load's history is complete and internally consistent."""
    pointer = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
    if pointer != [{"load_id": inputs.prior_load_id}]:
        raise ValueError(
            f"atlas_current latest is {pointer!r}, expected the prior load {inputs.prior_load_id!r};"
            " refusing to stage history against the wrong load"
        )
    inventory = query("SELECT id,assessment_key FROM atlas_sources WHERE load_id=:load", {"load": inputs.prior_load_id})[
        "rows"
    ]
    if len(inventory) != inputs.expected_prior_sources:
        raise ValueError(
            f"prior load has {len(inventory)} sources, expected {inputs.expected_prior_sources};"
            " the census binding for the prior release does not match this staging plan"
        )
    expected = {row["assessment_key"] for row in inventory if row["assessment_key"] is not None}

    headers: dict[str, dict] = {}
    after = None
    columns = ",".join(sql_column_names(TABLE_COLUMNS["assessments"]))
    while True:
        page = query(
            f"SELECT {columns} FROM atlas_assessments WHERE load_id=:load"
            " AND (CAST(:after AS TEXT) IS NULL OR pipeline_key>CAST(:after AS TEXT))"
            " ORDER BY pipeline_key LIMIT 20",
            {"load": inputs.prior_load_id, "after": after},
        )["rows"]
        if not page:
            break
        headers.update({row["pipeline_key"]: row for row in page})
        after = page[-1]["pipeline_key"]
    missing = expected - headers.keys()
    if missing:
        raise ValueError(f"prior load lacks assessment headers for {sorted(missing)[:5]} ({len(missing)} missing)")

    counts_query = (
        "SELECT pipeline_key,cohort,summary->'verdict_counts' AS verdict_counts"
        " FROM atlas_cohort_summaries WHERE load_id=:load"
    )
    counts = query(counts_query, {"load": inputs.prior_load_id})["rows"]
    totals = {row["pipeline_key"]: 0 for row in counts}
    for row in counts:
        totals[row["pipeline_key"]] += sum(row["verdict_counts"].values())
    populations = query(
        "SELECT pipeline_key,cohort,population,sample_count FROM atlas_populations WHERE load_id=:load",
        {"load": inputs.prior_load_id},
    )["rows"]
    observed: dict[str, int] = {}
    for row in populations:
        observed[row["pipeline_key"]] = observed.get(row["pipeline_key"], 0) + row["sample_count"]
    if totals != observed:
        raise ValueError("prior cohort summary verdict totals do not match population sample counts; refusing to stage")

    return PriorState(
        latest_load_id=inputs.prior_load_id,
        source_count=len(inventory),
        assessment_key_count=len(expected),
        header_count=len(headers),
        item_total=sum(totals.values()),
        populations_match_summaries=True,
    )


def plan_writes(
    prior_state: PriorState, inputs: HistoryStagingInputs, sizing: HistoryStagingSizing
) -> HistoryStagingPlan:
    """Compute the planned writes from typed inputs only; no database access."""
    return HistoryStagingPlan(
        load_row={
            "load_id": inputs.target_load_id,
            "catalog_root": inputs.catalog_root,
            "snapshot_id": inputs.snapshot_id,
            "review_completeness_version": str(REVIEW_COMPLETENESS_VERSION),
            "display_policy": "preview",
        },
        prior_load_id=inputs.prior_load_id,
        target_load_id=inputs.target_load_id,
        snapshot_id=inputs.snapshot_id,
        workers=sizing.workers,
        item_batch_rows=sizing.item_batch_rows,
        planned_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
    )


def assert_plan_target(plan: HistoryStagingPlan, inputs: HistoryStagingInputs, prior_state: PriorState) -> None:
    """The pre-write typed-target check: plan, inputs and live pointer must agree."""
    if plan.target_load_id != inputs.target_load_id:
        raise ValueError(f"plan targets {plan.target_load_id!r} but the manifest input is {inputs.target_load_id!r}")
    if plan.prior_load_id != inputs.prior_load_id:
        raise ValueError(f"plan builds on {plan.prior_load_id!r} but the manifest input is {inputs.prior_load_id!r}")
    if plan.snapshot_id != inputs.snapshot_id:
        raise ValueError(f"plan records snapshot {plan.snapshot_id!r} but the manifest input is {inputs.snapshot_id!r}")
    if prior_state.latest_load_id != inputs.prior_load_id:
        raise ValueError(
            f"live atlas_current latest is {prior_state.latest_load_id!r}, expected prior {inputs.prior_load_id!r}"
        )


def _stage_load_row(query: Query, plan: HistoryStagingPlan) -> None:
    query(
        "INSERT INTO atlas_loads (load_id,catalog_root,snapshot_id,created_at,"
        "review_completeness_version,display_policy)"
        " VALUES (:load,:root,:snapshot,:created,:version,:policy) ON CONFLICT DO NOTHING",
        {
            "load": plan.load_row["load_id"],
            "root": plan.load_row["catalog_root"],
            "snapshot": plan.load_row["snapshot_id"],
            "created": datetime.now(tz=UTC).isoformat(),
            "version": plan.load_row["review_completeness_version"],
            "policy": plan.load_row["display_policy"],
        },
    )
    staged = query("SELECT snapshot_id FROM atlas_loads WHERE load_id=:load", {"load": plan.load_row["load_id"]})["rows"]
    if staged != [{"snapshot_id": plan.load_row["snapshot_id"]}]:
        raise ValueError(f"staged load row mismatch after insert: {staged!r}")


def _write_receipt(inputs: HistoryStagingInputs, proof: dict) -> dict:
    path = StoragePath(f"{inputs.receipt_root}/history-staging/receipt.json")
    if path.exists():
        raise ValueError(
            f"history-staging receipt already exists at {path}; bump the release prefix instead of overwriting"
        )
    raw = (json.dumps(proof, indent=2) + "\n").encode()
    path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError(f"receipt readback mismatch at {path}")
    return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def _prior_item_total(query: Query, inputs: HistoryStagingInputs) -> int:
    totals_query = (
        "SELECT pipeline_key,summary->'verdict_counts' AS verdict_counts"
        " FROM atlas_cohort_summaries WHERE load_id=:load"
    )
    rows = query(totals_query, {"load": inputs.prior_load_id})["rows"]
    total = sum(sum(row["verdict_counts"].values()) for row in rows)
    if total <= 0:
        raise ValueError("prior load reports no items; cannot size the history copy")
    return total


def _prior_headers(query: Query, inputs: HistoryStagingInputs) -> set[str]:
    rows = query("SELECT pipeline_key FROM atlas_assessments WHERE load_id=:load", {"load": inputs.prior_load_id})[
        "rows"
    ]
    return {row["pipeline_key"] for row in rows}


def run_history_staging(
    inputs: HistoryStagingInputs,
    *,
    dry_run: bool,
    sizing: HistoryStagingSizing | None = None,
    query: Query | None = None,
    query_factory: Callable[[], Query] | None = None,
    retain: Callable[..., dict[str, int]] | None = None,
    feeds: tuple[Path, ...] | None = None,
) -> dict:
    """Run one history-staging pass; ``dry_run`` plans and verifies without writing.

    Returns the proof/receipt payload. In write mode, stages the load row, copies the
    prior history in parallel, verifies the copy, and writes the receipt artifact.
    """
    inputs.validate()
    control = RetryControl()
    query = query if query is not None else open_paced_query(inputs.service_url, inputs.applet_id, control)
    query_factory = (
        query_factory
        if query_factory is not None
        else (lambda: open_paced_query(inputs.service_url, inputs.applet_id, control))
    )
    if sizing is None:
        sizing = history_staging_sizing(_prior_item_total(query, inputs))
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)

    alerts.stage_started(inputs.release_id, "history_staging", {"mode": "dry-run" if dry_run else "write"}, alert_feeds)
    try:
        prior_state = read_prior_state(query, inputs)
        plan = plan_writes(prior_state, inputs, sizing)
        assert_plan_target(plan, inputs, prior_state)
        proof: dict[str, object] = {
            "kind": "retained-history-staging",
            "release_id": inputs.release_id,
            "prior_load_id": plan.prior_load_id,
            "load_id": plan.target_load_id,
            "snapshot_id": plan.snapshot_id,
            "source_revision": inputs.source_revision,
            "prior_catalog_sources": prior_state.assessment_key_count,
            "prior_source_count": prior_state.source_count,
            "prior_assessment_headers": prior_state.header_count,
            "prior_item_count": prior_state.item_total,
            "population_and_summary_counts_match": prior_state.populations_match_summaries,
            "workers": plan.workers,
            "item_batch_rows": plan.item_batch_rows,
            "dry_run": dry_run,
        }
        if dry_run:
            pointer = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
            if pointer != [{"load_id": inputs.prior_load_id}]:
                raise ValueError(f"atlas_current changed during dry run: {pointer!r}")
            proof["selected_pointer_changed"] = False
            evidence = {"mode": "dry-run", "target_load_id": plan.target_load_id}
            alerts.stage_finished(inputs.release_id, "history_staging", evidence, alert_feeds)
            return proof

        _stage_load_row(query, plan)
        copied = (retain if retain is not None else retain_prior_assessments_parallel)(
            query,
            query_factory,
            TABLE_COLUMNS,
            plan.prior_load_id,
            plan.target_load_id,
            workers=plan.workers,
            item_batch_rows=plan.item_batch_rows,
            stop=control.stop,
        )
        copied_keys = query(
            "SELECT pipeline_key FROM atlas_assessments WHERE load_id=:load", {"load": plan.target_load_id}
        )["rows"]
        if {row["pipeline_key"] for row in copied_keys} != _prior_headers(query, inputs):
            raise ValueError("copied history does not cover every prior assessment header")
        pointer = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"]
        if pointer != [{"load_id": inputs.prior_load_id}]:
            raise ValueError(
                f"atlas_current moved during staging: {pointer!r}; selection belongs to catalog load, not staging"
            )
        proof["copied"] = copied
        proof["completed_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
        configure_coreweave_s3()
        receipt = _write_receipt(inputs, proof)
        alerts.stage_finished(
            inputs.release_id,
            "history_staging",
            {"mode": "write", "target_load_id": plan.target_load_id, **receipt},
            alert_feeds,
        )
        return {**proof, "receipt": receipt}
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "history_staging", {"error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON HistoryStagingInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    inputs = HistoryStagingInputs(**json.loads(args.inputs.read_text(encoding="utf-8")))
    proof = run_history_staging(inputs, dry_run=args.mode == "dry-run")
    label = "HISTORY-STAGING-DRY-RUN" if args.mode == "dry-run" else "HISTORY-STAGED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
