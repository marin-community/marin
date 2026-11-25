# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""QC browse: copy the prior release's selected review panels onto the target load.

One maintained module replaces the per-release QC-browse plan and copy workers (see
INVENTORY.md). The plan step enumerates the prior load's selected review panels and
validates every one against the frozen declarations document and the prior load's
stored bindings; the copy step replicates the four review tables with exact SQL
equality and publishes each presentation pointer only after its panel verifies.
Release-identifying values arrive as typed :class:`QcBrowseInputs` and are checked
against the plan and the live pointer before any write (DESIGN.md §4, §8b).

Per DESIGN.md §8b the plan step IS the dry run: zero SQL writes and no plan seal.
The copy step has no partial-write mode.
"""

import argparse
import hashlib
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.release_stages.paced_query import (
    MAX_ATTEMPTS_PER_STATEMENT,
    MAX_RETRIES_PER_RUN,
    RETRY_WAIT_SECONDS,
    RetryControl,
    open_paced_query,
)
from infra.marina.apps.rl_data_catalog.server.content_access import content_is_withheld

Query = Callable[[str, dict[str, Any]], dict[str, Any]]

DEFAULT_SERVICE_URL = "https://marina.oa.dev"
DEFAULT_APPLET_ID = "c9742a15-fb46-4d56-bc88-aa916351ebd3"

PLAN_SUFFIX = "qc-browse-copy-plan"
COMPLETION_SUFFIX = "qc-browse-retention"
BROWSE_DIR = "prior-qc-browse"
PLAN_PAGE_ROWS = 20
PAGE_BYTES = 1024 * 1024
# Recorded in the completion document; the values come from the paced query client.
RETRY_POLICY = {
    "wait_seconds": RETRY_WAIT_SECONDS,
    "per_statement_retries": MAX_ATTEMPTS_PER_STATEMENT,
    "shared_run_retries": MAX_RETRIES_PER_RUN,
}

LATEST_POINTER_SQL = "SELECT load_id FROM atlas_current WHERE name='latest'"
PANEL_PAGE_SQL = (
    "SELECT source_id,pipeline_key,receipt FROM atlas_review_current WHERE load_id=:load"
    " AND (CAST(:after AS TEXT) IS NULL OR source_id>CAST(:after AS TEXT)) ORDER BY source_id"
    f" LIMIT {PLAN_PAGE_ROWS}"
)
SOURCE_BINDING_SQL = (
    "SELECT jsonb_build_object('actor_census_input',declaration->'actor_census_input',"
    "'evidence_url',declaration#>'{independent_qc,evidence_url}') AS binding FROM atlas_sources"
    " WHERE load_id=:load AND id=:source"
)
PANEL_POINTER_SQL = "SELECT pipeline_key,receipt FROM atlas_review_current WHERE load_id=:load AND source_id=:source"
PUBLISH_POINTER_SQL = (
    "INSERT INTO atlas_review_current (load_id,source_id,pipeline_key,receipt)"
    " VALUES (:new,:source,:key,CAST(:receipt AS JSONB)) ON CONFLICT DO NOTHING"
)


@dataclass(frozen=True)
class CopyTable:
    """One review table copied per panel with keyset-paginated reads and a page-row cap."""

    name: str
    columns: tuple[str, ...]
    keys: tuple[str, ...]
    page_rows: int
    size: str


COPY_TABLES = (
    CopyTable(
        "atlas_review_batches", ("batch_key", "record", "chunk_count"), ("batch_key",), 64, "pg_column_size(record)"
    ),
    CopyTable(
        "atlas_review_batch_chunks",
        ("batch_key", "chunk_index", "payload"),
        ("batch_key", "chunk_index"),
        8,
        "pg_column_size(payload)",
    ),
    CopyTable("atlas_review_batch_tasks", ("task_id", "batch_key"), ("task_id",), 1024, "128"),
    CopyTable(
        "atlas_raw_reviews",
        ("task_id", "record", "reviewer"),
        ("task_id",),
        1024,
        "pg_column_size(record)+COALESCE(pg_column_size(reviewer),0)",
    ),
)


@dataclass(frozen=True)
class QcBrowseInputs:
    """Typed inputs for one QC-browse run; every value is a graph output."""

    release_id: str
    prior_load_id: str
    target_load_id: str
    source_revision: str
    declarations: ArtifactRef
    output_root: str
    expected_panels: int
    # Explicit per-panel copy workers, sized by the graph from qc_browse_sizing.
    workers: int
    exclusions: tuple[str, ...] = ()
    service_url: str = DEFAULT_SERVICE_URL
    applet_id: str = DEFAULT_APPLET_ID

    def validate(self) -> None:
        """Refuse malformed inputs before any query runs."""
        for name in ("prior_load_id", "target_load_id"):
            value = getattr(self, name)
            if len(value) != 32 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be a 32-hex load ID, got {value!r}")
        if not self.release_id or not self.source_revision or not self.output_root:
            raise ValueError("release_id, source_revision and output_root are required")
        if self.expected_panels <= 0:
            raise ValueError(f"expected_panels must be positive, got {self.expected_panels}")
        if self.workers <= 0:
            raise ValueError(f"workers must be positive, got {self.workers}")


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


def read_declarations(reference: ArtifactRef) -> dict[str, dict]:
    """Digest-verify the frozen declarations document and return its per-source rows."""
    payload = StoragePath(reference.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError(f"frozen declarations digest mismatch: {reference.uri}")
    document = json.loads(payload)
    if not isinstance(document, dict) or not document:
        raise ValueError("frozen declarations document is empty")
    return document


def declaration_binding(declaration: dict) -> dict:
    """The stored atlas_sources binding a frozen declaration must equal."""
    return {
        "actor_census_input": declaration.get("actor_census_input"),
        "evidence_url": (declaration.get("independent_qc") or {}).get("evidence_url"),
    }


def selected_uri(declaration: dict) -> str:
    """The selected review artifact URI, normalized to ``.artifact.json``."""
    uri = (declaration.get("independent_qc") or {}).get("evidence_url")
    return uri if uri and uri.endswith(".artifact.json") else str(uri) + "/.artifact.json"


PANEL_PIN_FIELDS = ("dataset_id", "dataset_revision", "converter_version", "task_specs_sha256", "retained_tasks")


def pin_currency(declaration: dict, receipt: dict) -> bool:
    """Whether the panel's presentation pin tuple matches the frozen declaration's inputs."""
    panel = receipt["panel"]
    actor_input = declaration.get("actor_census_input") or {}
    return tuple(panel.get(field) for field in PANEL_PIN_FIELDS) == (
        declaration.get("dataset_id"),
        declaration.get("dataset_revision"),
        (declaration.get("pipeline") or {}).get("version"),
        actor_input.get("task_specs_sha256"),
        actor_input.get("measured_tasks"),
    )


def _check_latest_pointer(query: Query, inputs: QcBrowseInputs, when: str) -> None:
    """The prior load must still be the served selection; selection belongs to the catalog load."""
    pointer = query(LATEST_POINTER_SQL, {})["rows"]
    if pointer != [{"load_id": inputs.prior_load_id}]:
        raise ValueError(
            f"atlas_current latest is {pointer!r}, expected the prior load {inputs.prior_load_id!r}"
            f" ({when}); refusing to copy QC browse records against the wrong load"
        )


def _validate_panel(query: Query, panel: dict, declaration: dict | None, prior_load_id: str) -> dict:
    """Validate one selected panel against its frozen declaration and stored binding.

    Returns the stored binding rows so the copy can detect a declaration change.
    """
    source_id = panel["source_id"]
    if declaration is None:
        raise ValueError(f"no frozen declaration for selected panel source: {source_id}")
    if content_is_withheld(declaration):
        raise ValueError(f"selected panel content is withheld: {source_id}")
    binding = query(SOURCE_BINDING_SQL, {"load": prior_load_id, "source": source_id})["rows"]
    if binding != [{"binding": declaration_binding(declaration)}]:
        raise ValueError(f"stored declaration binding differs from the frozen declaration: {source_id}")
    receipt = panel["receipt"]
    if selected_uri(declaration) != receipt["artifact"]["uri"]:
        raise ValueError(f"selected review artifact differs from the frozen declaration: {source_id}")
    if panel["pipeline_key"] != "qc:" + receipt["artifact"]["sha256"]:
        raise ValueError(f"review pipeline does not bind the artifact hash: {source_id}")
    if pin_currency(declaration, receipt) != receipt["current_pin"]:
        raise ValueError(f"review presentation currency differs from the receipt pin: {source_id}")
    return binding


def plan_qc_browse(
    inputs: QcBrowseInputs, *, query: Query, dry_run: bool, feeds: tuple[Path, ...] | None = None
) -> dict:
    """Enumerate and validate the prior load's selected panels; the plan step is the dry run.

    In dry-run the plan is computed and validated with zero writes and returned in the
    proof; in write mode it is sealed write-once under the output root.
    """
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "qc_browse_plan", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        configure_coreweave_s3()
        declarations = read_declarations(inputs.declarations)
        _check_latest_pointer(query, inputs, "before planning")
        panels: list[dict] = []
        after = None
        while True:
            page = query(PANEL_PAGE_SQL, {"load": inputs.prior_load_id, "after": after})["rows"]
            if not page:
                break
            panels.extend(page)
            after = page[-1]["source_id"]
        excluded = set(inputs.exclusions)
        panels = [panel for panel in panels if panel["source_id"] not in excluded]
        if len({panel["source_id"] for panel in panels}) != len(panels):
            raise ValueError("selected panels repeat a source_id")
        if len(panels) != inputs.expected_panels:
            raise ValueError(f"prior load selects {len(panels)} panels, expected {inputs.expected_panels}")
        for panel in panels:
            _validate_panel(query, panel, declarations.get(panel["source_id"]), inputs.prior_load_id)
        plan: dict[str, Any] = {
            "kind": f"atlas-{inputs.release_id}-{PLAN_SUFFIX}",
            "release_id": inputs.release_id,
            "prior_load_id": inputs.prior_load_id,
            "target_load_id": inputs.target_load_id,
            "source_revision": inputs.source_revision,
            "source_count": len(panels),
            "excluded_stale_panels": list(inputs.exclusions),
            "panels": panels,
            "declarations_ref": {"uri": inputs.declarations.uri, "sha256": inputs.declarations.sha256},
            "all_selected_review_refs_unchanged": True,
            "full_actor_inputs_unchanged": True,
            "review_pin_currency_unchanged": True,
            "qa_bindings_changed": False,
            "sql_writes": 0,
        }
        if dry_run:
            _check_latest_pointer(query, inputs, "after planning, before any write")
        plan_ref = None if dry_run else seal(f"{inputs.output_root}/{BROWSE_DIR}/plan.json", plan)
        proof: dict[str, Any] = {
            "kind": plan["kind"],
            "release_id": inputs.release_id,
            "plan": plan,
            "panels": len(panels),
            "excluded": len(inputs.exclusions),
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        plan_document = None
        if plan_ref is not None:
            plan_document = {"uri": plan_ref.uri, "sha256": plan_ref.sha256}
            proof["plan_ref"] = plan_document
        else:
            proof["selected_pointer_changed"] = False
        evidence = {"mode": mode, "panels": len(panels), **({"plan": plan_document} if plan_document else {})}
        alerts.stage_finished(inputs.release_id, "qc_browse_plan", evidence, alert_feeds)
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "qc_browse_plan", {"error": str(error)[:500]}, alert_feeds)
        raise


def copy_panel(query: Query, panel: dict, prior_load_id: str, target_load_id: str, declaration: dict) -> dict:
    """Copy one panel's four review tables with exact SQL equality, then publish its pointer."""
    source_id = panel["source_id"]
    pipeline_key = panel["pipeline_key"]
    params = {"old": prior_load_id, "new": target_load_id, "key": pipeline_key, "source": source_id}
    source = _validate_panel(query, panel, declaration, prior_load_id)
    receipt = panel["receipt"]
    prior_pointer = query(PANEL_POINTER_SQL, {"load": prior_load_id, "source": source_id})["rows"]
    if prior_pointer != [{"pipeline_key": pipeline_key, "receipt": receipt}]:
        raise ValueError("Original review pointer differs: " + source_id)
    tables: dict[str, dict] = {}
    for table in COPY_TABLES:
        key_list = ",".join(table.keys)
        value_list = ",".join(table.columns)
        previous_keys = ",".join(f"previous.{column}" for column in table.keys)
        join_keys = " AND ".join(f"current.{column}=previous.{column}" for column in table.keys)
        current_values = ",".join(f"current.{column}" for column in table.columns)
        previous_values = ",".join(f"previous.{column}" for column in table.columns)
        after = None
        total = 0
        pages = 0
        while True:
            bound = dict(params)
            predicate = ""
            if after is not None:
                bound.update({f"after_{index}": after[column] for index, column in enumerate(table.keys)})
                placeholders = ",".join(f":after_{index}" for index in range(len(table.keys)))
                predicate = f" AND ({key_list}) > ({placeholders})"
            page = query(
                f"SELECT {key_list},{table.size} AS bytes FROM {table.name}"
                f" WHERE load_id=:old AND pipeline_key=:key{predicate} ORDER BY {key_list} LIMIT {table.page_rows}",
                bound,
            )["rows"]
            if not page:
                break
            selected = []
            size = 0
            for row in page:
                if selected and size + row["bytes"] > PAGE_BYTES:
                    break
                selected.append(row)
                size += row["bytes"]
            slots = []
            for index, row in enumerate(selected):
                bound.update({f"id_{index}_{column}": row[column] for column in table.keys})
                slots.append("(" + ",".join(f":id_{index}_{column}" for column in table.keys) + ")")
            in_list = f"({key_list}) IN ({','.join(slots)})"
            names = ("pipeline_key", *table.columns)
            query(
                f"INSERT INTO {table.name} (load_id,{','.join(names)})"
                f" SELECT :new,{','.join(f'previous.{column}' for column in names)} FROM {table.name} previous"
                f" WHERE previous.load_id=:old AND previous.pipeline_key=:key"
                f" AND ({previous_keys}) IN ({','.join(slots)}) ON CONFLICT DO NOTHING",
                bound,
            )
            verify = query(
                f"SELECT * FROM (WITH previous_page AS MATERIALIZED (SELECT {value_list} FROM {table.name}"
                f" WHERE load_id=:old AND pipeline_key=:key AND {in_list}),current_page AS MATERIALIZED"
                f" (SELECT {value_list} FROM {table.name} WHERE load_id=:new AND pipeline_key=:key AND {in_list})"
                f" SELECT COUNT(*) AS rows,COUNT(*) FILTER (WHERE current.{table.keys[0]} IS NULL OR"
                f" ROW({current_values}) IS DISTINCT FROM ROW({previous_values})) AS mismatches"
                f" FROM previous_page previous LEFT JOIN current_page current ON {join_keys}) AS bounded_check",
                bound,
            )["rows"][0]
            if verify != {"rows": len(selected), "mismatches": 0}:
                raise ValueError("Original review rows differ: " + source_id + "/" + table.name)
            total += len(selected)
            pages += 1
            after = selected[-1]
        counts = query(
            f"SELECT (SELECT COUNT(*) FROM {table.name} WHERE load_id=:old AND pipeline_key=:key) AS old_rows,"
            f"(SELECT COUNT(*) FROM {table.name} WHERE load_id=:new AND pipeline_key=:key) AS new_rows",
            params,
        )["rows"][0]
        if counts != {"old_rows": total, "new_rows": total}:
            raise ValueError("Review row counts differ: " + source_id + "/" + table.name)
        tables[table.name] = {"rows": total, "pages": pages, "mismatches": 0}
    if query(SOURCE_BINDING_SQL, {"load": prior_load_id, "source": source_id})["rows"] != source:
        raise ValueError("Source declaration changed during review copy: " + source_id)
    # Publish the presentation pointer only after every original row passed exact SQL equality.
    query(PUBLISH_POINTER_SQL, {**params, "receipt": json.dumps(receipt)})
    stored = query(PANEL_POINTER_SQL, {"load": target_load_id, "source": source_id})["rows"]
    if stored != [{"pipeline_key": pipeline_key, "receipt": receipt}]:
        raise ValueError("Review presentation selection differs: " + source_id)
    return {
        "source_id": source_id,
        "pipeline_key": pipeline_key,
        "original_receipt": receipt,
        "selected_receipt": receipt,
        "presentation_pin_currency_preserved": True,
        "tables": tables,
        "qa_bindings_changed": False,
        "task_or_reviewer_identities_changed": False,
    }


def _stage_panel(inputs: QcBrowseInputs, query: Query, panel: dict, declarations: dict[str, dict]) -> dict:
    """Copy one planned panel, or assert its staged receipt instead of recopying."""
    source_id = panel["source_id"]
    declaration = declarations.get(source_id)
    if declaration is None:
        raise ValueError(f"no frozen declaration for planned panel source: {source_id}")
    path = StoragePath(f"{inputs.output_root}/{BROWSE_DIR}/panels/{hashlib.sha256(source_id.encode()).hexdigest()}.json")
    if path.exists():
        result = json.loads(path.read_bytes())
        stored = query(PANEL_POINTER_SQL, {"load": inputs.target_load_id, "source": source_id})["rows"]
        if stored != [{"pipeline_key": panel["pipeline_key"], "receipt": panel["receipt"]}]:
            raise ValueError(f"staged panel pointer differs from the plan on resume: {source_id}")
        if result["qa_bindings_changed"] is not False or result["original_receipt"] != panel["receipt"]:
            raise ValueError(f"staged panel receipt disagrees with the plan on resume: {source_id}")
        return result
    result = copy_panel(query, panel, inputs.prior_load_id, inputs.target_load_id, declaration)
    result.update(
        catalog_load_id=inputs.target_load_id,
        source_revision=inputs.source_revision,
        completed_at=datetime.now(tz=UTC).isoformat(),
    )
    seal(str(path), result)
    return result


def _plan_reference(plan_ref_or_plan: ArtifactRef | dict) -> dict | None:
    """The sealed {uri, sha256} reference, or None for a plan document passed directly."""
    if isinstance(plan_ref_or_plan, dict) and "panels" in plan_ref_or_plan:
        return None
    if isinstance(plan_ref_or_plan, dict):
        return dict(plan_ref_or_plan)
    return plan_ref_or_plan.model_dump(mode="json")


def _read_plan(plan_ref_or_plan: ArtifactRef | dict) -> dict:
    """Digest-verify a sealed plan reference, or accept the plan document directly."""
    reference = _plan_reference(plan_ref_or_plan)
    if reference is None:
        assert isinstance(plan_ref_or_plan, dict)
        return plan_ref_or_plan
    payload = StoragePath(reference["uri"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
        raise ValueError(f"sealed plan digest mismatch: {reference['uri']}")
    return json.loads(payload)


def _plan_provenance(inputs: QcBrowseInputs, plan_ref_or_plan: ArtifactRef | dict, plan: dict) -> dict:
    """The plan's {uri, sha256} provenance for the completion document.

    A plan document passed directly is digested in seal format at its canonical
    output path; a sealed reference is recorded as given.
    """
    reference = _plan_reference(plan_ref_or_plan)
    if reference is not None:
        return reference
    raw = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    return {"uri": f"{inputs.output_root}/{BROWSE_DIR}/plan.json", "sha256": hashlib.sha256(raw).hexdigest()}


class _ThreadClient(threading.local):
    """One paced SQL client per copy thread, built lazily from the factory."""

    query: Query | None = None


def _copy_panels(
    inputs: QcBrowseInputs,
    plan: dict,
    declarations: dict[str, dict],
    control: RetryControl,
    query_factory: Callable[[], Query],
    workers: int,
) -> list[dict]:
    """Copy every planned panel across ``workers`` paced clients, in declared order.

    Panels are disjoint (one pipeline key each), so each thread copies its own
    panels through its own client; one failure sets the shared stop event so the
    remaining threads halt instead of retrying against a broken budget. The
    results are collected in the plan's declared order, and the per-panel seals
    stay write-once at their source-keyed paths.
    """
    clients = _ThreadClient()

    def copy_one(panel: dict) -> dict:
        if control.stop.is_set():
            raise RuntimeError("QC-browse copy stopped after another client failed")
        try:
            if clients.query is None:
                clients.query = query_factory()
            return _stage_panel(inputs, clients.query, panel, declarations)
        except BaseException:
            control.stop.set()
            raise

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="atlas-qc-browse") as executor:
        futures = [executor.submit(copy_one, panel) for panel in plan["panels"]]
        results: list[dict] = []
        failure: BaseException | None = None
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as error:
                control.stop.set()
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure
    return results


def run_qc_browse_copy(
    inputs: QcBrowseInputs,
    plan_ref_or_plan: ArtifactRef | dict,
    *,
    query: Query | None = None,
    query_factory: Callable[[], Query] | None = None,
    feeds: tuple[Path, ...] | None = None,
) -> dict:
    """Copy every planned panel, then seal the completion receipt.

    ``query`` injects the SQL client for tests; otherwise ``query_factory`` builds it,
    defaulting to a paced client on the typed service and applet with a fresh shared
    retry budget. There is no dry-run mode: the plan step is the dry run (§8b).
    """
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    alerts.stage_started(inputs.release_id, "qc_browse", {"mode": "write"}, alert_feeds)
    try:
        inputs.validate()
        configure_coreweave_s3()
        control = RetryControl()
        injected_query = query is not None
        if query is None:
            query = (
                query_factory()
                if query_factory is not None
                else open_paced_query(inputs.service_url, inputs.applet_id, control)
            )
        plan = _read_plan(plan_ref_or_plan)
        if plan.get("kind") != f"atlas-{inputs.release_id}-{PLAN_SUFFIX}":
            raise ValueError(f"plan kind {plan.get('kind')!r} does not bind release {inputs.release_id!r}")
        if plan.get("prior_load_id") != inputs.prior_load_id or plan.get("target_load_id") != inputs.target_load_id:
            raise ValueError("plan load IDs disagree with the typed inputs; refusing to copy (DESIGN.md §8b)")
        if plan.get("source_revision") != inputs.source_revision or plan.get("excluded_stale_panels") != list(
            inputs.exclusions
        ):
            raise ValueError("plan source revision or exclusions disagree with the typed inputs")
        if plan.get("declarations_ref") != {"uri": inputs.declarations.uri, "sha256": inputs.declarations.sha256}:
            raise ValueError("plan was built from a different frozen declarations document")
        if plan.get("source_count") != len(plan.get("panels", [])):
            raise ValueError("plan source_count disagrees with its panels")
        _check_latest_pointer(query, inputs, "before any copy")
        completion_path = StoragePath(f"{inputs.output_root}/{BROWSE_DIR}/completion.json")
        if completion_path.exists():
            raise FileExistsError("Verify the existing completion instead of restaging")
        declarations = read_declarations(inputs.declarations)
        if inputs.workers == 1:
            results = [_stage_panel(inputs, query, panel, declarations) for panel in plan["panels"]]
        else:
            if injected_query and query_factory is None:
                raise ValueError(
                    "a single injected client cannot serve parallel workers;" " provide query_factory (or set workers=1)"
                )
            factory = (
                query_factory
                if query_factory is not None
                else (lambda: open_paced_query(inputs.service_url, inputs.applet_id, control))
            )
            results = _copy_panels(inputs, plan, declarations, control, factory, inputs.workers)
        _check_latest_pointer(query, inputs, "after every panel")
        receipt: dict[str, Any] = {
            "kind": f"atlas-{inputs.release_id}-{COMPLETION_SUFFIX}",
            "status": "PASS",
            "release_id": inputs.release_id,
            "source_load_id": inputs.prior_load_id,
            "target_load_id": inputs.target_load_id,
            "source_revision": inputs.source_revision,
            "plan": _plan_provenance(inputs, plan_ref_or_plan, plan),
            "source_count": len(results),
            "results": sorted(results, key=lambda row: row["source_id"]),
            "qa_bindings_changed": False,
            "task_or_reviewer_identities_changed": False,
            "selected_pointer_changed": False,
            "retry_policy": RETRY_POLICY,
            "completed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        completion_ref = seal(str(completion_path), receipt)
        completion = {"uri": completion_ref.uri, "sha256": completion_ref.sha256}
        alerts.stage_finished(
            inputs.release_id,
            "qc_browse",
            {"mode": "write", "panels": len(results), "completion": completion},
            alert_feeds,
        )
        return {**receipt, "completion": completion}
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "qc_browse", {"error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    """CLI: ``--step plan`` dry-runs or seals the copy plan; ``--step copy`` is write-only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON QcBrowseInputs document")
    parser.add_argument("--step", choices=("plan", "copy"), required=True)
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    document["declarations"] = ArtifactRef(**document["declarations"])
    document["exclusions"] = tuple(document.get("exclusions") or ())
    inputs = QcBrowseInputs(**document)
    if args.step == "copy" and args.mode == "dry-run":
        raise ValueError("the QC browse copy has no dry-run mode: the plan step is the dry run (DESIGN.md §8b)")
    if args.step == "plan":
        query = open_paced_query(inputs.service_url, inputs.applet_id, RetryControl())
        proof = plan_qc_browse(inputs, query=query, dry_run=args.mode == "dry-run")
        label = "QC-BROWSE-PLAN-DRY-RUN" if args.mode == "dry-run" else "QC-BROWSE-PLAN-SEALED"
        print(f"{label} {json.dumps(proof)}", flush=True)
        return
    plan_path = f"{inputs.output_root}/{BROWSE_DIR}/plan.json"
    payload = StoragePath(plan_path).read_bytes()
    proof = run_qc_browse_copy(inputs, ArtifactRef(uri=plan_path, sha256=hashlib.sha256(payload).hexdigest()))
    print(f"QC-BROWSE-RETAINED {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
