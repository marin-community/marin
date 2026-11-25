# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Main-source sync: report and queue the training sources origin/main declares.

One maintained stage closes the loop between main's RL source declarations and the
Atlas release. At a plan-pinned origin/main revision it evaluates main's
``all_sources()``, canonicalizes every ID through the alias table, diffs the
training sources against the build's declared set plus the Atlas's standing
decisions, seals a typed report of the sources new on main, and appends each new
source to the ingest queue as an unclaimed row so the subsequent build picks it up.

The stage never auto-admits anything and never removes sources. The sealed report
is write-once per release prefix: a rerun reuses it and records its own queue
action in the run proof only. The queue append is idempotent: a source the table
already names is skipped, so repeat runs and later builds make no duplicate rows.
The queue file is a write target the stage maintains, in the manner of the
watchdog feed; the stage reads it only to place its rows inside the table and to
refuse a duplicate ID — no queue content flows into any decision.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

REPORT_KIND = "atlas-main-source-sync-report"
MAIN_SYNC_QUEUE = Path(
    os.environ.get(
        "ATLAS_MAIN_SYNC_QUEUE",
        Path.home() / "Documents/experiments/active/new-rl-data-ingest/CANDIDATES.md",
    )
)
CANDIDATES_LOCK_NAME = ".candidates.lock"
EXTRACT_TIMEOUT_SECONDS = 3600
LOCK_WAIT_SECONDS = 60

# Runs inside an origin/main checkout at the pinned revision, under that tree's own
# uv environment, and prints one JSON document: main's declaration layer as rows.
# It imports only main's catalog; nothing from the release checkout leaks in.
_EXTRACT_SCRIPT = """
import json

from experiments.post_training.task_curation.sources import all_sources

rows = []
for source in all_sources().values():
    info = source.info
    dataset = source.dataset
    receipt = info.license_receipt
    use = getattr(source.config, "intended_use", None)
    rows.append(
        {
            "id": info.id,
            "dataset_repo": dataset.name if dataset else None,
            "dataset_revision": dataset.revision if dataset else None,
            "url": dataset.url if dataset else None,
            "num_tasks": info.count,
            "task_format": info.task_format.value if info.task_format else None,
            "data_restriction": info.data_restriction.value,
            "license_ids": sorted(getattr(receipt, "license_ids", []) or []) if receipt is not None else [],
            "intended_use": use.value if use is not None else None,
        }
    )
print(json.dumps(rows, sort_keys=True))
"""

# Runs under the queue lock with one JSON payload argument: it refuses an ID the
# table already names, inserts the missing rows after the table's last row, and
# replaces the file atomically.
_APPEND_SCRIPT = """
import json
import os
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
with open(payload["queue_path"], encoding="utf-8") as handle:
    lines = handle.read().splitlines(keepends=True)

def row_id(line: str) -> str:
    return line.split("|")[1].strip() if line.lstrip().startswith("|") else ""

present = {row_id(line) for line in lines}
appended, already = [], []
insert_at = max(index for index, line in enumerate(lines) if line.lstrip().startswith("|")) + 1
for row in payload["rows"]:
    if row["id"] in present:
        already.append(row["id"])
        continue
    lines.insert(insert_at, row["text"] + "\\n")
    insert_at += 1
    appended.append(row["id"])

tmp = payload["queue_path"] + ".main-sync-tmp"
with open(tmp, "w", encoding="utf-8") as handle:
    handle.write("".join(lines))
os.replace(tmp, payload["queue_path"])
print(json.dumps({"appended": appended, "already_present": already}))
"""


class MainSourceSyncPlan(BaseModel):
    """The plan's main-source sync opt-in: the pinned origin/main revision."""

    main_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    queue_path: str | None = None
    """The ingest queue to append to; the operator default (env-overridable) when None."""


class MainSourceRow(BaseModel):
    """One origin/main training source the build's decision set does not carry."""

    source_id: str
    """The canonical Atlas ID the alias table resolves main's generated ID to."""
    dataset_repo: str | None = None
    dataset_revision: str | None = None
    url: str | None = None
    num_tasks: int | None = None
    task_format: str | None = None
    data_restriction: str | None = None
    license_ids: tuple[str, ...] = ()
    intended_use: str | None = None


class MainSourceSyncReport(BaseModel):
    """The typed sync report sealed into the release's document set.

    The report carries the diff and the first run's queue action; a rerun reuses
    the sealed document and records only its own proof.
    """

    kind: Literal["atlas-main-source-sync-report"]
    release_id: str
    source_revision: str
    main_source_revision: str
    main_source_count: int
    main_train_source_count: int
    eval_source_count: int
    declared_count: int
    decided_elsewhere_count: int
    new_sources: tuple[MainSourceRow, ...]
    appended: tuple[str, ...] = ()
    already_queued: tuple[str, ...] = ()
    queue_path: str | None = None


@dataclass(frozen=True)
class MainSourceSyncInputs:
    """Typed inputs for one main-source sync; every value is a graph output."""

    release_id: str
    source_revision: str
    main_source_revision: str
    declarations: ArtifactRef
    report_root: str
    queue_path: str | None

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision or not self.report_root:
            raise ValueError("release_id, source_revision and report_root are required")
        if len(self.main_source_revision) != 40 or any(
            char not in "0123456789abcdef" for char in self.main_source_revision
        ):
            raise ValueError(f"main_source_revision must be 40-hex, got {self.main_source_revision!r}")


def canonicalized_main_rows(rows: list[dict[str, Any]]) -> dict[str, MainSourceRow]:
    """Canonicalize main's extracted rows through the alias table, keyed by Atlas ID.

    Main generates source IDs from its typed upstream keys; records published under
    historical IDs resolve through the alias table, so the diff joins both sides on
    the canonical identity.
    """
    from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id  # noqa: PLC0415

    canonical: dict[str, MainSourceRow] = {}
    for row in rows:
        source_id = canonical_source_id(row["id"])
        if source_id in canonical:
            raise ValueError(f"main declares two sources at canonical ID {source_id}: {row['id']}")
        canonical[source_id] = MainSourceRow(
            source_id=source_id,
            dataset_repo=row["dataset_repo"],
            dataset_revision=row["dataset_revision"],
            url=row["url"],
            num_tasks=row["num_tasks"],
            task_format=row["task_format"],
            data_restriction=row["data_restriction"],
            license_ids=tuple(row["license_ids"]),
            intended_use=row["intended_use"],
        )
    return canonical


def build_decision_set(declarations: dict) -> set[str]:
    """The canonical IDs the build has a decision about.

    The build's declared set (the sealed declarations document) plus the standing
    decision records: staged pending intake, retired admissions, license bans and
    declaration-only metadata exclusions. A source in any of these never re-queues.
    """
    from experiments.post_training.task_curation.metadata_exclusions import metadata_exclusions  # noqa: PLC0415
    from experiments.post_training.task_curation.retirement import retired_admission_source_ids  # noqa: PLC0415
    from experiments.post_training.task_curation.sources import PENDING_MEASUREMENT_SOURCES  # noqa: PLC0415
    from infra.marina.apps.rl_data_catalog.server.licensing import license_bans  # noqa: PLC0415
    from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id  # noqa: PLC0415

    decided = {canonical_source_id(source_id) for source_id in declarations}
    decided |= {canonical_source_id(source.info.id) for source in PENDING_MEASUREMENT_SOURCES}
    decided |= retired_admission_source_ids()
    decided |= set(license_bans())
    decided |= set(metadata_exclusions())
    return decided


def extract_main_sources(repo_root: Path, main_source_revision: str) -> list[dict[str, Any]]:
    """Evaluate origin/main's ``all_sources()`` at the pinned revision in isolation.

    The checkout must already contain the pinned revision (fetch it first); the
    stage archives that tree into a temporary directory and runs the extraction
    under the archived tree's own uv environment, so main's declarations resolve
    against main's code and never against the release checkout's.
    """
    pinned = subprocess.run(
        ["git", "cat-file", "-e", f"{main_source_revision}^{{commit}}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if pinned.returncode != 0:
        raise ValueError(
            f"the checkout does not contain the pinned main revision {main_source_revision}; fetch origin/main first"
        )
    with tempfile.TemporaryDirectory(prefix="atlas-main-sources-") as directory:
        tree = Path(directory) / "main"
        tree.mkdir()
        archive = subprocess.run(
            ["git", "archive", main_source_revision],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True, capture_output=True)
        extraction = subprocess.run(
            ["uv", "run", "--no-progress", "python", "-c", _EXTRACT_SCRIPT],
            cwd=tree,
            capture_output=True,
            text=True,
            timeout=EXTRACT_TIMEOUT_SECONDS,
            check=False,
        )
        if extraction.returncode != 0:
            detail = (extraction.stderr or extraction.stdout)[-800:]
            raise ValueError(f"main source extraction failed at {main_source_revision}: {detail}")
        return json.loads(extraction.stdout)


def append_queue_rows(rows: list[dict[str, str]], queue_path: Path) -> tuple[list[str], list[str]]:
    """Append unclaimed rows to the ingest queue under its lock; never duplicate.

    The child runs under ``lockf`` on the queue's lock with the same wait the
    ingest agents use, refuses an ID the table already names, inserts after the
    table's last row and replaces the file atomically. The stage verifies each
    appended ID landed before reporting it.
    """
    payload_path = queue_path.parent / ".main-sync-append-payload.json"
    payload_path.write_text(json.dumps({"queue_path": str(queue_path), "rows": rows}), encoding="utf-8")
    try:
        child = subprocess.run(
            [
                "/usr/bin/lockf",
                "-k",
                "-t",
                str(LOCK_WAIT_SECONDS),
                str(queue_path.parent / CANDIDATES_LOCK_NAME),
                sys.executable,
                "-c",
                _APPEND_SCRIPT,
                str(payload_path),
            ],
            capture_output=True,
            text=True,
            timeout=LOCK_WAIT_SECONDS * 2,
            check=False,
        )
    finally:
        payload_path.unlink(missing_ok=True)
    if child.returncode != 0:
        detail = (child.stderr or child.stdout)[-800:]
        raise ValueError(f"queue append failed under the lock: {detail}")
    result = json.loads(child.stdout)
    table = [
        line.split("|")[1].strip()
        for line in queue_path.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("|")
    ]
    missing = [source_id for source_id in result["appended"] if source_id not in table]
    if missing:
        raise ValueError(f"queue append readback missing rows: {missing}")
    return result["appended"], result["already_present"]


def queue_row(row: MainSourceRow) -> dict[str, str]:
    """The unclaimed ingest-queue row one new source becomes: ID, link, priority."""
    link = row.url or (
        f"https://huggingface.co/datasets/{row.dataset_repo}/tree/{row.dataset_revision}"
        if row.dataset_repo and row.dataset_revision
        else ""
    )
    pin = f" @ {row.dataset_revision[:8]}" if row.dataset_revision else ""
    tasks = f", {row.num_tasks} tasks" if row.num_tasks is not None else ""
    return {
        "id": row.source_id,
        "text": (
            f"| {row.source_id} | {link} "
            f"| P1 — new on main (main-sync stage; unclaimed); pin{pin}{tasks}; "
            f"stage-0 license gate first; graders fail closed |"
        ),
    }


def _seal_report(report: MainSourceSyncReport, report_root: str) -> dict:
    """Seal the report write-once; a rerun reuses the sealed document."""
    raw = (json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(f"{report_root}/main-source-sync-report.json")
    if path.exists():
        existing = json.loads(path.read_bytes())
        if existing.get("kind") != REPORT_KIND:
            raise ValueError(f"sealed document at {path} is not a main-source sync report")
    else:
        path.write_bytes(raw, auto_mkdir=True)
        if path.read_bytes() != raw:
            raise ValueError("main-source sync report readback mismatch")
    return {"uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run_main_source_sync(inputs: MainSourceSyncInputs, *, dry_run: bool = False, feeds=None) -> dict:
    """Diff main's training sources against the build's decision set and queue the new.

    Seals the typed report write-once under the stage's report root and returns the
    proof whose ``report`` reference the graph receipt binds. The queue append runs
    only in write mode; a dry run reports the diff without touching the queue.
    """
    inputs.validate()
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    alerts.stage_started(
        inputs.release_id,
        "main_source_sync",
        {"main_source_revision": inputs.main_source_revision, "mode": "dry-run" if dry_run else "write"},
        alert_feeds,
    )
    try:
        repo_root = Path(__file__).resolve().parents[4]
        raw_rows = extract_main_sources(repo_root, inputs.main_source_revision)
        main_rows = canonicalized_main_rows(raw_rows)

        payload = StoragePath(inputs.declarations.uri).read_bytes()
        if hashlib.sha256(payload).hexdigest() != inputs.declarations.sha256:
            raise ValueError(f"declarations digest mismatch: {inputs.declarations.uri}")
        declared = json.loads(payload)
        decided = build_decision_set(declared)
        # The diff scopes to main's training declarations: the selectable RL sources
        # the release queue exists to ingest. Eval-side sources are counted, not queued.
        new_sources = tuple(
            main_rows[source_id]
            for source_id in sorted(set(main_rows) - decided)
            if main_rows[source_id].intended_use == "train"
        )

        queue_path = Path(inputs.queue_path) if inputs.queue_path else None
        appended: list[str] = []
        already_queued: list[str] = []
        if queue_path is not None and not dry_run and new_sources:
            appended, already_queued = append_queue_rows([queue_row(row) for row in new_sources], queue_path)

        report = MainSourceSyncReport(
            kind=REPORT_KIND,
            release_id=inputs.release_id,
            source_revision=inputs.source_revision,
            main_source_revision=inputs.main_source_revision,
            main_source_count=len(main_rows),
            main_train_source_count=sum(1 for row in main_rows.values() if row.intended_use == "train"),
            eval_source_count=sum(1 for row in main_rows.values() if row.intended_use != "train"),
            declared_count=len(declared),
            decided_elsewhere_count=len(decided - set(declared)),
            new_sources=new_sources,
            appended=tuple(appended),
            already_queued=tuple(already_queued),
            queue_path=str(queue_path) if queue_path is not None else None,
        )
        receipt = None if dry_run else _seal_report(report, inputs.report_root)
        alerts.stage_finished(
            inputs.release_id,
            "main_source_sync",
            {
                "mode": "dry-run" if dry_run else "write",
                "new_sources": len(new_sources),
                "appended": len(appended),
                "already_queued": len(already_queued),
                "report": receipt,
            },
            alert_feeds,
        )
        proof = {
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "main_source_revision": inputs.main_source_revision,
            "mode": "dry-run" if dry_run else "write",
            "main_source_count": report.main_source_count,
            "main_train_source_count": report.main_train_source_count,
            "eval_source_count": report.eval_source_count,
            "declared_count": report.declared_count,
            "decided_elsewhere_count": report.decided_elsewhere_count,
            "new_sources": [row.model_dump(mode="json") for row in new_sources],
            "appended": appended,
            "already_queued": already_queued,
        }
        if receipt is not None:
            proof["report"] = receipt
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "main_source_sync", {"error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--source-revision", required=True, help="The build's landed revision")
    parser.add_argument("--main-source-revision", required=True, help="The pinned origin/main revision")
    parser.add_argument("--declarations", required=True, help="The sealed declarations document URI")
    parser.add_argument("--declarations-sha256", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--queue-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    inputs = MainSourceSyncInputs(
        release_id=args.release_id,
        source_revision=args.source_revision,
        main_source_revision=args.main_source_revision,
        declarations=ArtifactRef(uri=args.declarations, sha256=args.declarations_sha256),
        report_root=args.report_root,
        queue_path=args.queue_path,
    )
    proof = run_main_source_sync(inputs, dry_run=args.dry_run)
    print(json.dumps(proof, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
