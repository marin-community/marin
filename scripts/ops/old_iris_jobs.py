#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep a Google Sheet of the Iris root jobs that run past an age threshold, and act on owner decisions.

The ``old-iris-jobs`` Loom watch (``infra/loom/watches/old-iris-jobs.md``) runs this
script. Each owner has a tab with one row per old job. Children count toward their
root, because a cancel applies to the root. The owner sets the Decision cell of a
row: ``Kill`` makes the next run cancel the job, and the keep decision hides the
question until the Keep until date. A job that stops running leaves the sheet.

The sheet is the only state. A run reads every owner tab, then writes it again, so
an edit that an owner makes during a run can be lost.
"""

import datetime
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

import click
import google.auth
from google.auth import impersonated_credentials
from google.auth.transport.requests import AuthorizedSession
from iris.cli.connect import open_iris_client
from iris.cli.task import format_resources
from iris.client.workload import JobStatus, ResourceRequest
from iris.cluster.types import LOCAL_CLUSTER, JobName
from iris.resources.state import JobState, TaskState
from rigging.timing import Timestamp

ACTIVE_TASK_STATES = (TaskState.ASSIGNED, TaskState.BUILDING, TaskState.RUNNING)
MS_PER_DAY = 86_400_000
KEEP_DAYS = 30
KILL_DECISION = "Kill"
KEEP_DECISION = f"Keep {KEEP_DAYS} days"
DECISIONS = (KILL_DECISION, KEEP_DECISION)
CANCEL_REQUESTED_STATUS = "Cancel requested"
UNKNOWN_DECISION_STATUS = "Unknown decision"
HEADER = ("Cluster", "Job", "Started", "Age (days)", "Resources", "Listed", "Decision", "Keep until", "Status")
DECISION_COLUMN = HEADER.index("Decision")
LAST_COLUMN = chr(ord("A") + len(HEADER) - 1)
SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@dataclass(frozen=True)
class OldJob:
    """A root job running past the threshold, with the resources its active tasks hold."""

    # The scanned cluster, or the peer that runs a federated job.
    cluster: str
    root: JobName
    started_at_ms: int
    # Active task count per task request; empty while every task waits.
    occupied: Mapping[ResourceRequest, int]

    @property
    def holds_accelerator(self) -> bool:
        return any(resources.device is not None for resources in self.occupied)


@dataclass(frozen=True)
class SheetRow:
    """One job row of an owner tab. The owner edits ``decision``; the script writes the other cells."""

    cluster: str
    job: str
    started: str = ""
    age_days: str = ""
    resources: str = ""
    listed: str = ""
    decision: str = ""
    keep_until: str = ""
    status: str = ""

    @classmethod
    def from_cells(cls, cells: Sequence[str]) -> "SheetRow":
        # The Sheets API omits the empty cells at the end of a row.
        return cls(*cells, *[""] * (len(HEADER) - len(cells)))

    def cells(self) -> list[str]:
        return [
            self.cluster,
            self.job,
            self.started,
            self.age_days,
            self.resources,
            self.listed,
            self.decision,
            self.keep_until,
            self.status,
        ]


def active_task_count(job: JobStatus) -> int:
    return sum(job.task_state_counts.get(state, 0) for state in ACTIVE_TASK_STATES)


def find_old_jobs(
    scanned_cluster: str,
    running_jobs: Iterable[JobStatus],
    cutoff_ms: int,
    task_resources: Callable[[JobName], ResourceRequest],
) -> list[OldJob]:
    """Group running jobs by the cluster that runs them and their root; keep the trees started before ``cutoff_ms``.

    ``task_resources`` gives a job's per-task request; only jobs with an active task need it.
    """
    # Offset paging over a changing job set can return a job twice.
    unique = {job.job_id: job for job in running_jobs}
    trees: dict[tuple[str, JobName], list[tuple[int, JobStatus]]] = defaultdict(list)
    for job in unique.values():
        if job.started_at is None:
            continue
        cluster = scanned_cluster if job.execution_cluster_id == LOCAL_CLUSTER else job.execution_cluster_id
        trees[(cluster, job.job_id.root_job)].append((job.started_at.epoch_ms(), job))
    old_jobs = []
    for (cluster, root), members in trees.items():
        started_at_ms = min(started for started, _ in members)
        if started_at_ms >= cutoff_ms:
            continue
        occupied: Counter[ResourceRequest] = Counter()
        for _, job in members:
            if count := active_task_count(job):
                occupied[task_resources(job.job_id)] += count
        old_jobs.append(OldJob(cluster, root, started_at_ms, dict(occupied)))
    return old_jobs


def tab_title(owner: str) -> str:
    """Return the tab of an owner; an email owner uses the part before the ``@``."""
    return owner.split("@")[0]


def _decided_row(row: SheetRow, today: datetime.date) -> SheetRow:
    """Apply the owner's decision to the Keep until and Status cells; an expired keep clears the decision."""
    if row.decision == KILL_DECISION:
        # The date of the first request stays while the job shuts down.
        status = row.status if row.status.startswith(CANCEL_REQUESTED_STATUS) else f"{CANCEL_REQUESTED_STATUS} {today}"
        return replace(row, keep_until="", status=status)
    if row.decision == KEEP_DECISION:
        try:
            keep_until = datetime.date.fromisoformat(row.keep_until)
        except ValueError:
            # The cell is empty for a new decision. An owner's edit that is not an ISO date starts the period again.
            keep_until = today + datetime.timedelta(days=KEEP_DAYS)
        if keep_until <= today:
            return replace(row, decision="", keep_until="", status="")
        return replace(row, keep_until=str(keep_until), status="")
    return replace(row, keep_until="", status=UNKNOWN_DECISION_STATUS if row.decision else "")


def plan_tabs(
    old_jobs: Sequence[OldJob], tabs: Mapping[str, Sequence[SheetRow]], now_ms: int
) -> tuple[dict[str, list[SheetRow]], list[JobName]]:
    """Return the rows that each owner tab must hold, and the jobs to cancel.

    ``tabs`` holds the current rows of each owner tab. A job gets a row when it has
    an active task; the row stays through a preemption that leaves every task
    pending, and goes when the job stops running. Rows list accelerator holders
    first, then oldest first. Every tab in ``tabs`` is in the result, so a tab
    whose jobs all stopped is written empty.
    """
    today = Timestamp.from_ms(now_ms).as_naive_utc().date()
    current = {(row.cluster, row.job): row for rows in tabs.values() for row in rows}
    planned: dict[str, list[SheetRow]] = {title: [] for title in tabs}
    cancels = []
    for job in sorted(old_jobs, key=lambda job: (not job.holds_accelerator, job.started_at_ms)):
        row = current.get((job.cluster, str(job.root)))
        if row is None:
            if not job.occupied:
                continue
            row = SheetRow(job.cluster, str(job.root), listed=str(today))
        row = _decided_row(row, today)
        if row.decision == KILL_DECISION:
            cancels.append(job.root)
        resources = "; ".join(
            sorted(f"{count}x ({format_resources(request)})" for request, count in job.occupied.items())
        )
        row = replace(
            row,
            started=str(Timestamp.from_ms(job.started_at_ms).as_naive_utc().date()),
            age_days=f"{(now_ms - job.started_at_ms) / MS_PER_DAY:.1f}",
            resources=resources,
        )
        planned.setdefault(tab_title(job.root.user), []).append(row)
    return planned, cancels


def sheets_session(service_account: str) -> AuthorizedSession:
    """Return a session that calls the Sheets API as ``service_account``.

    The Loom VM's metadata token lacks the spreadsheets scope, so the VM account impersonates itself.
    """
    source, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
    return AuthorizedSession(
        impersonated_credentials.Credentials(
            source_credentials=source, target_principal=service_account, target_scopes=[SHEETS_SCOPE]
        )
    )


def _call(session: AuthorizedSession, method: str, url: str, **kwargs) -> dict:
    response = session.request(method, url, timeout=60, **kwargs)
    if not response.ok:
        raise RuntimeError(f"{method} {url} failed with {response.status_code}: {response.text}")
    return response.json()


def _rows_range(title: str, first_row: int) -> str:
    quoted = title.replace("'", "''")
    return f"'{quoted}'!A{first_row}:{LAST_COLUMN}"


def read_owner_tabs(session: AuthorizedSession, sheet_id: str) -> dict[str, list[SheetRow]]:
    """Return the job rows of each owner tab. A tab is an owner tab when its first row is the header."""
    sheets = _call(session, "GET", f"{SHEETS_API}/{sheet_id}", params={"fields": "sheets.properties.title"})["sheets"]
    titles = [sheet["properties"]["title"] for sheet in sheets]
    ranges = _call(
        session,
        "GET",
        f"{SHEETS_API}/{sheet_id}/values:batchGet",
        params={"ranges": [_rows_range(title, 1) for title in titles]},
    )["valueRanges"]
    tabs = {}
    for title, value_range in zip(titles, ranges, strict=True):
        values = value_range.get("values", [])
        if values and tuple(values[0]) == HEADER:
            tabs[title] = [SheetRow.from_cells(cells) for cells in values[1:] if cells]
    return tabs


def add_owner_tabs(session: AuthorizedSession, sheet_id: str, titles: Sequence[str]) -> None:
    """Add a tab for each title, with a frozen header row and a dropdown in the Decision column."""
    if not titles:
        return
    added = _call(
        session,
        "POST",
        f"{SHEETS_API}/{sheet_id}:batchUpdate",
        json={
            "requests": [
                {"addSheet": {"properties": {"title": title, "gridProperties": {"frozenRowCount": 1}}}}
                for title in titles
            ]
        },
    )["replies"]
    validations = [
        {
            "setDataValidation": {
                "range": {
                    "sheetId": reply["addSheet"]["properties"]["sheetId"],
                    "startRowIndex": 1,
                    "startColumnIndex": DECISION_COLUMN,
                    "endColumnIndex": DECISION_COLUMN + 1,
                },
                "rule": {
                    "condition": {
                        "type": "ONE_OF_LIST",
                        "values": [{"userEnteredValue": decision} for decision in DECISIONS],
                    },
                    "strict": True,
                    "showCustomUi": True,
                },
            }
        }
        for reply in added
    ]
    _call(session, "POST", f"{SHEETS_API}/{sheet_id}:batchUpdate", json={"requests": validations})


def write_owner_tabs(session: AuthorizedSession, sheet_id: str, tabs: Mapping[str, Sequence[SheetRow]]) -> None:
    """Replace the content of each tab in ``tabs`` with the header and its rows."""
    if not tabs:
        return
    _call(
        session,
        "POST",
        f"{SHEETS_API}/{sheet_id}/values:batchClear",
        json={"ranges": [_rows_range(title, 1) for title in tabs]},
    )
    _call(
        session,
        "POST",
        f"{SHEETS_API}/{sheet_id}/values:batchUpdate",
        json={
            # RAW keeps dates and ages as the text that the next run reads.
            "valueInputOption": "RAW",
            "data": [
                {"range": _rows_range(title, 1), "values": [list(HEADER), *(row.cells() for row in rows)]}
                for title, rows in tabs.items()
            ],
        },
    )


@click.command()
@click.option(
    "--cluster",
    required=True,
    help="Iris cluster to scan. The federation hub also covers the jobs it runs on peers.",
)
@click.option(
    "--min-age-days",
    type=click.FloatRange(min=0, min_open=True),
    required=True,
    help="List root jobs whose running tree started more than this many days ago.",
)
@click.option("--sheet-id", required=True, help="ID of the Google Sheet that holds one tab per owner.")
@click.option(
    "--impersonate",
    "service_account",
    required=True,
    help="Service account with edit access to the sheet.",
)
@click.option("--dry-run", is_flag=True, help="Print the planned rows and cancels; change no sheet and no job.")
def main(
    cluster: str,
    min_age_days: float,
    sheet_id: str,
    service_account: str,
    dry_run: bool,
) -> None:
    now_ms = Timestamp.now().epoch_ms()
    cutoff_ms = now_ms - int(min_age_days * MS_PER_DAY)
    session = sheets_session(service_account)
    tabs = read_owner_tabs(session, sheet_id)
    with open_iris_client(cluster_name=cluster, workspace=None) as client:
        # ListJobs omits resources; the per-job status carries them.
        old_jobs = find_old_jobs(
            cluster,
            client.list_jobs(state=JobState.RUNNING),
            cutoff_ms,
            lambda job_id: client.job_status(job_id).resources,
        )
        planned, cancels = plan_tabs(old_jobs, tabs, now_ms)
        if dry_run:
            for title, rows in planned.items():
                for row in rows:
                    click.echo("\t".join([title, *row.cells()]))
        else:
            # The hub routes a cancel to the peer that runs a federated job.
            for root in cancels:
                client.cancel_job(root)
    if not dry_run:
        add_owner_tabs(session, sheet_id, [title for title in planned if title not in tabs])
        write_owner_tabs(session, sheet_id, planned)
    listed = sum(len(rows) for rows in planned.values())
    undecided = sum(not row.decision for rows in planned.values() for row in rows)
    click.echo(
        f"{listed} old running jobs in {len(planned)} owner tabs; {undecided} without a decision; "
        f"{len(cancels)} cancels requested" + ("; dry run, nothing changed" if dry_run else "")
    )


if __name__ == "__main__":
    main()
