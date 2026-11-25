#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep a Google Sheet of the Iris root jobs that run past an age threshold, and act on owner decisions.

The ``old-iris-jobs`` Loom watch (``infra/loom/watches/old-iris-jobs.md``) runs this
script. Each owner has a tab with one row per old job. Children count toward their
root, because a cancel applies to the root. The owner sets the Decision cell of a
row: ``Kill`` makes the next run cancel the job, and a revisit decision leaves the job
alone until the Revisit on date, when the run clears the decision. A job that stops running
leaves the sheet.

The sheet is the only state. A run reads every owner tab, then writes it again, so
an edit that an owner makes between the read and the write is lost.
"""

import datetime
import secrets
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass, replace

import click
import google.auth
from connectrpc.errors import ConnectError
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
KILL_DECISION = "Kill"
REVISIT_DAYS_BY_DECISION = {"Revisit in 30 days": 30, "Revisit in 365 days": 365}
DECISIONS = (KILL_DECISION, *REVISIT_DAYS_BY_DECISION)
CANCEL_REQUESTED_STATUS = "Cancel requested"
CANCEL_FAILED_STATUS = "Cancel failed"
UNKNOWN_DECISION_STATUS = "Unknown decision"
HEADER = ("Cluster", "Job", "Started", "Age (days)", "Resources", "Listed", "Decision", "Revisit on", "Status")
DECISION_COLUMN = HEADER.index("Decision")
LAST_COLUMN = chr(ord("A") + len(HEADER) - 1)
# Pixel widths of the columns whose text is longer than the default width shows.
WIDE_COLUMN_WIDTHS = {HEADER.index("Job"): 380, HEADER.index("Resources"): 420}
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
    revisit_on: str = ""
    status: str = ""

    @classmethod
    def from_cells(cls, cells: Sequence[str]) -> "SheetRow":
        # The Sheets API omits the empty cells at the end of a row.
        return cls(*cells, *[""] * (len(HEADER) - len(cells)))

    def cells(self) -> list[str]:
        return list(astuple(self))


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
    """Apply the owner's decision to the Revisit on and Status cells; a reached revisit date clears the decision."""
    if row.decision == KILL_DECISION:
        # The date of the first request stays while the job shuts down.
        status = row.status if row.status.startswith(CANCEL_REQUESTED_STATUS) else f"{CANCEL_REQUESTED_STATUS} {today}"
        return replace(row, revisit_on="", status=status)
    if row.decision in REVISIT_DAYS_BY_DECISION:
        # The status names the decision and its date, so a changed decision starts a new period.
        prefix = f"{row.decision} from "
        try:
            decided = datetime.date.fromisoformat(row.status.removeprefix(prefix))
        except ValueError:
            decided = today
        revisit_on = decided + datetime.timedelta(days=REVISIT_DAYS_BY_DECISION[row.decision])
        if revisit_on <= today:
            return replace(row, decision="", revisit_on="", status="")
        return replace(row, revisit_on=str(revisit_on), status=f"{prefix}{decided}")
    return replace(row, revisit_on="", status=UNKNOWN_DECISION_STATUS if row.decision else "")


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


def mark_failed_cancels(
    tabs: Mapping[str, Sequence[SheetRow]], failures: Mapping[str, str]
) -> dict[str, list[SheetRow]]:
    """Return ``tabs`` with the error of each failed cancel in the Status cell of its job; the next run tries again."""
    return {
        title: [
            replace(row, status=f"{CANCEL_FAILED_STATUS}: {failures[row.job]}") if row.job in failures else row
            for row in rows
        ]
        for title, rows in tabs.items()
    }


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


def _tab_format_requests(tab_id: int) -> list[dict]:
    """Return the requests that give a tab its Decision dropdown, wrapped cells, and column widths."""
    column_widths = [
        {
            "updateDimensionProperties": {
                "range": {"sheetId": tab_id, "dimension": "COLUMNS", "startIndex": column, "endIndex": column + 1},
                "properties": {"pixelSize": width},
                "fields": "pixelSize",
            }
        }
        for column, width in WIDE_COLUMN_WIDTHS.items()
    ]
    return [
        {
            "setDataValidation": {
                "range": {
                    "sheetId": tab_id,
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
        },
        {
            "repeatCell": {
                "range": {"sheetId": tab_id},
                "cell": {"userEnteredFormat": {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"}},
                "fields": "userEnteredFormat(wrapStrategy,verticalAlignment)",
            }
        },
        *column_widths,
    ]


def add_owner_tabs(session: AuthorizedSession, sheet_id: str, titles: Sequence[str]) -> None:
    """Add a formatted tab with a frozen header row for each title.

    One request adds every tab with its header, so a failure cannot leave a tab that
    ``read_owner_tabs`` does not recognize.
    """
    requests: list[dict] = []
    for title in titles:
        # The request must name the new tab's ID for the format and header requests that follow it.
        tab_id = secrets.randbelow(2**31 - 1) + 1
        requests += [
            {"addSheet": {"properties": {"sheetId": tab_id, "title": title, "gridProperties": {"frozenRowCount": 1}}}},
            *_tab_format_requests(tab_id),
            {
                "updateCells": {
                    "start": {"sheetId": tab_id, "rowIndex": 0, "columnIndex": 0},
                    "rows": [{"values": [{"userEnteredValue": {"stringValue": name}} for name in HEADER]}],
                    "fields": "userEnteredValue",
                }
            },
        ]
    if requests:
        _call(session, "POST", f"{SHEETS_API}/{sheet_id}:batchUpdate", json={"requests": requests})


def write_owner_tabs(
    session: AuthorizedSession,
    sheet_id: str,
    tabs: Mapping[str, Sequence[SheetRow]],
    previous_row_counts: Mapping[str, int],
) -> None:
    """Write the header and rows of each tab in one request.

    ``previous_row_counts`` gives the rows that each tab held; blank rows overwrite the ones no longer listed.
    """
    if not tabs:
        return
    blank = [""] * len(HEADER)
    _call(
        session,
        "POST",
        f"{SHEETS_API}/{sheet_id}/values:batchUpdate",
        json={
            # RAW keeps dates and ages as the text that the next run reads.
            "valueInputOption": "RAW",
            "data": [
                {
                    "range": _rows_range(title, 1),
                    "values": [
                        list(HEADER),
                        *(row.cells() for row in rows),
                        *[blank] * max(0, previous_row_counts.get(title, 0) - len(rows)),
                    ],
                }
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
    failures: dict[str, str] = {}
    with open_iris_client(cluster_name=cluster, workspace=None) as client:
        # ListJobs omits resources; the per-job status carries them.
        old_jobs = find_old_jobs(
            cluster,
            client.list_jobs(state=JobState.RUNNING),
            cutoff_ms,
            lambda job_id: client.job_status(job_id).resources,
        )
        # The scan takes minutes. Read the sheet after it, so the time in which an owner's edit can be lost is short.
        tabs = read_owner_tabs(session, sheet_id)
        planned, cancels = plan_tabs(old_jobs, tabs, now_ms)
        if dry_run:
            for title, rows in planned.items():
                for row in rows:
                    click.echo("\t".join([title, *row.cells()]))
        else:
            # The hub routes a cancel to the peer that runs a federated job.
            for root in cancels:
                try:
                    client.cancel_job(root)
                except ConnectError as error:
                    # The sheet update for the other jobs must still happen; the run fails after it.
                    failures[str(root)] = str(error)
    if not dry_run:
        planned = mark_failed_cancels(planned, failures)
        add_owner_tabs(session, sheet_id, [title for title in planned if title not in tabs])
        write_owner_tabs(session, sheet_id, planned, {title: len(rows) for title, rows in tabs.items()})
    listed = sum(len(rows) for rows in planned.values())
    undecided = sum(not row.decision for rows in planned.values() for row in rows)
    click.echo(
        f"{listed} old running jobs in {len(planned)} owner tabs; {undecided} without a decision; "
        f"{len(cancels)} cancels requested" + ("; dry run, nothing changed" if dry_run else "")
    )
    if failures:
        raise click.ClickException(f"{len(failures)} cancels failed: {failures}")


if __name__ == "__main__":
    main()
