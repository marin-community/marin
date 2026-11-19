# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The self-serve difficulty report: a CLI over the campaign's durable artifacts.

Answers, from artifacts alone (never a hand-edited status log):

* per solver — identity, every run with its Iris job, submit/seal evidence and
  wall time, terminal state and what "succeeded" sealed;
* per run — sources measured by that run vs inherited, pending (with reason),
  not_measured (by reason code), and its own GPU-hours and tokens rather than
  the cumulative counters the catalog rows carry across a continuation chain;
* per solver — current catalog totals over the frozen queue population;
* which runs are active right now, per solver — including runs whose output
  root was never written;
* one source across solvers with ``--source``.

Every number names its evidence. When an artifact a number needs was never
written — a run killed before its completion.json, per-run GPU-hours predating
the run-seal, a selection size requiring prepared.json — the report says so
explicitly instead of guessing.
"""

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import re
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from finestore.reader import ReadView
from finestore.rl.difficulty import MEASUREMENTS, Solver
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

DIFFICULTY_ROOT = "s3://marin-rno2a/marin/rl-data/atlas/difficulty/"
IRIS_CLUSTER = "marin"
IRIS_JOB_PREFIX = "/benfeuer/atlas-difficulty"
CATALOG_COLUMNS = (
    "source_id",
    "status",
    "reason",
    "created_at",
    "n",
    "solved",
    "pending_tasks",
    "blocked_tasks",
    "gpu_hours",
    "input_tokens",
    "output_tokens",
    "wilson_95",
)
TERMINAL_IRIS_STATES = ("succeeded", "failed", "cancelled", "killed")
# Row-delta classification: a row identical to the run's inherited base across
# every accounting column added nothing; anything else is the run's own work.
_DELTA_COLUMNS = ("n", "solved", "pending_tasks", "blocked_tasks", "gpu_hours", "input_tokens", "output_tokens")

RUN_LABEL_PATTERN = re.compile(r"^(?P<solver>hosted|small|large|snowball)-(?P<version>v[\w]+)$")
JOB_NAME_PATTERN = re.compile(
    r"^/benfeuer/atlas-difficulty-(?P<solver>hosted|small|large|snowball)"
    r"(?P<variant>-canary|-preflight)?-(?P<date>\d{8})-(?P<version>v[\w]+)$"
)
QUEUE_CACHE_ROOT = Path(tempfile.gettempdir()) / "difficulty-report-cache"
# Per-solver active summary order, matching the job and run label naming.
SOLVER_LABELS = ("hosted", "small", "large", "snowball")


class ReportError(RuntimeError):
    """The report refused to answer from the available artifacts."""


# ---------------------------------------------------------------------------
# artifact loading


def _read_json(path: StoragePath) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _file_sha256(path: StoragePath) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(source_id: str) -> str:
    return hashlib.sha256(source_id.encode()).hexdigest()


def _latest_rows(rows: Sequence[dict]) -> dict[str, dict]:
    """The latest catalog row per source id (a resumed run's catalog keeps superseded rows)."""
    latest: dict[str, dict] = {}
    for row in rows:
        prior = latest.get(row["source_id"])
        if prior is None or str(prior["created_at"]) < str(row["created_at"]):
            latest[row["source_id"]] = row
    return latest


@dataclasses.dataclass(frozen=True)
class RunArtifacts:
    """Everything one run root durably records."""

    root: str
    run_label: str
    solver: str | None
    serving: dict | None
    parent: str | None
    queue_sha256: str | None
    completion: dict | None
    run_seal: dict | None
    traffic_seal: dict | None
    rows: list[dict]

    @property
    def kind(self) -> str:
        """run, canary or preflight from the run label's own naming."""
        return "run" if RUN_LABEL_PATTERN.match(self.run_label) else "other"


# The artifact readers are I/O bound object-store reads; one run's catalog can
# take tens of seconds, so runs and directory listings load in parallel.
_LOAD_WORKERS = 16


def discover_run_roots(base: str) -> list[str]:
    """Run roots under the difficulty tree: children of version dirs carrying run receipts.

    A run root has prepared.json (the campaign's prepared receipt) or
    serving.json (the serving identity) — including runs that died between the
    two. Sample queues, qualified snapshots and audit directories have neither.
    """
    configure_coreweave_s3()
    base_path = StoragePath(base)
    versions = sorted(base_path.ls(), key=str)
    with ThreadPoolExecutor(max_workers=_LOAD_WORKERS) as pool:
        listings = list(pool.map(lambda version: sorted(version.ls(), key=str), versions))
    roots = []
    for children in listings:
        for child in children:
            if (child / "prepared.json").exists() or (child / "serving.json").exists():
                roots.append(str(child))
    return roots


def load_run(root: str) -> RunArtifacts:
    """Load one run's durable artifacts. Rows come only from the run's own catalog."""
    configure_coreweave_s3()
    path = StoragePath(root)
    serving = _read_json(path / "serving.json")
    parent_receipt = _read_json(path / "trials" / "parent-campaign.json")
    rows: list[dict] = []
    catalog = path / "trials" / "catalog"
    if catalog.exists():
        rows = [dict(row) for row in ReadView(str(catalog)).iter_rows(MEASUREMENTS, columns=list(CATALOG_COLUMNS))]
    run_label = root.rstrip("/").split("/")[-1]
    label_match = RUN_LABEL_PATTERN.match(run_label)
    # A run that died before writing serving.json still has its solver in the
    # run label (small-v17a); other labels (canaries, judge probes) carry none.
    solver = (serving or {}).get("solver") or (label_match.group("solver") if label_match else None)
    return RunArtifacts(
        root=root,
        run_label=run_label,
        solver=solver,
        serving=serving,
        parent=(parent_receipt or {}).get("root"),
        queue_sha256=(parent_receipt or {}).get("source_queue_sha256"),
        completion=_read_json(path / "completion.json"),
        run_seal=_read_json(path / "trials" / "run-seal.json"),
        traffic_seal=_read_json(path / "trials" / "model-requests-final.json"),
        rows=rows,
    )


def load_prepared_selection(root: str) -> tuple[int, tuple[str, ...]] | None:
    """The run's prepared selection size and source ids, from its prepared.json."""
    prepared = _read_json(StoragePath(root) / "prepared.json")
    if prepared is None:
        return None
    sources = prepared.get("sources", [])
    return len(sources), tuple(
        source["sample"]["source_id"] for source in sources if isinstance(source, dict) and "sample" in source
    )


def discover_queues(base: str) -> dict[str, str]:
    """Frozen queue paths by digest, from every sample version's receipt under the base."""
    configure_coreweave_s3()
    base_path = StoragePath(base)
    queues: dict[str, str] = {}
    for version in sorted(base_path.ls(), key=str):
        receipt_path = version / "samples" / "receipt.json"
        receipt = _read_json(receipt_path)
        if not isinstance(receipt, dict):
            continue
        queue_uri, queue_sha = receipt.get("queue"), receipt.get("queue_sha256")
        if queue_uri and queue_sha:
            queues.setdefault(queue_sha, queue_uri)
    return queues


# ---------------------------------------------------------------------------
# chains, classification and totals


def ancestor_latest_rows(run: RunArtifacts, by_root: dict[str, RunArtifacts]) -> dict[str, dict]:
    """The latest inherited row per source across the run's ancestor chain."""
    inherited: dict[str, dict] = {}
    cursor = run
    seen = {run.root}
    while cursor.parent is not None and cursor.parent in by_root and cursor.parent not in seen:
        cursor = by_root[cursor.parent]
        seen.add(cursor.root)
        for source_id, row in _latest_rows(cursor.rows).items():
            prior = inherited.get(source_id)
            if prior is None or str(prior["created_at"]) < str(row["created_at"]):
                inherited[source_id] = row
    return inherited


def classify_run_sources(run: RunArtifacts, by_root: dict[str, RunArtifacts]) -> dict[str, Any]:
    """Per-source accounting for one run: own vs inherited, with pending/not_measured reasons.

    With the run's own checkpointed seal (trials/run-seal.json) the own windows
    and token deltas are exact. Older runs recorded only cumulative counters,
    so their own share is derived by subtracting the latest inherited row's
    accounting; the report labels that basis.

    ``added`` counts only sources no ancestor in the run's chain measured: a
    child that re-aggregates a parent's source does work (``own``) but adds
    nothing the chain had not already measured.
    """
    inherited_base = ancestor_latest_rows(run, by_root)
    own_windows = (run.run_seal or {}).get("sources", {})
    classification: dict[str, dict[str, Any]] = {}
    for source_id, row in _latest_rows(run.rows).items():
        base = inherited_base.get(source_id)
        seal_row = own_windows.get(_source_key(source_id))
        if seal_row is not None:
            own_gpu_hours = float(seal_row.get("own_gpu_hours", 0.0))
            own_input_tokens = int(seal_row.get("own_input_tokens", 0))
            own_output_tokens = int(seal_row.get("own_output_tokens", 0))
            own = True
        elif base is None:
            own_gpu_hours = float(row["gpu_hours"] or 0)
            own_input_tokens = int(row["input_tokens"] or 0)
            own_output_tokens = int(row["output_tokens"] or 0)
            own = True
        else:
            own_gpu_hours = max(0.0, float(row.get("gpu_hours") or 0) - float(base.get("gpu_hours") or 0))
            own_input_tokens = max(0, int(row.get("input_tokens") or 0) - int(base.get("input_tokens") or 0))
            own_output_tokens = max(0, int(row.get("output_tokens") or 0) - int(base.get("output_tokens") or 0))
            own = any(
                float(row.get(name) or 0) != float(base.get(name) or 0)
                for name in _DELTA_COLUMNS
                if name in row or name in base
            )
        classification[source_id] = {
            "status": row["status"],
            "reason": row["reason"],
            "n": row["n"],
            "solved": row["solved"],
            "own": own,
            "own_gpu_hours": round(own_gpu_hours, 4),
            "own_input_tokens": own_input_tokens,
            "own_output_tokens": own_output_tokens,
        }
    exact = bool(own_windows) or run.run_seal is not None
    return {
        "sources": classification,
        "added": sum(1 for source_id in classification if source_id not in inherited_base),
        "own_basis": "exact own windows from trials/run-seal.json" if exact else "derived from cumulative row deltas",
    }


def lineage_totals(runs: Sequence[RunArtifacts], population: Sequence[str]) -> dict[str, Any]:
    """Current catalog totals over a lineage's frozen queue population."""
    latest: dict[str, dict] = {}
    for run in runs:
        for source_id, row in _latest_rows(run.rows).items():
            prior = latest.get(source_id)
            if prior is None or str(prior["created_at"]) < str(row["created_at"]):
                latest[source_id] = row
    counts: dict[str, int] = defaultdict(int)
    reasons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for source_id in population:
        row = latest.get(source_id)
        if row is None:
            counts["no_measurement_row"] += 1
            continue
        counts[row["status"]] += 1
        if row["reason"]:
            reasons[row["status"]][row["reason"]] += 1
    return {
        "population": len(population),
        "status_counts": dict(sorted(counts.items())),
        "reasons": {status: dict(sorted(items.items())) for status, items in sorted(reasons.items())},
    }


def collect_source_rows(runs: Sequence[RunArtifacts], source_ids: Sequence[str]) -> dict[str, list[dict]]:
    """Per source id: every run's latest row, with the run root attached."""
    collected: dict[str, list[dict]] = defaultdict(list)
    wanted = set(source_ids)
    for run in runs:
        for source_id, row in _latest_rows(run.rows).items():
            if source_id in wanted:
                collected[source_id].append(
                    {
                        **row,
                        "run_root": run.root,
                        "run_label": run.run_label,
                        "solver": run.solver,
                    }
                )
    return collected


# ---------------------------------------------------------------------------
# iris job state


@dataclasses.dataclass(frozen=True)
class IrisJob:
    job_id: str
    state: str
    submitted: str | None
    reason: str | None
    kind: str  # run | canary | preflight | other
    solver: str | None = None  # from JOB_NAME_PATTERN; None when the job id does not match

    @property
    def active(self) -> bool:
        return self.state.lower() not in TERMINAL_IRIS_STATES


def list_iris_jobs(limit: int = 300) -> list[IrisJob]:
    """Root-level atlas-difficulty Iris jobs from the submitting cluster's list."""
    listing = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "iris",
            f"--cluster={IRIS_CLUSTER}",
            "job",
            "list",
            "--prefix",
            IRIS_JOB_PREFIX,
            "--limit",
            str(limit),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if listing.returncode != 0:
        raise ReportError(f"iris job list failed: {listing.stderr.strip()[:400]}")
    jobs = []
    for line in listing.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith(IRIS_JOB_PREFIX):
            continue
        fields = stripped.split(maxsplit=3)
        if len(fields) < 2:
            continue
        job_id, state = fields[0], fields[1]
        if "/" in job_id[len(IRIS_JOB_PREFIX) :]:
            continue  # a child task row, not a root job
        match = JOB_NAME_PATTERN.match(job_id)
        kind = "other"
        solver = None
        if match:
            kind = {"-canary": "canary", "-preflight": "preflight"}.get(match.group("variant"), "run")
            solver = match.group("solver")
        jobs.append(
            IrisJob(
                job_id=job_id,
                state=state.lower(),
                submitted=fields[2] if len(fields) > 2 else None,
                reason=fields[3] if len(fields) > 3 else None,
                kind=kind,
                solver=solver,
            )
        )
    return jobs


def active_runs_by_solver(jobs: Sequence[IrisJob]) -> dict[str, int]:
    """Active run jobs per solver label, including runs whose output root was never written.

    Canary and preflight jobs carry a solver label but are not runs, so they
    stay out of the summary.
    """
    counts = {solver: 0 for solver in SOLVER_LABELS}
    for job in jobs:
        if job.active and job.kind == "run" and job.solver in counts:
            counts[job.solver] += 1
    return counts


def associate_jobs(runs: Sequence[RunArtifacts], jobs: Sequence[IrisJob]) -> dict[str, dict[str, Any]]:
    """Associate run roots with root Iris jobs by solver and run label.

    The job name encodes solver-date-version and each run root's label is its
    last path segment; the association is exact only when one job and one root
    share it uniquely, and the report says so otherwise.
    """
    by_label: dict[tuple[str, str], list[IrisJob]] = defaultdict(list)
    for job in jobs:
        match = JOB_NAME_PATTERN.match(job.job_id)
        if match and job.kind == "run":
            by_label[(match.group("solver"), match.group("version"))].append(job)
    associations: dict[str, dict[str, Any]] = {}
    for run in runs:
        match = RUN_LABEL_PATTERN.match(run.run_label)
        if match is None or run.solver is None:
            continue
        candidates = by_label.get((match.group("solver"), match.group("version")), [])
        if len(candidates) == 1:
            job = candidates[0]
            entry: dict[str, Any] = {
                "job_id": job.job_id,
                "state": job.state,
                "submitted": job.submitted,
                "active": job.active,
            }
            if job.reason:
                entry["reason"] = job.reason
        elif len(candidates) > 1:
            entry = {
                "job_id": None,
                "ambiguous": [job.job_id for job in candidates],
                "active": any(job.active for job in candidates),
            }
        else:
            entry = {"job_id": None, "active": False, "note": "no root Iris job matched this run label"}
        associations[run.root] = entry
    return associations


# ---------------------------------------------------------------------------
# queues


def load_queue_population(queue_path: str, expected_sha256: str | None) -> tuple[tuple[str, ...], str]:
    """The frozen queue's source ids, byte-verified, cached locally by digest."""
    configure_coreweave_s3()
    path = StoragePath(queue_path)
    digest = _file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ReportError(f"queue {queue_path} digest {digest[:12]} differs from the run binding {expected_sha256[:12]}")
    cached = QUEUE_CACHE_ROOT / f"{digest[:16]}.json"
    if cached.exists():
        return tuple(json.loads(cached.read_text())), digest
    rows = json.loads(path.read_text())
    ids = tuple(sorted(row["declaration"]["id"] for row in rows))
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(ids))
    return ids, digest


# ---------------------------------------------------------------------------
# report assembly


def build_report(
    base: str = DIFFICULTY_ROOT,
    solvers: Sequence[str] = (),
    include_prepared: bool = False,
    with_iris: bool = True,
    queue_overrides: dict[str, str] | None = None,
    source_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """The full report document; every figure carries its evidence basis."""
    configure_coreweave_s3()
    roots = discover_run_roots(base)
    with ThreadPoolExecutor(max_workers=_LOAD_WORKERS) as pool:
        loaded = list(pool.map(load_run, roots))
    if solvers:
        wanted = set(solvers)
        loaded = [run for run in loaded if run.solver in wanted]
    runs = [run for run in loaded if run.solver is not None]
    unattributed = [run for run in loaded if run.solver is None]
    by_root = {run.root: run for run in runs}
    associations = associate_jobs(runs, list_iris_jobs() if with_iris else [])

    prepared_selections: dict[str, tuple[int, tuple[str, ...]]] = {}
    if include_prepared:
        for run in runs:
            summary = load_prepared_selection(run.root)
            if summary is not None:
                prepared_selections[run.root] = summary

    queues = {**discover_queues(base), **(queue_overrides or {})}
    lineage_runs: dict[tuple[str, str], list[RunArtifacts]] = defaultdict(list)
    for run in runs:
        if run.queue_sha256 is not None:
            lineage_runs[(run.solver, run.queue_sha256)].append(run)

    solver_sections: dict[str, Any] = {}
    for solver in Solver:
        if solvers and solver.value not in wanted:
            continue
        solver_runs = sorted((run for run in runs if run.solver == solver.value), key=lambda run: run.root)
        if not solver_runs:
            continue
        per_run = [run_section(run, by_root, associations, prepared_selections.get(run.root)) for run in solver_runs]
        totals = {}
        for (run_solver, sha), group in sorted(lineage_runs.items()):
            if run_solver != solver.value:
                continue
            queue_path = queues.get(sha)
            if queue_path is None:
                totals[sha[:12]] = {
                    "queue_path": None,
                    "note": "queue file not discovered under the report base; pass --queue <sha>=<path>",
                }
                continue
            population, digest = load_queue_population(queue_path, sha)
            totals[sha[:12]] = {"queue_path": queue_path, "queue_sha256": digest, **lineage_totals(group, population)}
        solver_sections[solver.value] = {
            "runs": per_run,
            "catalog_totals_by_queue": totals,
            "identity": identity_from_serving(
                next((run.serving for run in reversed(solver_runs) if run.serving is not None), None)
            ),
        }

    jobs = list_iris_jobs() if with_iris else []
    active_jobs = [
        {
            "job_id": job.job_id,
            "state": job.state,
            "submitted": job.submitted,
            "kind": job.kind,
            "solver": job.solver,
            "reason": job.reason,
        }
        for job in jobs
        if job.active
    ]
    associated_ids = {entry.get("job_id") for entry in associations.values()}
    report = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "base": base,
        "runs_discovered": len(loaded),
        "solvers": solver_sections,
        "unattributed_runs": [
            {"root": run.root, "note": "no serving.json and a non-solver run label (judge canary or probe)"}
            for run in unattributed
        ],
        "active_iris_jobs": active_jobs,
        "active_runs_by_solver": active_runs_by_solver(jobs) if with_iris else None,
        "active_jobs_without_run_root": [job["job_id"] for job in active_jobs if job["job_id"] not in associated_ids],
    }
    if source_ids:
        rows = collect_source_rows(runs, source_ids)
        report["source_detail"] = {
            source_id: sorted(items, key=lambda item: str(item["created_at"])) for source_id, items in rows.items()
        }
        report["sources_without_rows"] = sorted(set(source_ids) - set(rows))
    return report


def run_section(
    run: RunArtifacts,
    by_root: dict[str, RunArtifacts],
    associations: dict[str, dict[str, Any]],
    prepared: tuple[int, tuple[str, ...]] | None,
) -> dict[str, Any]:
    """The per-run report section with its evidence bases."""
    classification = classify_run_sources(run, by_root)
    sources = classification["sources"]
    pending_reasons: dict[str, int] = defaultdict(int)
    not_measured_reasons: dict[str, int] = defaultdict(int)
    own_status_counts: dict[str, int] = defaultdict(int)
    for item in sources.values():
        own_status_counts[item["status"]] += 1
        if item["status"] == "pending" and item["reason"]:
            pending_reasons[item["reason"]] += 1
        if item["status"] == "not_measured" and item["reason"]:
            not_measured_reasons[item["reason"]] += 1
    rows_by_time = sorted(run.rows, key=lambda row: str(row["created_at"]))
    selection: dict[str, Any] = (
        {
            "prepared_sources": prepared[0],
            "sources_without_rows": len(set(prepared[1]) - set(sources)),
        }
        if prepared is not None
        else {
            "prepared_sources": None,
            "note": "selection size not read (pass --prepared); the run seal records it going forward",
        }
    )
    return {
        "root": run.root,
        "run_label": run.run_label,
        "kind": run.kind,
        "parent": run.parent,
        "identity": identity_from_serving(run.serving),
        "iris": associations.get(run.root),
        "sealed": {
            "completion_present": run.completion is not None,
            "completion_worker_exit_zero": (run.completion or {}).get("worker_exit") == 0,
            "final_traffic_sealed": run.traffic_seal is not None,
            "traffic_seal_reconstructed": bool((run.traffic_seal or {}).get("reconstructed")),
            "traffic_seal_tokens": (
                None
                if run.traffic_seal is None
                else {
                    "input": (run.traffic_seal or {}).get("input_tokens"),
                    "output": (run.traffic_seal or {}).get("output_tokens"),
                }
            ),
            "basis": seal_basis(run),
        },
        "catalog": {
            "rows": len(run.rows),
            "sources_with_rows": len(sources),
            "added": classification["added"],
            "aggregated_by_run": sum(1 for item in sources.values() if item["own"]),
            "aggregated_by_run_statuses": dict(sorted(own_status_counts.items())),
            "inherited": sum(1 for item in sources.values() if not item["own"]),
            "own_basis": classification["own_basis"],
            "pending_reasons": dict(sorted(pending_reasons.items())),
            "not_measured_reasons": dict(sorted(not_measured_reasons.items())),
            "own_gpu_hours": round(sum(item["own_gpu_hours"] for item in sources.values()), 4),
            "own_input_tokens": sum(item["own_input_tokens"] for item in sources.values()),
            "own_output_tokens": sum(item["own_output_tokens"] for item in sources.values()),
            "first_row_created_at": rows_by_time[0]["created_at"] if rows_by_time else None,
            "last_row_created_at": rows_by_time[-1]["created_at"] if rows_by_time else None,
            "selection": selection,
        },
        "run_seal_present": run.run_seal is not None,
    }


def identity_from_serving(serving: dict | None) -> dict | None:
    """The solver identity a run's serving receipt records."""
    if serving is None:
        return None
    identity = {
        "solver": serving.get("solver"),
        "model": serving.get("model"),
        "revision": serving.get("requested_revision"),
        "tokenizer_revision": serving.get("tokenizer_revision"),
        "chat_template_sha256": serving.get("chat_template_sha256"),
        "serving_jobs": serving.get("serving_jobs"),
    }
    judge = serving.get("judge")
    if judge:
        identity["judge"] = {
            "model": judge.get("model"),
            "revision": judge.get("revision"),
            "effective_temperature": judge.get("effective_temperature"),
        }
    probe = ((serving or {}).get("tools_thinking_probe") or {}).get("completion") or {}
    if probe.get("system_fingerprint"):
        identity["system_fingerprint"] = probe["system_fingerprint"]
    return identity


def seal_basis(run: RunArtifacts) -> str:
    """What this run's terminal state sealed, naming missing artifacts explicitly."""
    parts = []
    if run.completion is not None:
        parts.append(f"completion.json (worker_exit {run.completion.get('worker_exit')})")
    else:
        parts.append("no completion.json — the run never sealed its worker exit (killed or still running)")
    if run.traffic_seal is not None:
        parts.append(
            "final traffic sealed"
            + (" (reconstructed from per-source receipts)" if run.traffic_seal.get("reconstructed") else "")
        )
    else:
        parts.append("no trials/model-requests-final.json — the traffic seal was never written")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# text rendering


def render_text(report: dict[str, Any]) -> str:
    lines = [f"Atlas difficulty report — {report['generated_at']}", ""]
    if report.get("source_detail") is not None or report.get("sources_without_rows"):
        lines.extend(render_source_section(report))
    for solver, section in report["solvers"].items():
        lines.append(f"== {solver} ==")
        lines.append(f"  identity: {_identity_line(section['identity'])}")
        for run in section["runs"]:
            lines.extend(render_run(run))
        for sha, totals in section["catalog_totals_by_queue"].items():
            if totals.get("queue_path") is None:
                lines.append(f"  totals over queue {sha}: unavailable — {totals.get('note')}")
                continue
            counts = totals["status_counts"]
            lines.append(
                f"  totals over queue {sha} ({totals['queue_path']}, {totals['population']} sources): "
                + ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
            )
            for status, reasons in totals["reasons"].items():
                if reasons:
                    joined = ", ".join(f"{reason}: {count}" for reason, count in sorted(reasons.items()))
                    lines.append(f"    {status} reasons: {joined}")
        lines.append("")
    lines.append("== active Iris jobs ==")
    if report.get("active_runs_by_solver") is not None:
        summary = report["active_runs_by_solver"]
        lines.append("  active runs by solver: " + " ".join(f"{solver}={count}" for solver, count in summary.items()))
    if report["active_iris_jobs"]:
        for job in report["active_iris_jobs"]:
            solver = f"  solver={job['solver']}" if job.get("solver") else ""
            lines.append(
                f"  {job['job_id']}  state={job['state']}  submitted={job['submitted']}  kind={job['kind']}{solver}"
            )
    else:
        lines.append("  none")
    if report["active_jobs_without_run_root"]:
        lines.append(
            "  active jobs with no discovered run root (their output root was never written): "
            + ", ".join(report["active_jobs_without_run_root"])
        )
    for unattributed in report.get("unattributed_runs", []):
        lines.append(f"  unattributed run (no serving.json): {unattributed['root']}")
    return "\n".join(lines)


def _identity_line(identity: dict | None) -> str:
    if identity is None:
        return "unknown (no serving.json)"
    line = f"{identity.get('model')} @ {identity.get('revision')}"
    if identity.get("judge"):
        line += f" (judge: {identity['judge']['model']})"
    return line


def render_run(run: dict[str, Any]) -> list[str]:
    catalog = run["catalog"]
    iris = run["iris"] or {}
    state = iris.get("state") or "unknown"
    submitted = iris.get("submitted") or "unknown"
    label = run["run_label"] + (f" ({run['kind']})" if run["kind"] != "run" else "")
    lines = [
        f"  {label}  {run['root']}",
        f"    iris: {iris.get('job_id') or 'unassociated'} state={state} submitted={submitted}"
        + (" ACTIVE" if iris.get("active") else ""),
        f"    sealed: {run['sealed']['basis']}",
    ]
    if catalog["first_row_created_at"] and catalog["last_row_created_at"]:
        start = dt.datetime.fromisoformat(catalog["first_row_created_at"])
        end = dt.datetime.fromisoformat(catalog["last_row_created_at"])
        suffix = (
            ""
            if run["sealed"]["completion_present"]
            else " — excludes preparation and serving startup; no completion.json timestamp exists"
        )
        lines.append(f"    wall (first→last catalog row): {end - start}{suffix}")
    else:
        lines.append("    wall: unavailable — the run wrote no catalog rows")
    lines.append(
        f"    catalog: {catalog['sources_with_rows']} sources with rows; "
        f"added={catalog['added']} (no ancestor in its chain measured these); "
        f"aggregated by this run={catalog['aggregated_by_run']}"
        + (f" ({catalog['aggregated_by_run_statuses']})" if catalog.get("aggregated_by_run_statuses") else "")
        + f" inherited={catalog['inherited']} ({catalog['own_basis']})"
    )
    if catalog["pending_reasons"]:
        lines.append(
            "    pending: " + ", ".join(f"{reason} x{count}" for reason, count in catalog["pending_reasons"].items())
        )
    if catalog["not_measured_reasons"]:
        lines.append(
            "    not_measured: "
            + ", ".join(f"{reason} x{count}" for reason, count in catalog["not_measured_reasons"].items())
        )
    lines.append(
        f"    own cost: gpu-hours={catalog['own_gpu_hours']} tokens in/out="
        f"{catalog['own_input_tokens']}/{catalog['own_output_tokens']}"
    )
    return lines


def render_source_section(report: dict[str, Any]) -> list[str]:
    """One source across solvers: each run's latest row, newest evidence last."""
    lines = []
    detail = report.get("source_detail") or {}
    missing = report.get("sources_without_rows") or []
    for source_id in sorted(set(detail) | set(missing)):
        lines.append(f"== source {source_id} ==")
        if source_id in missing:
            lines.append("  no measurement row in any discovered run")
        for row in detail.get(source_id, []):
            interval = (
                f" wilson95=({row['wilson_95']['low']:.3f},{row['wilson_95']['high']:.3f})"
                if row.get("wilson_95")
                else ""
            )
            lines.append(
                f"  {row['solver']} {row['run_label']}: {row['status']}"
                + (f" ({row['reason']})" if row["reason"] else "")
                + f" n={row['n']} solved={row['solved']}{interval}"
                + f" gpu_hours={round(float(row['gpu_hours'] or 0), 3)}"
                + f" tokens={row['input_tokens']}/{row['output_tokens']} at {row['created_at']}"
            )
    return lines


# ---------------------------------------------------------------------------
# CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DIFFICULTY_ROOT, help="The difficulty artifact root to report over.")
    parser.add_argument("--solver", action="append", choices=[solver.value for solver in Solver])
    parser.add_argument("--source", action="append", help="Show one source across solvers; repeatable.")
    parser.add_argument("--prepared", action="store_true", help="Read prepared.json for exact per-run selections.")
    parser.add_argument("--no-iris", action="store_true", help="Skip the Iris job listing.")
    parser.add_argument(
        "--queue",
        action="append",
        metavar="SHA256=PATH",
        help="Queue path override for a run binding digest; repeatable.",
    )
    parser.add_argument("--json", type=Path, help="Write the JSON report here (text is always printed).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    queue_overrides = {}
    for item in args.queue or []:
        sha, _, path = item.partition("=")
        queue_overrides[sha] = path
    report = build_report(
        base=args.base,
        solvers=tuple(args.solver or ()),
        include_prepared=args.prepared,
        with_iris=not args.no_iris,
        queue_overrides=queue_overrides,
        source_ids=tuple(args.source or ()),
    )
    print(render_text(report))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
