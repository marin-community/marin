# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the difficulty report: per-run own-vs-inherited accounting,
population totals, Iris association, and the missing-artifact honesty rules."""

import hashlib
import json
import subprocess

import pytest
from finestore.rl.difficulty import register_difficulty
from finestore.store import DataStore

from experiments.post_training.task_curation import difficulty_report as drep


def _row(source_id: str, created_at: str, status: str = "measured", **overrides) -> dict:
    row = {
        "source_id": source_id,
        "status": status,
        "reason": None if status == "measured" else "awaiting_scored_coverage",
        "created_at": created_at,
        "n": 2,
        "solved": 1,
        "pending_tasks": 0,
        "blocked_tasks": 0,
        "gpu_hours": 0.5,
        "input_tokens": 100,
        "output_tokens": 50,
        "wilson_95": {"low": 0.1, "high": 0.9},
    }
    row.update(overrides)
    return row


def _run(root: str, rows: list[dict], **overrides) -> drep.RunArtifacts:
    fields = dict(
        root=root,
        run_label=root.rstrip("/").split("/")[-1],
        solver="hosted",
        serving={"solver": "hosted", "model": "m", "requested_revision": "r"},
        parent=None,
        queue_sha256="q" * 64,
        completion=None,
        run_seal=None,
        traffic_seal=None,
        rows=rows,
    )
    fields.update(overrides)
    return drep.RunArtifacts(**fields)


def test_classify_run_sources_splits_own_from_inherited():
    parent = _run(
        "s3://x/2026.11.14.1/hosted-v1",
        [_row("Src:a", "2026-11-14T01:00:00+00:00", gpu_hours=1.0, input_tokens=100)],
    )
    child_rows = [
        # carries the parent's totals forward and adds its own window
        _row("Src:a", "2026-11-15T01:00:00+00:00", gpu_hours=1.5, input_tokens=150),
        # measured fresh by the child
        _row("Src:b", "2026-11-15T01:00:00+00:00", gpu_hours=0.2, input_tokens=10),
        # re-aggregated with nothing new
        _row("Src:a", "2026-11-15T02:00:00+00:00", gpu_hours=1.5, input_tokens=150),
    ]
    child = _run("s3://x/2026.11.15.1/hosted-v2", child_rows, parent=parent.root)
    by_root = {parent.root: parent, child.root: child}
    classification = drep.classify_run_sources(child, by_root)
    assert classification["own_basis"] == "derived from cumulative row deltas"
    sources = classification["sources"]
    assert sources["Src:a"]["own"] is True
    assert sources["Src:a"]["own_gpu_hours"] == 0.5
    assert sources["Src:a"]["own_input_tokens"] == 50
    assert sources["Src:b"]["own"] is True
    assert sources["Src:b"]["own_gpu_hours"] == 0.2


def test_classify_marks_a_row_that_added_nothing_as_inherited():
    parent = _run("s3://x/2026.11.14.1/hosted-v1", [_row("Src:a", "2026-11-14T01:00:00+00:00")])
    child = _run("s3://x/2026.11.15.1/hosted-v2", [_row("Src:a", "2026-11-15T01:00:00+00:00")], parent=parent.root)
    classification = drep.classify_run_sources(child, {parent.root: parent, child.root: child})
    assert classification["sources"]["Src:a"]["own"] is False
    assert classification["sources"]["Src:a"]["own_gpu_hours"] == 0.0


def test_added_counts_only_sources_no_ancestor_in_the_chain_measured():
    grandparent = _run("s3://x/2026.11.13.1/hosted-v0", [_row("Src:c", "2026-11-13T01:00:00+00:00")])
    parent = _run(
        "s3://x/2026.11.14.1/hosted-v1",
        [_row("Src:a", "2026-11-14T01:00:00+00:00")],
        parent=grandparent.root,
    )
    child = _run(
        "s3://x/2026.11.15.1/hosted-v2",
        [
            # re-aggregates a source the parent measured: own work, but never added
            _row("Src:a", "2026-11-15T01:00:00+00:00", gpu_hours=1.5, input_tokens=150),
            # re-aggregates a source two steps up the chain: still not added
            _row("Src:c", "2026-11-15T01:00:00+00:00", gpu_hours=1.2, input_tokens=120),
            # measured fresh by the child
            _row("Src:b", "2026-11-15T01:00:00+00:00"),
        ],
        parent=parent.root,
    )
    classification = drep.classify_run_sources(
        child, {grandparent.root: grandparent, parent.root: parent, child.root: child}
    )
    assert classification["sources"]["Src:a"]["own"] is True
    assert classification["sources"]["Src:c"]["own"] is True
    assert classification["added"] == 1


def test_classify_uses_the_run_seal_windows_when_present():
    source_key = hashlib.sha256(b"Src:a").hexdigest()
    run = _run(
        "s3://x/2026.11.18.1/hosted-v13c",
        [_row("Src:a", "2026-11-18T01:00:00+00:00", gpu_hours=1.5, input_tokens=150)],
        run_seal={
            "sources": {source_key: {"source_id": "Src:a", "own_gpu_hours": 0.25, "own_input_tokens": 30}},
        },
    )
    classification = drep.classify_run_sources(run, {})
    assert classification["own_basis"] == "exact own windows from trials/run-seal.json"
    assert classification["sources"]["Src:a"]["own_gpu_hours"] == 0.25
    assert classification["sources"]["Src:a"]["own_input_tokens"] == 30


def test_lineage_totals_count_the_population_and_reasons():
    runs = [
        _run(
            "s3://x/2026.11.14.1/hosted-v1",
            [
                _row("Src:a", "2026-11-14T01:00:00+00:00", status="measured"),
                _row("Src:b", "2026-11-14T01:00:00+00:00", status="pending", reason="provider wait"),
                _row(
                    "Src:c",
                    "2026-11-14T01:00:00+00:00",
                    status="not_measured",
                    reason="missing_environment_artifact:prepared_image",
                ),
            ],
        )
    ]
    totals = drep.lineage_totals(runs, ("Src:a", "Src:b", "Src:c", "Src:d"))
    assert totals["population"] == 4
    assert totals["status_counts"] == {
        "measured": 1,
        "no_measurement_row": 1,
        "not_measured": 1,
        "pending": 1,
    }
    assert totals["reasons"]["not_measured"] == {"missing_environment_artifact:prepared_image": 1}


def test_lineage_totals_keep_the_latest_row_per_source():
    runs = [
        _run(
            "s3://x/2026.11.14.1/hosted-v1",
            [
                _row("Src:a", "2026-11-14T01:00:00+00:00", status="pending"),
                _row("Src:a", "2026-11-14T02:00:00+00:00", status="measured"),
            ],
        )
    ]
    totals = drep.lineage_totals(runs, ("Src:a",))
    assert totals["status_counts"] == {"measured": 1}


def test_associate_jobs_exact_and_ambiguous():
    runs = [
        _run("s3://x/2026.11.14.44/hosted-v12", []),
        _run("s3://x/2026.11.16.49/small-v17a", [], solver="small"),
        _run("s3://x/2026.11.16.50/small-v17a", [], solver="small"),
    ]
    jobs = [
        drep.IrisJob(
            "/benfeuer/atlas-difficulty-hosted-20261114-v12", "failed", "2026-11-15T14:40:05+00:00", "timeout", "run"
        ),
        drep.IrisJob(
            "/benfeuer/atlas-difficulty-small-20261116-v17a", "failed", "2026-11-16T10:00:00+00:00", None, "run"
        ),
        drep.IrisJob(
            "/benfeuer/atlas-difficulty-small-20261117-v17a", "failed", "2026-11-17T11:00:00+00:00", None, "run"
        ),
    ]
    associations = drep.associate_jobs(runs, jobs)
    assert associations["s3://x/2026.11.14.44/hosted-v12"]["job_id"] == jobs[0].job_id
    ambiguous = associations["s3://x/2026.11.16.49/small-v17a"]
    assert ambiguous["job_id"] is None
    assert len(ambiguous["ambiguous"]) == 2


def test_list_iris_jobs_attaches_the_solver_label(monkeypatch):
    listing = subprocess.CompletedProcess(
        [],
        0,
        stdout=(
            "/benfeuer/atlas-difficulty-hosted-20261118-v20  running  2026-11-18T19:00:00+00:00\n"
            "/benfeuer/atlas-difficulty-small-canary-20261117-v3  running  2026-11-17T10:00:00+00:00\n"
            "/benfeuer/atlas-difficulty/child-task  running\n"
            "unrelated-job  running\n"
        ),
        stderr="",
    )
    monkeypatch.setattr(drep.subprocess, "run", lambda *args, **kwargs: listing)
    jobs = drep.list_iris_jobs()
    assert [(job.job_id, job.solver, job.kind) for job in jobs] == [
        ("/benfeuer/atlas-difficulty-hosted-20261118-v20", "hosted", "run"),
        ("/benfeuer/atlas-difficulty-small-canary-20261117-v3", "small", "canary"),
    ]


def test_active_runs_by_solver_counts_active_run_jobs_per_solver():
    jobs = [
        drep.IrisJob(
            "/benfeuer/atlas-difficulty-hosted-20261118-v20",
            "running",
            "2026-11-18T19:00:00+00:00",
            None,
            "run",
            "hosted",
        ),
        drep.IrisJob(
            "/benfeuer/atlas-difficulty-large-20261118-v20", "running", "2026-11-18T18:00:00+00:00", None, "run", "large"
        ),
        drep.IrisJob(
            "/benfeuer/atlas-difficulty-small-canary-20261117-v3",
            "running",
            "2026-11-17T10:00:00+00:00",
            None,
            "canary",
            "small",
        ),
        drep.IrisJob(
            "/benfeuer/atlas-difficulty-snowball-20261117-v9",
            "failed",
            "2026-11-17T09:00:00+00:00",
            None,
            "run",
            "snowball",
        ),
    ]
    assert drep.active_runs_by_solver(jobs) == {"hosted": 1, "small": 0, "large": 1, "snowball": 0}


def test_load_run_reads_a_local_tree_with_a_real_catalog(tmp_path):
    root = tmp_path / "2026.11.18.1" / "hosted-v13c"
    (root / "trials").mkdir(parents=True)
    (root / "serving.json").write_text(json.dumps({"solver": "hosted", "model": "m", "requested_revision": "r"}))
    (root / "trials" / "parent-campaign.json").write_text(
        json.dumps({"root": "s3://x/hosted-v12", "source_queue_sha256": "q" * 64, "serving_receipt": {}})
    )
    (root / "trials" / "model-requests-final.json").write_text(json.dumps({"input_tokens": 5, "output_tokens": 6}))
    (root / "completion.json").write_text(json.dumps({"worker_exit": 0}))
    with DataStore.open(str(root / "trials" / "catalog")) as store:
        register_difficulty(store)
        with store.transaction() as transaction:
            transaction.table(drep.MEASUREMENTS).add(_row("Src:a", "2026-11-18T01:00:00+00:00"))
    run = drep.load_run(str(root))
    assert run.solver == "hosted"
    assert run.parent == "s3://x/hosted-v12"
    assert run.completion["worker_exit"] == 0
    assert run.traffic_seal["input_tokens"] == 5
    assert [row["source_id"] for row in run.rows] == ["Src:a"]
    assert run.kind == "run"


def test_load_run_falls_back_to_the_label_solver_without_serving(tmp_path):
    root = tmp_path / "2026.11.18.1" / "small-v17a"
    root.mkdir(parents=True)
    (root / "prepared.json").write_text(json.dumps({"sources": []}))
    run = drep.load_run(str(root))
    assert run.solver == "small"
    assert run.kind == "run"


def test_seal_basis_names_missing_artifacts():
    run = _run("s3://x/2026.11.14.44/hosted-v12", [])
    assert "no completion.json" in drep.seal_basis(run)
    assert "traffic seal was never written" in drep.seal_basis(run)
    sealed = _run(
        "s3://x/2026.11.14.44/hosted-v12",
        [],
        completion={"worker_exit": 0},
        traffic_seal={"input_tokens": 1, "output_tokens": 2},
    )
    basis = drep.seal_basis(sealed)
    assert "completion.json (worker_exit 0)" in basis
    assert "final traffic sealed" in basis


def test_render_text_reports_the_missing_artifacts():
    run_section = {
        "root": "s3://x/2026.11.14.44/hosted-v12",
        "run_label": "hosted-v12",
        "kind": "run",
        "parent": None,
        "identity": {"solver": "hosted", "model": "m", "revision": "r"},
        "iris": {"job_id": "/benfeuer/j", "state": "failed", "submitted": "t", "active": False},
        "sealed": {
            "completion_present": False,
            "completion_worker_exit_zero": False,
            "final_traffic_sealed": True,
            "traffic_seal_reconstructed": True,
            "traffic_seal_tokens": {"input": 1},
            "basis": (
                "no completion.json — the run never sealed its worker exit (killed or still running); "
                "final traffic sealed (reconstructed from per-source receipts)"
            ),
        },
        "catalog": {
            "rows": 2,
            "sources_with_rows": 2,
            "added": 1,
            "aggregated_by_run": 1,
            "aggregated_by_run_statuses": {"measured": 1},
            "inherited": 1,
            "own_basis": "derived from cumulative row deltas",
            "pending_reasons": {"awaiting_scored_coverage": 1},
            "not_measured_reasons": {},
            "own_gpu_hours": 0.5,
            "own_input_tokens": 10,
            "own_output_tokens": 5,
            "first_row_created_at": "2026-11-15T14:41:00+00:00",
            "last_row_created_at": "2026-11-18T14:40:00+00:00",
            "selection": {"prepared_sources": None, "note": "selection size not read (pass --prepared)"},
        },
        "run_seal_present": False,
    }
    report = {
        "generated_at": "2026-11-18T16:00:00+00:00",
        "solvers": {"hosted": {"runs": [run_section], "catalog_totals_by_queue": {}, "identity": None}},
        "active_iris_jobs": [
            {
                "job_id": "/benfeuer/atlas-difficulty-hosted-20261118-v20",
                "state": "running",
                "submitted": "2026-11-18T15:00:00+00:00",
                "kind": "run",
                "solver": "hosted",
                "reason": None,
            }
        ],
        "active_runs_by_solver": {"hosted": 1, "small": 0, "large": 0, "snowball": 0},
        "active_jobs_without_run_root": ["/benfeuer/atlas-difficulty-hosted-20261118-v13"],
        "unattributed_runs": [],
    }
    text = drep.render_text(report)
    assert "no completion.json" in text
    assert "reconstructed" in text
    assert "derived from cumulative row deltas" in text
    assert "wall (first→last catalog row):" in text
    assert "added=1 (no ancestor in its chain measured these)" in text
    assert "atlas-difficulty-hosted-20261118-v13" in text
    assert "active runs by solver: hosted=1 small=0 large=0 snowball=0" in text
    assert "solver=hosted" in text


def test_render_source_section_lists_every_run_and_missing_sources():
    report = {
        "generated_at": "2026-11-18T16:00:00+00:00",
        "source_detail": {
            "Src:a": [
                {
                    "solver": "hosted",
                    "run_label": "hosted-v12",
                    "status": "measured",
                    "reason": None,
                    "n": 2,
                    "solved": 1,
                    "wilson_95": {"low": 0.1, "high": 0.9},
                    "gpu_hours": 0.5,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "created_at": "2026-11-18T01:00:00+00:00",
                }
            ]
        },
        "sources_without_rows": ["Src:missing"],
        "solvers": {},
        "active_iris_jobs": [],
        "active_jobs_without_run_root": [],
        "unattributed_runs": [],
    }
    text = drep.render_text(report)
    assert "== source Src:a ==" in text
    assert "hosted-v12" in text
    assert "wilson95=(0.100,0.900)" in text
    assert "== source Src:missing ==" in text
    assert "no measurement row" in text


def test_build_report_raises_when_a_queue_digest_differs(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps([]))
    monkeypatch.setattr(drep, "discover_run_roots", lambda base: [])
    monkeypatch.setattr(drep, "load_run", lambda root: pytest.fail("no runs expected"))
    with pytest.raises(drep.ReportError, match="differs from the run binding"):
        drep.load_queue_population(str(queue_path), "0" * 64)
