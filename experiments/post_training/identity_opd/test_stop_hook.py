# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import signal
from argparse import Namespace
from types import SimpleNamespace

from experiments.post_training.identity_opd.stop_hook import (
    GAP_METRIC,
    JOB_TERMINAL,
    KILLED,
    STEP_METRIC,
    parse_job_list,
    run_hook,
    step_rows,
    stop_reason,
)

JOB = "/karan/opd-dry-coordinator-0705e7a1"
JOB_LIST = f"""JOB ID                                   STATE    SUBMITTED
{JOB}/train-0                            killed   2026-10-26 07:12
{JOB}                                    failed   2026-10-26 07:11
"""


def _rows(advantages):
    return [{STEP_METRIC: step, GAP_METRIC: value, "policy/loss": 0.1} for step, value in enumerate(advantages, 1)]


def test_five_strict_rises_fire_and_four_do_not():
    # gap = -advantage, so a falling advantage is a rising gap.
    assert stop_reason(_rows([0.0, -0.1, -0.2, -0.3, -0.4])) is None
    assert "5 consecutive" in stop_reason(_rows([0.0, -0.1, -0.2, -0.3, -0.4, -0.5]))
    assert stop_reason(_rows([0.0, -0.1, -0.2, -0.2, -0.3, -0.4, -0.5])) is None
    assert stop_reason(_rows([0.5, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.4])) is None


def test_a_nonfinite_metric_fires_at_its_step():
    rows = _rows([0.0, 0.1, 0.2])
    rows[1]["policy/loss"] = math.nan
    assert stop_reason(rows) == "nonfinite policy/loss=nan at step 2"


def test_rows_without_the_gap_are_ignored_and_steps_are_ordered():
    history = [
        {STEP_METRIC: 2, GAP_METRIC: -0.2},
        {"_runtime": 5.0},
        {STEP_METRIC: 1, GAP_METRIC: -0.1},
        {STEP_METRIC: 3, "timing/step": 9.0},
    ]
    assert [row[STEP_METRIC] for row in step_rows(history)] == [1, 2]


def _args(tmp_path):
    return Namespace(
        project="entity/project",
        run_name_prefix="opd-dry",
        record=str(tmp_path / "record.json"),
        job=JOB,
        iris_config="config.yaml",
        last_step=2,
        interval=1,
    )


class Api:
    """W&B with no matching run yet, then (when given) a run that logs ``rows``."""

    def __init__(self, runs=(), rows=()):
        self._runs, self._rows = list(runs), list(rows)

    def runs(self, project, per_page):
        return self._runs

    def run(self, path):
        return SimpleNamespace(state="running", scan_history=lambda: self._rows)


def test_parse_job_list_reads_each_job_state():
    assert parse_job_list(JOB_LIST) == {JOB: "failed", f"{JOB}/train-0": "killed"}
    assert parse_job_list("No jobs found.\n") == {}


def test_a_job_that_fails_before_its_wandb_run_ends_the_hook_without_a_cancel(tmp_path, capsys, caplog):
    polls, cancels = [], []
    states = [{JOB: "running"}, parse_job_list(JOB_LIST)]

    def job_states(args):
        polls.append(args.job)
        return states[len(polls) - 1]

    record = run_hook(
        _args(tmp_path),
        api_factory=Api,
        job_states=job_states,
        cancel=cancels.append,
        sleep=lambda seconds: None,
    )
    assert len(polls) == 2 and cancels == []
    assert record["stop_reason"] == JOB_TERMINAL and not record["fired"] and "cancel" not in record
    assert record["iris_states"] == {JOB: "failed", f"{JOB}/train-0": "killed"}
    assert json.loads((tmp_path / "record.json").read_text()) == record
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == record
    assert caplog.messages == [f"OPD stop hook {JOB_TERMINAL}. Record: {tmp_path / 'record.json'}"]


def test_sigterm_while_waiting_writes_a_killed_record(tmp_path, capsys, caplog):
    def job_states(args):
        os.kill(os.getpid(), signal.SIGTERM)
        return {JOB: "running"}

    record = run_hook(
        _args(tmp_path),
        api_factory=Api,
        job_states=job_states,
        cancel=lambda args: None,
        sleep=lambda seconds: None,
    )
    assert record["stop_reason"] == KILLED and not record["fired"]
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["stop_reason"] == KILLED
    assert caplog.messages == [f"OPD stop hook {KILLED}. Record: {tmp_path / 'record.json'}"]


def test_a_found_run_that_fires_cancels_the_job(tmp_path):
    # The dry .4 trainer run (nvhmk151) carried job_type "worker_monitor".
    run = SimpleNamespace(id="abc", name="opd-dry-2026-356871096913", job_type="worker_monitor")
    rows = [{STEP_METRIC: step, GAP_METRIC: -0.1 * step} for step in range(1, 7)]
    cancels = []
    record = run_hook(
        _args(tmp_path),
        api_factory=lambda: Api([run], rows),
        job_states=lambda args: {JOB: "running"},
        cancel=lambda args: cancels.append(args.job) or {"job": args.job, "returncode": 0},
        sleep=lambda seconds: None,
    )
    assert record["fired"] and record["run_path"] == "entity/project/abc"
    assert cancels == [JOB] and record["cancel"]["returncode"] == 0
