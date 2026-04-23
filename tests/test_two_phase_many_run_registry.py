# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many.run_registry import build_run_registry as run_registry
from experiments.domain_phase_mix.exploratory.two_phase_many.run_registry.build_run_registry import (
    _qsplit300m_shard_index,
    _load_qsplit300m_recoveries,
    _summary,
)
from experiments.domain_phase_mix.launch_two_phase_many_qsplit240_300m_6b import build_run_specs
from experiments.domain_phase_mix.qsplit240_replay import select_run_specs_for_shard


def test_qsplit300m_shard_index_matches_contiguous_sharding() -> None:
    run_specs = build_run_specs(panel="all")
    shard_count = 8

    expected_by_run_name: dict[str, int] = {}
    for shard_index in range(shard_count):
        for spec in select_run_specs_for_shard(run_specs, shard_index=shard_index, shard_count=shard_count):
            expected_by_run_name[spec.run_name] = shard_index

    observed_by_run_name = {
        spec.run_name: _qsplit300m_shard_index(run_name=spec.run_name, shard_count=shard_count) for spec in run_specs
    }

    assert observed_by_run_name == expected_by_run_name


def test_summary_reports_only_missing_qsplit300m_runs() -> None:
    logical_runs = pd.DataFrame(
        [
            {
                "family": "qsplit240_300m_6b",
                "logical_status": "completed",
                "run_name": "run_00001",
                "active_recovery_job_id": None,
            },
            {
                "family": "qsplit240_300m_6b",
                "logical_status": "missing",
                "run_name": "run_00002",
                "active_recovery_job_id": None,
            },
            {
                "family": "qsplit240_300m_6b",
                "logical_status": "running",
                "run_name": "run_00003",
                "active_recovery_job_id": "/calvinxu/recovery-run-00003",
            },
            {
                "family": "regmix_raw_subset_optima",
                "logical_status": "completed",
                "run_name": "baseline_regmix",
                "active_recovery_job_id": None,
            },
        ]
    )
    run_attempts = pd.DataFrame([{"family": "qsplit240_300m_6b", "run_name": "run_00002"}])
    live_watchlist = pd.DataFrame([{"job_state": "RUNNING"}])

    summary = _summary(logical_runs, run_attempts, live_watchlist)

    assert summary["logical_run_count"] == 4
    assert summary["attempt_count"] == 1
    assert summary["live_watch_count"] == 1
    assert summary["qsplit240_300m_incomplete_runs"] == ["run_00002", "run_00003"]
    assert summary["qsplit240_300m_active_recovery_runs"] == ["run_00003"]
    assert summary["qsplit240_300m_unrecovered_missing_runs"] == ["run_00002"]


def test_load_qsplit300m_recoveries_reads_latest_monitor_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    older = tmp_path / "20260414-2147_monitoring_state.json"
    newer = tmp_path / "20260414-2200_monitoring_state.json"
    older.write_text(
        json.dumps(
            {
                "recoveries": [
                    {
                        "run_name": "run_00001",
                        "job_id": "/old",
                        "submission_workspace": "/tmp/old",
                        "resubmit_command": "old",
                        "latest_status": "running",
                        "latest_note": "old",
                    }
                ]
            }
        )
    )
    newer.write_text(
        json.dumps(
            {
                "recoveries": [
                    {
                        "run_name": "run_00119",
                        "job_id": "/calvinxu/recovery-00119",
                        "submission_workspace": "/tmp/recovery",
                        "resubmit_command": "resume-00119",
                        "latest_status": "running",
                        "latest_note": "latest",
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(run_registry, "SCRATCH_DIR", tmp_path)

    recoveries = _load_qsplit300m_recoveries()

    assert list(recoveries) == ["run_00119"]
    assert recoveries["run_00119"].job_id == "/calvinxu/recovery-00119"
