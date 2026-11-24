# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Phase 2b pieces: pooled difficulty, the census manifest, the launcher."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.balanced_reasoning.catalog import (
    CensusEntry,
    build_tasks_manifest,
    manifest_rows,
    read_census_entries,
    validate_task_spec,
)
from experiments.post_training.task_curation.balanced_reasoning.difficulty import (
    SOLVER_ALL,
    LedgerAttempt,
    TaskDifficulty,
    tally_attempts,
)
from experiments.post_training.task_curation.balanced_reasoning.launch import (
    check_credentials,
    driver_command,
    parse_endpoint_address,
)
from experiments.post_training.task_curation.balanced_reasoning.test_lane import make_task

SHA_A = "aa" * 32
SHA_B = "bb" * 32


def ledger_attempt(task_id: str, sha: str, solver: str, reward: float | None, attempt: str) -> LedgerAttempt:
    return LedgerAttempt(
        source_id="org:source",
        task_sha256=sha,
        solver=solver,
        outcome="scored",
        reward=reward,
        attempt_id=attempt,
        task_id=task_id,
    )


def test_pooled_tally_folds_every_solver() -> None:
    attempts = [
        ledger_attempt("t1", SHA_A, "hosted", 0.0, "h1"),
        ledger_attempt("t1", SHA_A, "large", 1.0, "l1"),
        ledger_attempt("t1", SHA_A, "small", 1.0, "s1"),
        ledger_attempt("t2", SHA_B, "hosted", 1.0, "h2"),
    ]
    tasks = tally_attempts(attempts, solver=SOLVER_ALL)
    assert tasks[("org:source", SHA_A)].solve_rate == pytest.approx(2 / 3)
    assert tasks[("org:source", SHA_A)].basis == "ledger:pooled"
    assert tasks[("org:source", SHA_A)].task_id == "t1"
    # hosted-only keeps its own basis and rate
    hosted = tally_attempts(attempts, solver="hosted")
    assert hosted[("org:source", SHA_A)].solve_rate == 0.0
    assert hosted[("org:source", SHA_A)].basis == "ledger:hosted"
    assert hosted[("org:source", SHA_B)].solve_rate == 1.0


def test_validate_task_spec_strips_unknown_fields() -> None:
    task = make_task("What is 6 + 6? Answer with the number only.", "12", task_id="t-strip")
    document = json.loads(task.model_dump_json())
    document["integrity_contract"] = None
    parsed, stripped = validate_task_spec(json.dumps(document))
    assert stripped == ("integrity_contract",)
    assert parsed.id == "t-strip"
    exact, _ = validate_task_spec(task.model_dump_json())
    assert exact.id == "t-strip" and exact.grader.kind == "verifyit"


def _write_census_release(root: Path, release: str, rows: list[dict]) -> CensusEntry:
    """A one-source census release: censuses.json plus the normalized tasks parquet."""
    release_dir = root / release
    source_dir = release_dir / ("0" * 20)
    normalized = source_dir / "normalized" / "tasks.parquet"
    normalized.parent.mkdir(parents=True)
    table = pa.Table.from_pydict(
        {
            "id": [row["id"] for row in rows],
            "path": [row["id"] for row in rows],
            "task_spec": [row["task_spec"] for row in rows],
            "task_sha256": [row["task_sha256"] for row in rows],
            "runtime": ["{}" for _ in rows],
        }
    )
    pq.write_table(table, normalized)
    (release_dir / "censuses.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "org:source",
                    "normalized_path": str(normalized),
                    "population": {"dataset_id": "org/dataset-retained"},
                }
            ]
        )
    )
    return CensusEntry(
        source_id="org:source",
        release=release,
        normalized_path=str(normalized),
        dataset_id="org/dataset-retained",
    )


def test_build_tasks_manifest_joins_census_to_ledger(tmp_path: Path) -> None:
    solved = make_task("What is 2 + 2? Answer with the number only.", "4", task_id="task-solved")
    document = json.loads(solved.model_dump_json())
    document["integrity_contract"] = None
    census_sha = "cc" * 32
    entry = _write_census_release(
        tmp_path,
        "2026.11.13.1",
        [
            {"id": "task-solved", "task_spec": json.dumps(document), "task_sha256": census_sha},
            {"id": "task-unknown", "task_spec": json.dumps(document), "task_sha256": "dd" * 32},
        ],
    )
    difficulty = {
        ("org:source", SHA_A): TaskDifficulty(
            source_id="org:source",
            task_sha256=SHA_A,
            task_id="task-solved",
            solve_rate=1.0,
            scored=3,
            basis="ledger:pooled",
        )
    }
    receipt = build_tasks_manifest(difficulty, [entry], str(tmp_path / "out"))
    assert receipt["rows"] == 1
    assert receipt["skipped"] == {"no_ledger_task": 0, "no_grader": 0, "no_user_turn": 0, "spec_invalid": 0}
    rows = manifest_rows(str(tmp_path / "out" / "tasks.jsonl"))
    assert len(rows) == 1
    row = rows[0]
    assert row.task_id == "task-solved"
    assert row.task_sha256 == SHA_A, "the manifest must carry the ledger digest the plan bands on"
    assert row.census_task_sha256 == census_sha
    assert row.stripped_fields == ("integrity_contract",)
    assert row.domain == "source"
    assert Path(row.task_path).exists()
    reloaded = json.loads(Path(row.task_path).read_text())
    assert "integrity_contract" not in reloaded


def test_build_tasks_manifest_counts_ungraded_tasks(tmp_path: Path) -> None:
    task = make_task("What is 2 + 2? Answer with the number only.", "4", task_id="task-none")
    document = json.loads(task.model_dump_json())
    document["grader"] = {"kind": "none", "reason": "the source publishes no runnable grader"}
    entry = _write_census_release(
        tmp_path,
        "2026.11.13.1",
        [{"id": "task-none", "task_spec": json.dumps(document), "task_sha256": "ee" * 32}],
    )
    difficulty = {
        ("org:source", SHA_A): TaskDifficulty(
            source_id="org:source",
            task_sha256=SHA_A,
            task_id="task-none",
            solve_rate=0.5,
            scored=2,
            basis="ledger:pooled",
        )
    }
    receipt = build_tasks_manifest(difficulty, [entry], str(tmp_path / "out"))
    assert receipt["rows"] == 0
    assert receipt["skipped"]["no_grader"] == 1


def test_read_census_entries_prefers_first_release(tmp_path: Path) -> None:
    first = _write_census_release(tmp_path, "2026.11.13.1", [])
    second = _write_census_release(tmp_path, "2026.11.16.1", [])
    entries = read_census_entries(str(tmp_path))
    assert [entry.release for entry in entries if entry.source_id == "org:source"] == [first.release]
    assert first.normalized_path != second.normalized_path


def test_parse_endpoint_address() -> None:
    listing = "\n".join(
        [
            "I2026 log noise line",
            "NAME ACCESS PEER ADDRESS TASK",
            "/muchanem/glm53-relay/glm-5.3 private local http://10.0.0.1:8020 /muchanem/task/0",
        ]
    )
    assert parse_endpoint_address(listing, "/muchanem/glm53-relay/glm-5.3") == "http://10.0.0.1:8020"
    with pytest.raises(ValueError, match="no registered address"):
        parse_endpoint_address("NAME ACCESS PEER ADDRESS TASK\n", "/muchanem/glm53-relay/glm-5.3")


def _arguments(**overrides):
    from argparse import Namespace

    defaults = {
        "selection": "s3://bucket/selection.json",
        "run_root": "s3://bucket/run",
        "job_name": "balanced-reasoning-pilot",
        "concurrency": 16,
        "target_cluster": "cw-us-east-08a",
        "relay_cluster": "cw-us-east-02a",
        "relay_endpoint": "/muchanem/glm53-relay/glm-5.3",
        "cpu": 8,
        "memory": "16GB",
        "disk": "50GB",
        "max_retries": 3,
        "priority": "production",
        "pool": "bulk",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_driver_command_carries_credentials_by_reference() -> None:
    command = driver_command(_arguments(), "http://10.0.0.1:8020/v1")
    joined = " ".join(command)
    assert "--target-cluster cw-us-east-08a" in joined
    assert "--timeout 259200" in joined
    assert "--sync-package" not in joined, "the driver needs the full workspace (marin deps beyond marin-core)"
    assert "env:GLM_BULK_TOKEN" in joined, "the bulk token rides as a secret file, never as argv"
    assert "GLM_BULK_TOKEN=" not in joined and "Bearer" not in joined
    assert command[-1] == "--allow-live"
    assert command[command.index("--pool") :] == ["--pool", "bulk", "--allow-live"]
    assert "--selection s3://bucket/selection.json" in joined
    assert "--concurrency 16" in joined


def test_check_credentials_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CW_KEY_ID", "CW_KEY_SECRET", "GLM_BULK_TOKEN"):
        monkeypatch.setenv(name, "value")
    check_credentials()
    monkeypatch.setenv("GLM_BULK_TOKEN", "")
    with pytest.raises(SystemExit, match="GLM_BULK_TOKEN"):
        check_credentials()


def test_report_aggregates_bands_resolutions_and_transport(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from experiments.post_training.task_curation.balanced_reasoning.cli import build_parser

    task_dir = tmp_path / "tasks" / "t1"
    task_dir.mkdir(parents=True)
    (task_dir / "attempt.json").write_text(
        json.dumps(
            {
                "band": "easy",
                "resolution_kind": "solved_shortest",
                "selected_index": 0,
                "attempts": [
                    {
                        "kind": "solve",
                        "prompt_tokens": 100,
                        "completion_tokens": 900,
                        "reasoning_tokens": 700,
                    }
                ],
            }
        )
    )
    (task_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "sequence": 1,
                "wall_time": 1.0,
                "event": {
                    "type": "taskforge.effects.events.Resolved",
                    "payload": {
                        "request": {},
                        "fingerprints": [],
                        "value": {
                            "usage": {"prompt_tokens": 100, "completion_tokens": 900},
                            "decode_time": 30.0,
                            "wall_time": 45.0,
                            "retries": ["transport_error:None"],
                        },
                    },
                },
            }
        )
        + "\n"
    )
    parser = build_parser()
    args = parser.parse_args(["report", "--run-root", str(tmp_path)])
    assert args.handler(args) == 0
    report = json.loads(capsys.readouterr().out)
    band = report["bands"]["easy"]
    assert band["tasks"] == 1 and band["completion_tokens"] == 900
    assert band["decode_tok_per_s"] == 30.0
    assert band["solve_rate"] == 1.0
    assert report["resolutions"] == {"solved_shortest": 1}
    assert report["transport"]["retry_markers"] == {"transport_error": 1}
    assert report["transport"]["completions_with_retries"] == 1
    assert report["graceful"]["form_pass_rate"] is None
