# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the difficulty rounds operating layer: launch command
composition and guards, seal reporting, queue digests and idempotency, and the
coverage observation collector."""

import json
from pathlib import Path

import pytest
from finestore.rl.difficulty import Solver
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation import difficulty_rounds as dr


def sample_plan(**overrides) -> dr.LaunchPlan:
    fields = dict(
        job_name="atlas-difficulty-small-20261116-v17a11",
        solver=Solver.SMALL,
        priority="interactive",
        sources=dr.CAMPAIGN_QUEUE_SOURCES,
        sources_sha256=dr.CAMPAIGN_QUEUE_SOURCES_SHA256,
        output_root="s3://marin-rno2a/marin/rl-data/atlas/difficulty/2026.11.17.2/small-v17a",
        local_root=Path("/tmp/atlas-difficulty-small-20261116-v17a11"),
        workers=8,
        attempts_per_task=3,
        daytona_provisioning="declarative",
        target_cluster=dr.TARGET_CLUSTER,
        source_ids=("TaskTrove:financebench",),
    )
    fields.update(overrides)
    return dr.LaunchPlan(**fields)


def test_launch_command_carries_every_explicit_parameter() -> None:
    command = dr.build_launch_command(sample_plan())
    joined = " ".join(command)
    assert "--target-cluster=cw-rno2a" in joined
    assert "--container-profile=container_profile_privileged" in joined
    assert "--egress-policy=egress_policy_cluster" in joined
    assert "--priority=interactive" in joined
    assert "-e DAYTONA_API_URL https://app.daytona.io/api" in joined
    assert "-e DAYTONA_TARGET us" in joined
    assert "--daytona-provisioning declarative" in joined
    assert "--sources-sha256 9d318f52e75dd0e76acaf450f3b824e236e9666b12458bd8dc517335fbfc5695" in joined
    assert "--source-id TaskTrove:financebench" in joined
    # The dry-run/plan argv carries placeholders, never a resolved key.
    assert "env:DAYTONA_EVAL_API_KEY" in joined
    assert "gcloud" not in joined


def test_launch_command_continuation_binds_previous_serving() -> None:
    command = dr.build_launch_command(
        sample_plan(
            previous_campaign="s3://marin-rno2a/marin/rl-data/atlas/difficulty/2026.11.17.1/small-v17a",
            previous_serving_sha256="abc123",
        )
    )
    joined = " ".join(command)
    assert "--previous-campaign s3://marin-rno2a/marin/rl-data/atlas/difficulty/2026.11.17.1/small-v17a" in joined
    assert "--previous-serving-sha256 abc123" in joined


def test_launch_command_requires_both_continuation_arguments() -> None:
    with pytest.raises(dr.RoundsError, match="both"):
        dr.build_launch_command(sample_plan(previous_campaign="s3://bucket/round"))
    with pytest.raises(dr.RoundsError, match="both"):
        dr.build_launch_command(sample_plan(previous_serving_sha256="abc"))


def test_resolved_daytona_key_never_reaches_the_recorded_command() -> None:
    plan = sample_plan()
    recorded = " ".join(dr.build_launch_command(plan))
    executed = " ".join(dr.build_launch_command(plan, daytona_key="dt_secret_value"))
    assert "dt_secret_value" not in recorded
    assert "-e DAYTONA_API_KEY dt_secret_value" in executed


def test_iris_job_exists_parses_list_rows(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        stdout = "JOB ID                     STATE     SUBMITTED\n/benfeuer/job-a  running  2026-11-17T10:00:00+00:00\n"
        return type("R", (), {"stdout": stdout})()

    monkeypatch.setattr(dr.subprocess, "run", fake_run)
    assert dr.iris_job_exists("marin", "job-a")
    assert not dr.iris_job_exists("marin", "job-b")

    # A child row does not make the root exist.
    def child_only_run(command, **kwargs):
        stdout = "/benfeuer/job-c/prepare-model-x  running  2026-11-17T10:00:00+00:00\n"
        return type("R", (), {"stdout": stdout})()

    monkeypatch.setattr(dr.subprocess, "run", child_only_run)
    assert not dr.iris_job_exists("marin", "job-c")


def test_seal_status_detects_completion_and_serving_digest(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "round"
    (output / "trials").mkdir(parents=True)
    (output / "completion.json").write_text(json.dumps({"worker_exit": 0}))
    (output / "serving.json").write_text('{"model": "small"}')
    (output / "trials" / "model-requests-final.json").write_text("{}")
    monkeypatch.setattr(dr, "_iris_root_job_state", lambda job_id: "succeeded")
    status = dr.seal_status(str(output), "/benfeuer/job-a")
    assert status["sealed"] and status["completion_exit_zero"]
    assert status["serving_json_sha256"] == file_sha256(output / "serving.json")

    (output / "completion.json").write_text(json.dumps({"worker_exit": 137}))
    monkeypatch.setattr(dr, "_iris_root_job_state", lambda job_id: "running")
    status = dr.seal_status(str(output), "/benfeuer/job-a")
    assert not status["sealed"]


def test_queue_digest_is_stable_and_order_insensitive() -> None:
    rows = [{"declaration": {"id": "a"}, "coverage": {"n": 1}}]
    rows_reversed = [{"coverage": {"n": 1}, "declaration": {"id": "a"}}]
    assert dr.queue_digest(rows) == dr.queue_digest(rows_reversed)
    assert dr.queue_digest(rows) != dr.queue_digest([{"declaration": {"id": "b"}}])


def test_plan_next_round_refuses_repeat_and_missing_serving(monkeypatch) -> None:
    document = dr.DifficultyCoverage(load_id=None, snapshot_id=None, generated_at="now", sources={}, counts={})
    declarations = {"a": {"id": "a"}}
    monkeypatch.setattr(dr, "frozen_source_ids", lambda: {"a"})
    sealed = {
        "output_root": "s3://marin-rno2a/marin/rl-data/atlas/difficulty/2026.11.17.1/small-v17a",
        "serving_json_sha256": "abc",
    }
    plan = dr.plan_next_round(sealed, document, declarations, ledger=[])
    assert plan["already_submitted"] is False
    assert "refusal" not in plan

    ledger = [{"queue_digest": plan["queue_digest"]}]
    plan = dr.plan_next_round(sealed, document, declarations, ledger=ledger)
    assert plan["already_submitted"] and "double-submit" in plan["refusal"]

    sealed_no_serving = {"output_root": "s3://x/round", "serving_json_sha256": None}
    plan = dr.plan_next_round(sealed_no_serving, document, declarations, ledger=[])
    assert "serving.json" in plan["refusal"]


def test_queue_source_ids_come_from_declarations() -> None:
    queue = [
        {"declaration": {"id": "TaskTrove:a"}, "coverage": {}},
        {"declaration": {"id": "TaskTrove:b"}, "coverage": {}},
    ]
    assert dr.queue_source_ids(queue) == ("TaskTrove:a", "TaskTrove:b")
    with pytest.raises(dr.RoundsError, match="no id"):
        dr.queue_source_ids([{"declaration": {"name": "x"}}])


def test_ledger_roundtrip(tmp_path: Path) -> None:
    ledger = tmp_path / "rounds.jsonl"
    assert dr.read_ledger(ledger) == []
    dr.append_ledger(ledger, {"queue_digest": "d1"})
    dr.append_ledger(ledger, {"queue_digest": "d2"})
    assert [row["queue_digest"] for row in dr.read_ledger(ledger)] == ["d1", "d2"]


def _observation_row(source_id: str, measurement_id: str, pinned: bool) -> dict:
    release = {"repo": "org/ds", "revision": "r1", "url": "https://example.com/ds"}
    row = {
        "measurement_id": measurement_id,
        "protocol_id": "atlas-difficulty-v3",
        "created_at": "2026-11-17T00:00:00+00:00",
        "source_id": source_id,
        "release": release,
        "converter_version": "c",
        "solver": "small",
        "status": "measured" if pinned else "not_measured",
        "reason": None if pinned else "test fixture: no qualified measurement for this source/model",
        "sample": None,
        "sample_receipt": None,
        "model": None,
        "settings": None,
        "n": 2 if pinned else 0,
        "solved": 1 if pinned else 0,
        "wilson_95": None,
        "infrastructure_attempts": 0,
        "pending_tasks": 0,
        "blocked_tasks": 0,
        "attempts_receipt": None,
        "gpu_hours": 0.5,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    if pinned:
        # Real stored rows carry the selected task list; the app's loader strips it
        # because the observation model forbids the extra key.
        row["sample"] = {
            "source_id": source_id,
            "release": release,
            "converter_version": "c",
            "converter_revision": "cr",
            "population_count": 2,
            "task_specs_sha256": "a" * 64,
            "drop_manifest": {"uri": "s3://bucket/drop.json", "sha256": "b" * 64},
            "seed": 7,
            "pre_oracle_filter": 0,
            "resample_after_oracle_filter": 0,
            "tasks": ["t1", "t2"],
        }
        row["model"] = {
            "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            "revision": "aca966e4",
            "tokenizer": "glm",
            "tokenizer_revision": "main",
            "chat_template_sha256": "c" * 64,
            "serving_receipt": {"uri": "s3://bucket/serving.json", "sha256": "d" * 64},
        }
        row["settings"] = {
            "context_tokens": 65536,
            "input_tokens": 49152,
            "output_tokens": 16384,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "thinking": "off",
            "deviations": [],
        }
        row["wilson_95"] = {"low": 0.1, "high": 0.9}
        row["sample_receipt"] = {"uri": "s3://bucket/sample.json", "sha256": "e" * 64}
        row["attempts_receipt"] = {"uri": "s3://bucket/attempts.tar.gz", "sha256": "f" * 64}
    return row


def test_collect_observations_groups_by_canonical_source(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    (root / "trials" / "trial-0").mkdir(parents=True)
    (root / "trials" / "trial-0" / "measurement.json").write_text(
        json.dumps(_observation_row("Source:one", "m1", pinned=True))
    )
    (root / "trials" / "trial-1").mkdir(parents=True)
    (root / "trials" / "trial-1" / "measurement.json").write_text(
        json.dumps(_observation_row("Source:one", "m2", pinned=False))
    )
    (root / "trials" / "catalog").mkdir()
    (root / "trials" / "judge").mkdir()

    observations = dr.collect_observations([str(root)])
    assert "Source:one" in observations
    assert {observation.measurement_id for observation in observations["Source:one"]} == {"m1", "m2"}

    # A row that cannot validate is counted as dropped, not fatal.
    bad = root / "trials" / "trial-2"
    bad.mkdir(parents=True)
    (bad / "measurement.json").write_text(json.dumps({"measurement_id": "m3"}))
    observations = dr.collect_observations([str(root)])
    assert "Source:one" in observations


def test_collect_observations_reads_qualified_measurements_json(tmp_path: Path) -> None:
    root = tmp_path / "qualified"
    root.mkdir()
    (root / "measurements.json").write_text(json.dumps([_observation_row("Source:two", "m9", pinned=True)]))
    observations = dr.collect_observations([str(root)])
    assert list(observations) == ["Source:two"]
    assert observations["Source:two"][0].n == 2

    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(dr.RoundsError, match=r"no measurements\.json"):
        dr.collect_observations([str(empty)])


def test_operator_stopped_round_seals_for_carry_over(monkeypatch, capsys, tmp_path):
    """next-round accepts an operator-stopped round (durable traffic seal) for carry-over."""
    declarations = {
        "Evalchemy:gsm8k": {
            "id": "Evalchemy:gsm8k",
            "status": "Available",
            "current_release": {"repo": "openai/gsm8k", "revision": "a" * 40},
            "pipeline": {"version": "v"},
            "actor_census_input": {"task_specs_sha256": "t" * 64, "measured_tasks": 1, "converter_revision": "c"},
        }
    }
    monkeypatch.setattr(dr, "fetch_declarations", lambda applet_id: declarations)
    monkeypatch.setattr(dr, "frozen_source_ids", lambda: {"Evalchemy:gsm8k"})
    fixture_row = json.loads(Path(__file__).with_name("observation_fixture_row.json").read_text())
    measurement = fixture_row
    measurements_dir = tmp_path / "measurements.json"
    measurements_dir.mkdir()
    (measurements_dir / "measurements.json").write_text(json.dumps([measurement]))
    root = Path("/tmp/rounds-operator-stopped-test/small-v17x")
    (root / "trials").mkdir(parents=True, exist_ok=True)
    (root / "serving.json").write_text(json.dumps({"solver": "small"}))
    (root / dr.SEAL_FINAL_TRAFFIC).write_text("{}")

    monkeypatch.setattr(dr, "_iris_root_job_state", lambda job_id: "killed")
    argv = [
        "next-round",
        "--watch-job-id",
        "/benfeuer/atlas-difficulty-small-20261116-v17x",
        "--watch-output-root",
        str(root),
        "--job-name",
        "atlas-difficulty-small-20261117-v18-test",
        "--output-root",
        "s3://marin-rno2a/marin/rl-data/atlas/difficulty/2026.11.17.9/small-v18-test",
        "--measurements",
        str(tmp_path / "measurements.json"),
        "--coverage-output-dir",
        str(tmp_path / "coverage"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
        "--operator-stopped",
        "--dry-run",
    ]
    monkeypatch.setattr("sys.argv", ["difficulty_rounds.py", *argv])
    dr.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["sealed"] is True
    assert payload["launch"]["dry_run"] is True
    assert payload["plan"]["source_ids"], "the queue carries carry-over sources"


def test_iris_job_exists_ignores_terminal_attempts(monkeypatch):
    """A failed prior attempt under the same name does not block a relaunch."""
    listing = (
        "JOB ID                                         STATE      SUBMITTED\n"
        "/benfeuer/atlas-difficulty-small-20261117-v18  failed     2026-11-17T18:25:54Z\n"
    )

    def fake_run(*args, **kwargs):
        class Completed:
            stdout = listing
            returncode = 0

        return Completed()

    monkeypatch.setattr(dr.subprocess, "run", fake_run)
    assert dr.iris_job_exists("marin", "atlas-difficulty-small-20261117-v18") is False

    live = listing.replace("failed", "running")
    monkeypatch.setattr(dr.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": live, "returncode": 0})())
    assert dr.iris_job_exists("marin", "atlas-difficulty-small-20261117-v18") is True
