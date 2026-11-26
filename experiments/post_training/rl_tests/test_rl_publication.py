# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, local
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import click
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finestore.rl.rl_progress import AttemptSummary, Verdict
from finestore.rl.run_records import (
    AttemptRecord,
    CheckpointReference,
    LaunchReceipt,
    RunIndexIdentity,
    read_attempts,
    read_evaluations,
    run_records_root,
    update_attempt,
    write_attempt,
    write_evaluations,
)
from finestore.store import DataStore
from fsspec.core import url_to_fs
from iris.resources.state import JobState, TaskState
from marin.execution.artifact import RECORD_FILENAME, Artifact
from marin.execution.lazy import ArtifactStep, run
from marin.rl.skyrl import ArtifactHfModel, SkyRLRun
from marin.rl.skyrl_records import read_stored_skyrl_attempts, recorded_launch
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import rl as runner
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_index
from experiments.post_training import rl_nightly_controls as nightly
from experiments.post_training import rl_presets as presets
from experiments.post_training import rl_publication as publication
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_index import APPLET_ID
from experiments.post_training.rl_index import list_attempts as read_progress_index
from experiments.post_training.rl_measurements import RUN_INPUTS_FILENAME, RunInputs, lineage_gpu_hours, run_inputs
from experiments.post_training.rl_tests.fixtures import (
    append_metrics,
    cohort_from_fixtures,
    copy_evaluations,
    edit_receipt,
    replace_evaluations,
    write_json,
    write_metrics,
    write_receipt,
    write_skyrl_run_record,
)
from experiments.post_training.rl_tests.fixtures import controls as write_controls
from experiments.post_training.skyrl_test_support import read_skyrl_metrics, selection_summary
from infra.marina.tests.rl_progress_fixtures import grader_report, progress_record

pytestmark = pytest.mark.usefixtures("source_provenance")


@pytest.mark.parametrize(
    "case, expected_updates, expected_time, expected_facts",
    [
        ("native-composed", 2, 3600, []),
        ("native-no-evaluation", 2, 3600, []),
        ("versioned-missing-evaluation", None, 3600, ["evaluation_records_available"]),
        ("checkpoint-lineage", 2, 7200, []),
        ("lineage-unknown-grandparent", None, None, []),
        ("lineage-unknown-allocation", 2, None, []),
        (
            "untracked-native",
            None,
            3600,
            ["eval_lite/grader_error_fraction", "eval_lite/initial/grader_error_fraction"],
        ),
        ("updates-override", 1, 3600, []),
        ("negative-duration", None, None, []),
    ],
)
@pytest.mark.timeout(180)
def test_native_attempt_records_keep_identity_measurements_and_failures(
    preset_table, recorded_runs, monkeypatch, case, expected_updates, expected_time, expected_facts
):
    def forbid_live(*args, **kwargs):
        pytest.fail("Host record builder requested live native execution/state")

    monkeypatch.setattr("marin.rl.skyrl.run_skyrl", forbid_live)
    monkeypatch.setattr("marin.rl.skyrl.read_skyrl_attempt", forbid_live)
    name = "train-long" if "updates-override" in case else "learning-medium"
    preset = (
        snowball_model.SNOWBALL.by_name[name]
        if name == "train-long"
        else replace(snowball_model.SNOWBALL.by_name[name], updates=2, measured=range(1, 3))
    )
    table = preset_table(preset)
    weights = Path(recipe.artifact_prefix(preset.recipe.layout.cluster)) / "weights" / "2026.11.10"
    write_skyrl_run_record(
        weights,
        "weights",
        "2026.11.10",
        fingerprint="fixture-initializer",
        hf_model_uri="s3://marin-test/models/weights/policy",
        global_step=1,
        tokenizer_uri=preset.model.tokenizer_uri,
        tokenizer_revision=preset.model.tokenizer_revision,
        iris_job_id="/tests/initializer",
    )
    native_table = table
    if case == "native-no-evaluation":
        without_evaluation = replace(preset, evaluations=(), checkpoints="none", checkpoint_every=None, export="none")
        native_table = replace(
            table, presets=tuple(without_evaluation if row.name == name else row for row in table.presets)
        )
    inputs = run_inputs(
        name,
        preset=native_table.by_name[name],
        settings=("trainer.policy.optimizer_config.lr=2e-06", "trainer.policy.optimizer_config.weight_decay=0.01")
        + ((f"trainer.max_steps={expected_updates}",) if name == "train-long" else ()),
        tasks=f"snowball-100k@{snowball_model.DEFAULT_SELECTION_VERSION}",
        init=recipe.stored_initializer(recipe.published_model("weights@2026.11.10", preset=preset), preset=preset),
    )
    source = recorded_runs(
        name,
        "tests/stored-progress",
        attempt_id="native-attempt",
        inputs=inputs,
        table=native_table,
    )
    with edit_receipt(source) as receipt:
        receipt["result"]["provenance"]["branch"] = "fixture/branch"
        if case in ("untracked-native", "versioned-missing-evaluation"):
            receipt["config"]["skyrl"]["trainer"]["logger"] = ["console", "files"]
    metrics = read_skyrl_metrics(str(source))
    records = read_attempts(str(source / "exports"))
    if case in {
        "checkpoint-lineage",
        "lineage-unknown-grandparent",
        "lineage-unknown-allocation",
        "negative-duration",
    }:
        ancestor = deepcopy(receipt)
        ancestor["config"]["run"]["attempt_id"] = "ancestor"
        if case == "lineage-unknown-allocation":
            ancestor["config"]["iris"]["allocation"].pop("gpus_per_node")
        ancestor["result"].update(
            state="failed",
            started_at="2026-11-06T16:40:00+00:00",
            finished_at="2026-11-06T15:40:00+00:00" if case == "negative-duration" else "2026-11-06T17:40:00+00:00",
        )
        native_ancestor = LaunchReceipt.model_validate(ancestor)
        records["ancestor"] = AttemptRecord(
            run_id=receipt["config"]["run"]["id"],
            attempt_id="ancestor",
            config=native_ancestor.config,
            result=native_ancestor.result,
            metrics_started_at=native_ancestor.result.started_at,
            elapsed_excluding_eval_seconds=123.0,
        )
        if case == "lineage-unknown-grandparent":
            records["ancestor"] = records["ancestor"].model_copy(
                update={
                    "resumed_step": 1,
                    "parent": CheckpointReference(
                        export_root=str(source / "unknown-exports"), attempt_id="unknown", step=1
                    ),
                }
            )
        records["native-attempt"] = records["native-attempt"].model_copy(
            update={
                "resumed_step": 1,
                "elapsed_excluding_eval_seconds": 321.0,
                "parent": CheckpointReference(export_root=str(source / "exports"), attempt_id="ancestor", step=1),
            }
        )
        metrics[1]["attempt_id"] = "ancestor"
    if case == "versioned-missing-evaluation":
        metrics = {}
        records["native-attempt"] = records["native-attempt"].model_copy(
            update={"evaluation_storage_version": 1, "metrics_started_at": None}
        )
    if case == "untracked-native":
        metrics = {}
        records["native-attempt"] = records["native-attempt"].model_copy(update={"metrics_started_at": None})
    with DataStore.open(run_records_root(str(source / "exports"))) as store:
        for record in records.values():
            update_attempt(store, record)
    write_metrics(source, metrics)
    if case == "versioned-missing-evaluation":
        replace_evaluations(source, [])
    if case == "negative-duration":
        receipt["result"]["finished_at"] = "2026-11-06T20:40:00+00:00"
        write_receipt(source, receipt)
        with pytest.raises(ValueError):
            publication.build_record(
                str(source),
                preset,
                table=table,
                attempt=read_stored_skyrl_attempts(str(source))[1]["native-attempt"],
            )
        return
    report = grader_report()
    report_root = source / "exports" / "grader_reports"
    write_json(report_root / "other-attempt" / "grader_report.json", {"evaluation": {}})
    if case == "native-composed":
        write_json(report_root / "native-attempt" / "grader_report.json", report)
    attempt = read_stored_skyrl_attempts(str(source))[1]["native-attempt"]
    attempt = attempt.model_copy(update={"index": None})
    row = publication.build_record(
        str(source),
        table.by_name["data-medium"],
        table=table,
        attempt=attempt,
    ).model_dump(mode="json")
    assert row["artifact_uri"] == str(source) and row["attempt_id"] == "native-attempt"
    assert row["iris_job_id"] == "/tests/native-run"
    assert row["run_id"] == receipt["config"]["run"]["id"] and row["series"] == name
    assert row["label"] == " ".join((name, *inputs.settings, f"tasks={inputs.tasks}", "init=weights@2026.11.10"))
    assert row["issue"] is None and row["gate_result"] is None
    assert row["branch"] == "fixture/branch"
    assert (row["optimizer_steps"], row["elapsed_seconds"]) == (expected_updates, expected_time)
    assert (row["planned_updates"], row["measured_updates"]) == ((1, [1]) if case == "updates-override" else (2, [1, 2]))
    assert row["timing_scope"] == "checkpoint_lineage_native_jobs"
    assert row["status"] == ("failed" if expected_facts else "succeeded"), (row["failed_facts"], row["evidence"])
    assert row["failed_facts"] == expected_facts, (row["failed_facts"], row["evidence"])
    if case == "native-no-evaluation":
        assert row["evaluations"] == []
    if case in {"checkpoint-lineage", "lineage-unknown-grandparent", "lineage-unknown-allocation"}:
        assert lineage_gpu_hours(attempt, recorded_launch(attempt)) == (80.0 if case == "checkpoint-lineage" else None)
    if case == "checkpoint-lineage":
        assert row["elapsed_excluding_eval_seconds"] == 444.0
    if case == "native-composed":
        assert row["engine"]["vllm_version"] == "fixture-vllm"
        assert row["grader_report"] == report
    else:
        assert row["grader_report"] is None


def test_stored_evaluation_keeps_its_original_selection_and_comparability(preset_table, recorded_runs):
    preset = replace(snowball_model.SNOWBALL.by_name["learning-medium"], updates=2, measured=range(1, 3))
    table = preset_table(preset)
    built = runner.build_experiment(preset, table=table)
    source = recorded_runs(preset.name, "tests/eval-coverage", attempt_id="recorded", inputs=built.inputs)
    attempt = read_stored_skyrl_attempts(str(source))[1]["recorded"]
    baseline = publication.build_record(str(source), preset, table=table, attempt=attempt)
    newer = replace(
        preset,
        held_out_tasks="eval-lite@2026.11.24.99",
        evaluations=(replace(preset.evaluations[0], name="renamed_eval", every=preset.checkpoint_every),),
    )
    write_json(Path(recipe.selection_artifact(newer.held_out_tasks).path()) / "summary.json", selection_summary(40))
    report = gate.GateReport(
        preset=preset.name,
        verdict=Verdict.INVALID,
        effect=None,
        reason="The evaluation protocol differs",
        run=gate.RunMetadata(preset_name=preset.name, replicates=1, attempt_ids=(attempt.attempt_id,)),
        control=gate.ControlMetadata(version="1234abcd", attempts=()),
        invariants=(),
        comparisons=(),
        gaps=(),
    )
    row = publication.build_record(str(source), newer, table=preset_table(newer), attempt=attempt, report=report)
    assert row.status == "succeeded" and row.failed_facts == [], (row.failed_facts, row.evidence)
    native = recorded_launch(attempt)
    (selection,) = native.validation_data
    expected_source = f"{selection.identity}/{selection.relative_path}"
    assert row.evaluation_data_identity == baseline.evaluation_data_identity == expected_source
    assert {item.evaluation_score.profile for item in row.evaluations if item.evaluation_score} == {"eval_lite"}
    assert row.gate_result is not None and row.gate_result.reason == report.reason
    assert {
        item.evaluation_score.step for item in row.evaluations if item.evaluation_score and item.evaluation_score.final
    } == {preset.updates}

    def family_identity(summary):
        digest = next(
            score.comparability
            for entry in summary.evaluations
            if (score := entry.evaluation_score) and score.final and score.statistic == "family_mean"
        )
        assert digest is not None
        return digest

    assert family_identity(row) == family_identity(baseline)
    outputs = read_evaluations(native.export_root, attempt_id=attempt.attempt_id, evaluation="eval/")
    index, item = next((i, item) for i, item in enumerate(outputs) if item.step == preset.updates)
    unanswered = item.answers[0].model_copy(update={"verification_status": "unavailable", "verifier_score": None})
    rewritten = item.model_copy(update={"answers": [unanswered, *item.answers[1:]]})
    replace_evaluations(source, [*outputs[:index], rewritten, *outputs[index + 1 :]])
    fewer_answers = publication.build_record(str(source), preset, table=table, attempt=attempt)
    assert family_identity(fewer_answers) != family_identity(baseline)

    changed = [
        (
            item.model_copy(update={"strata": item.strata.model_copy(update={"family": "alternate-family"})})
            if item.task_id == outputs[0].task_id
            else item
        )
        for item in outputs
    ]
    replace_evaluations(source, changed)
    differently_grouped = publication.build_record(str(source), preset, table=table, attempt=attempt)
    assert family_identity(differently_grouped) != family_identity(baseline)

    source_path = Path(selection.uri) / selection.relative_path
    samples = pq.read_table(source_path).to_pylist()
    task = json.loads(samples[0]["task_spec"])
    task.setdefault("labels", {})["comparison_fixture"] = "changed-content-with-same-id"
    samples[0]["task_spec"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(samples), source_path)
    changed_tasks = publication.build_record(str(source), preset, table=table, attempt=attempt)
    assert family_identity(changed_tasks) != family_identity(differently_grouped)

    altered_config = copy.deepcopy(attempt.native.config)
    altered_config["skyrl"]["generator"]["eval_sampling_params"]["temperature"] = 0.314
    altered = attempt.model_copy(update={"config": altered_config})
    changed_protocol = publication.build_record(str(source), preset, table=table, attempt=altered)
    assert family_identity(changed_protocol) != family_identity(changed_tasks)


def test_concurrent_issue_claims_keep_one_lineage_issue(tmp_path, monkeypatch):
    export_root = str(tmp_path / "exports")
    read_attempts = publication.read_attempts
    barrier = Barrier(2, timeout=10)
    thread = local()

    def read_with_race(path, *, view=None):
        attempts = read_attempts(path, view=view)
        if view is not None and not getattr(thread, "read", False):
            thread.read = True
            barrier.wait()
        return attempts

    monkeypatch.setattr(publication, "read_attempts", read_with_race)

    def claim(issue):
        try:
            return publication.assign_issue(export_root, "same-run", f"attempt-{issue}", issue)
        except publication.IssueConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, (73, 74)))
    [winner] = [claim for claim in claims if claim is not None]
    assert {attempt.issue for attempt in read_attempts(export_root).values()} == {winner}


def test_nightly_resumes_publication_and_refills_only_the_control_deficit(
    recorded_runs, monkeypatch, tmp_path, cache_native_compositions, capsys, subtests
):
    monkeypatch.setattr(nightly, "_OBSERVATION_SECONDS", 0)
    sources = ("accepted", "failed", "unprepared")
    preset = nightly.SNOWBALL.by_name["speed-medium"]
    control_base = "checkpoints/snowball-rl/controls/speed-medium-"
    version = gate.config_version(gate.compose_preset(preset))
    base = AttemptSummary.model_validate_json(
        json.dumps(
            progress_record()
            | {
                "model": "snowball",
                "series": preset.name,
                "preset_version": version,
                "is_control": True,
                "dirty": False,
                "branch": "main",
                "commit": "main-source",
                "base_commit": "main-source",
            }
        )
    )
    first = base.model_copy(update={"native_run_id": "control-0", "attempt_id": "first"})
    previous = [
        first,
        first.model_copy(update={"attempt_id": "retry"}),
        base.model_copy(update={"native_run_id": "control-1"}),
        base.model_copy(update={"native_run_id": "branch-control", "branch": "pull-branch"}),
        base.model_copy(update={"native_run_id": "old-control", "preset_version": "old"}),
        base.model_copy(update={"native_run_id": "failed-control", "status": "failed"}),
    ]
    rows, requests, states = {}, {}, {}
    child_status = None
    monkeypatch.setattr(rl_index, "list_attempts", lambda: [*previous, *rows.values()])

    def job_state(job):
        if str(job) not in states:
            raise ConnectError(Code.NOT_FOUND, f"Job {job} not found")
        return states[str(job)]

    def job_status(job):
        assert str(job) == child_job and child_status is not None
        return child_status

    client = SimpleNamespace(job_state=job_state, job_status=job_status)
    monkeypatch.setattr(nightly, "open_iris_client", lambda **_: nullcontext(client))
    unavailable = True

    def publish(record, _url):
        nonlocal unavailable
        if record.attempt_id == "accepted" and unavailable:
            unavailable = False
            raise RuntimeError("Marina temporarily unavailable")
        rows[record.attempt_id] = record
        return "https://progress.test"

    monkeypatch.setattr(publication, "publish_result", publish)
    state_dir = StoragePath(str(tmp_path))
    job_prefix = "/runner/checkpoints-snowball-rl-controls-speed-medium-"
    coordinator = job_prefix + "123-1-coordinator"
    child_job = f"{coordinator}/training"

    def submit(command, *, env):
        batch = command[command.index("--control") + 1]
        requested_preset = command[command.index("--preset") + 1]
        requests[requested_preset, batch] = int(command[command.index("--replicates") + 1])
        submit_job = f"{job_prefix}{batch}-coordinator"
        submission = publication.ControlSubmission(
            job=submit_job,
            outputs=tuple(str(tmp_path / (control_base + name) / recipe.RUN_ARTIFACT_VERSION) for name in sources),
            recorded_at=datetime.now(UTC),
        )
        StoragePath(env[publication.CONTROL_SUBMISSION_ENV]).write_text(submission.model_dump_json())
        assert "--wait" not in command
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(nightly, "subprocess", SimpleNamespace(Popen=submit))
    monkeypatch.setenv("IRIS_USER", "runner")
    legacy_job = job_prefix + "previous-coordinator"
    mochi_job = "/runner/checkpoints-mochi-rl-controls-learning-medium-previous-coordinator"
    states[mochi_job] = JobState.RUNNING
    rows["mochi"] = base.model_copy(
        update={
            "model": "mochi",
            "series": "learning-medium",
            "status": "running",
            "iris_job_id": mochi_job,
            "artifact_uri": str(tmp_path / "checkpoints/mochi-rl/controls/learning-medium-legacy/version"),
        }
    )
    argv = [
        "rl_nightly_controls",
        "--batch",
        "",
        "--presets",
        "snowball/speed-medium mochi/learning-medium",
        "--state-dir",
        str(state_dir),
    ]

    def tick(batch):
        argv[2] = batch
        monkeypatch.setattr("sys.argv", argv)
        return nightly.main()

    assert tick("123-1") == 0
    pending = state_dir / "snowball-speed-medium.json"
    first_handoff = publication.ControlSubmission.model_validate_json(pending.read_text())
    assert tick("124-1") == 0
    assert requests == {(preset.name, "123-1"): 3}
    assert publication.ControlSubmission.model_validate_json(pending.read_text()) == first_handoff
    grace_end = first_handoff.recorded_at + timedelta(minutes=15)
    assert (
        f"Pending record {pending}: Iris job {coordinator} not found yet. Grace ends at {grace_end.isoformat()}."
        in capsys.readouterr().out
    )
    states[legacy_job] = JobState.RUNNING
    legacy_path = recorded_runs(preset.name, control_base + "legacy", control=True, attempt_id="legacy")
    with edit_receipt(legacy_path) as receipt:
        receipt["config"]["run"]["coordinator_iris_job_id"] = legacy_job
        receipt["result"]["state"] = "succeeded"
    rows["legacy"] = base.model_copy(
        update={"status": "running", "iris_job_id": legacy_job, "artifact_uri": str(legacy_path)}
    )
    for attempt_id in sources[:2]:
        output = recorded_runs(preset.name, control_base + attempt_id, control=True, attempt_id=attempt_id)
        with edit_receipt(output) as receipt:
            receipt["config"]["run"]["coordinator_iris_job_id"] = coordinator
            if attempt_id == "failed":
                receipt["result"].update(state="failed", iris_job_id=child_job, iris_job_state=JobState.WORKER_FAILED)
    states[coordinator] = JobState.FAILED
    rows_before_publication = rows.copy()
    assert tick("125-1") == 1
    assert requests == {(preset.name, "123-1"): 3}
    assert "accepted" not in rows and rows["failed"].status == "failed"
    diagnostics = (
        (JobState.WORKER_FAILED, TaskState.PREEMPTED, "WorkloadEvictedDueToPreempted: priority", 0),
        (JobState.WORKER_FAILED, TaskState.WORKER_FAILED, "PodDeleted: pod was deleted while active", 0),
        (JobState.WORKER_FAILED, TaskState.WORKER_FAILED, "Evicted", 0),
        (JobState.FAILED, TaskState.FAILED, "Cumulative failed task attempts exceeded max_task_failures", 1),
    )
    _, failed_output, _ = first_handoff.outputs
    # Replay the same cohort to check each terminal diagnostic with the same native evidence.
    for state, task_state, reason, reconcile_exit in diagnostics:
        with subtests.test(reason=reason):
            rows.clear()
            rows.update(rows_before_publication)
            child_status = SimpleNamespace(state=state, task_state_counts={task_state: 1}, error_message=reason)
            pending.write_text(first_handoff.model_dump_json())
            with edit_receipt(Path(failed_output)) as receipt:
                receipt["result"]["iris_job_state"] = state
            assert tick("126-1") == reconcile_exit
            assert not pending.exists()
            assert requests == {(preset.name, "123-1"): 3}
            assert rows["accepted"].status == "succeeded" and rows["accepted"].preset_version == version
            assert rows["failed"].status == "failed" and "native_attempt" in rows["failed"].failed_facts
    states[legacy_job] = JobState.SUCCEEDED
    assert tick("127-1") == 0
    assert requests == {(preset.name, "123-1"): 3, (preset.name, "127-1"): 1}
    assert rows["legacy"].status == "succeeded"


def test_nightly_recovers_pending_controls(
    recorded_runs, monkeypatch, tmp_path, cache_native_compositions, capsys, subtests
):
    scenarios = (
        ("failed-before-preparation", 1),
        ("missing-handoff", 1),
        ("expired-grace", 0),
        ("legacy-nonterminal", 1),
        ("durable-nonterminal", 1),
        ("failed-coordinator-nonterminal", 1),
        ("evicted-coordinator-nonterminal", 0),
        ("evicted-coordinator-failed-child-nonterminal", 1),
    )
    for scenario, exit_code in scenarios:
        with subtests.test(scenario=scenario), monkeypatch.context() as case_env:
            _assert_nightly_recovery(recorded_runs, case_env, tmp_path, capsys, scenario, exit_code)


def _assert_nightly_recovery(recorded_runs, monkeypatch, tmp_path, capsys, scenario, exit_code):
    capsys.readouterr()
    monkeypatch.setattr(nightly, "_OBSERVATION_SECONDS", 0)
    monkeypatch.setenv("IRIS_USER", "runner")
    preset = nightly.SNOWBALL.by_name["speed-medium"]
    (tmp_path / scenario).mkdir()
    state_dir = StoragePath(str(tmp_path / scenario))
    pending = state_dir / "snowball-speed-medium.json"
    job_prefix = "/runner/checkpoints-snowball-rl-controls-speed-medium-"
    previous = job_prefix + "previous-coordinator"
    replacement = job_prefix + "new-coordinator"
    output_name = f"checkpoints/snowball-rl/controls/speed-medium-{scenario}"
    output = tmp_path / output_name / recipe.RUN_ARTIFACT_VERSION
    rows, launches, states = {}, {}, {}

    def publish(record, _url):
        rows[record.attempt_id] = record
        return "https://progress.test"

    def job_state(job):
        if str(job) not in states:
            raise ConnectError(Code.NOT_FOUND, f"Job {job} not found")
        return states[str(job)]

    def job_status(job):
        return SimpleNamespace(state=states[str(job)], task_state_counts={}, error_message="Evicted")

    def submit(command, *, env):
        key = (command[command.index("--preset") + 1], command[command.index("--control") + 1])
        launches[key] = int(command[command.index("--replicates") + 1])
        if scenario != "missing-handoff":
            handoff = publication.ControlSubmission(
                job=replacement, outputs=(str(output),), recorded_at=datetime.now(UTC)
            )
            StoragePath(env[publication.CONTROL_SUBMISSION_ENV]).write_text(handoff.model_dump_json())
        return SimpleNamespace(poll=lambda: 0)

    monkeypatch.setattr(rl_index, "list_attempts", lambda: list(rows.values()))
    monkeypatch.setattr(publication, "publish_result", publish)
    client = SimpleNamespace(job_state=job_state, job_status=job_status)
    monkeypatch.setattr(nightly, "open_iris_client", lambda **_: nullcontext(client))
    monkeypatch.setattr(nightly, "subprocess", SimpleNamespace(Popen=submit))
    if scenario.endswith("nonterminal"):
        assert output == recorded_runs(preset.name, output_name, control=True, attempt_id="partial")
        with edit_receipt(output) as receipt:
            receipt["config"]["run"]["coordinator_iris_job_id"] = previous
            receipt["result"].update(state="submitted", iris_job_id=f"{previous}/training", finished_at=None)
        states[previous] = {
            "failed-coordinator-nonterminal": JobState.FAILED,
            "evicted-coordinator-nonterminal": JobState.WORKER_FAILED,
            "evicted-coordinator-failed-child-nonterminal": JobState.WORKER_FAILED,
        }.get(scenario, JobState.SUCCEEDED)
        states[f"{previous}/training"] = {
            "failed-coordinator-nonterminal": JobState.KILLED,
            "evicted-coordinator-failed-child-nonterminal": JobState.FAILED,
        }.get(scenario, JobState.SUCCEEDED)
    if scenario == "failed-before-preparation":
        states[previous] = JobState.FAILED
    if scenario == "legacy-nonterminal":
        assert publication.publish_running(str(output), preset, nightly.SNOWBALL, previous, None, set())
        assert rows["partial"].status == "running"
    elif scenario != "missing-handoff":
        handoff = publication.ControlSubmission(
            job=previous, outputs=(str(output),), recorded_at=datetime.now(UTC) - timedelta(minutes=16)
        )
        pending.write_text(handoff.model_dump_json())

    def tick(batch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "rl_nightly_controls",
                "--batch",
                batch,
                "--presets",
                "snowball/speed-medium",
                "--state-dir",
                str(state_dir),
            ],
        )
        return nightly.main()

    assert tick("new") == exit_code
    captured = capsys.readouterr()
    if scenario == "durable-nonterminal":
        assert not launches and not rows
        assert publication.ControlSubmission.model_validate_json(pending.read_text()).job == previous
        assert "Publication requires a terminal native attempt" in captured.out
        assert "Terminal publication remains pending" in captured.err
        return
    assert launches == {(preset.name, "new"): gate.CONTROL_COUNT}
    if scenario == "missing-handoff":
        assert not pending.exists()
        assert "launcher exited without a pending handoff" in captured.err
        return
    assert publication.ControlSubmission.model_validate_json(pending.read_text()).job == replacement
    assert "launcher exited without a pending handoff" not in captured.err
    if scenario == "failed-before-preparation":
        assert "Coordinator ended before preparing a native attempt" in captured.out
    if scenario == "legacy-nonterminal":
        assert rows["partial"].status == "failed"
        assert rows["partial"].failed_facts == ["missing_terminal_receipt"]
        assert tick("later") == 0
        assert "missing_terminal_receipt" not in capsys.readouterr().out
    if scenario.endswith("nonterminal") and scenario != "legacy-nonterminal":
        assert rows["partial"].status == "failed"
        assert rows["partial"].failed_facts == ["coordinator_interrupted"]


def test_control_presets_select_multiple_models_without_duplicates():
    selected = nightly.select_presets("snowball/learning-medium mochi mochi/learning-medium snowball/speed-medium")
    assert sorted((model, preset.name) for model, preset in selected) == [
        ("mochi", "learning-medium"),
        ("mochi", "speed-medium"),
        ("snowball", "learning-medium"),
        ("snowball", "speed-medium"),
    ]


@pytest.mark.parametrize(
    "merged, permission, selection, error",
    [
        (False, "admin", "mochi", "merged pull request"),
        (True, "read", "mochi", "write permission"),
        (True, "write", "mochi snowball/data-medium", "snowball/data-medium"),
        (True, "write", "mochi <unknown>", "&lt;unknown&gt;"),
        (True, "write", "", "at least one model"),
        (True, "write", None, None),
        (True, "write", "mochi snowball/learning-medium", None),
        (True, "admin", "snowball/speed-medium mochi/learning-medium", None),
    ],
)
def test_control_comment_checks_repository_access_and_dispatches_main(
    monkeypatch, tmp_path, merged, permission, selection, error
):
    event = tmp_path / "comment.json"
    body = "can we run /rl-controls mochi?" if selection is None else f"/rl-controls {selection}".strip()
    event.write_text(
        json.dumps(
            {
                "issue": {"number": 14, "pull_request": {}},
                "comment": {"body": body, "user": {"login": "contributor"}},
            }
        )
    )
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/marin")
    calls = []
    pull = "repos/example/marin/pulls/14"
    access = "repos/example/marin/collaborators/contributor/permission"
    comments = "repos/example/marin/issues/14/comments"

    def github(command, **options):
        calls.append((command, options.get("input")))
        if pull in command:
            return json.dumps({"merged": merged}).encode()
        if access in command:
            return json.dumps({"permission": permission}).encode()
        return b'{"html_url":"https://github.com/example/marin/actions/runs/145"}'

    monkeypatch.setattr(nightly.subprocess, "check_output", github)
    nightly.dispatch_comment(event)
    if selection is None:
        assert calls == []
    elif error:
        *checks, (reply, _) = calls
        assert [command[-1] for command, _ in checks] == ([pull, access] if merged else [pull])
        assert comments in reply
        assert reply[-1].startswith("body=🤖 <code>") and error in reply[-1]
        assert len(reply[-1].splitlines()) == 1
    else:
        (pull_request, _), (permissions, _), (dispatch, payload), (reply, _) = calls
        assert pull_request[-1] == pull and permissions[-1] == access
        assert "X-GitHub-Api-Version: 2026-03-10" in dispatch
        assert "repos/example/marin/actions/workflows/marin-skyrl-nightly.yaml/dispatches" in dispatch
        assert json.loads(payload) == {"ref": "main", "inputs": {"controls": "true", "presets": selection}}
        assert comments in reply
        assert reply[-1] == "body=🤖 [RL control run](https://github.com/example/marin/actions/runs/145)"


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_snowball_publication_updates_attempts_and_publishes_controls(
    preset_table, rl_progress_client, recorded_runs, monkeypatch, cache_native_compositions
):
    client, request = rl_progress_client
    preset = replace(
        snowball_model.SNOWBALL.by_name["learning-medium"],
        guards=(*snowball_model.SNOWBALL.by_name["learning-medium"].guards, presets.Metric("policy/entropy", True)),
    )
    table = preset_table(preset)
    monkeypatch.setattr(publication, "configure_coreweave_s3", lambda: None)
    entropies = iter(0.48 + index * 0.01 for index in range(gate.CONTROL_COUNT))
    controls = write_controls(
        lambda *args, **kwargs: recorded_runs(*args, metrics={"policy/entropy": next(entropies)}, **kwargs),
        preset.name,
        cohort_run_id="nightly-batch",
    )
    sources = [
        recorded_runs(
            preset.name, f"tests/publication-{i}", metrics={"policy/entropy": float(i)}, attempt_id=f"native-{i}"
        )
        for i in range(2)
    ]
    handles = tuple(
        ArtifactStep.adopt(f"tests/publication-r{i}", recipe.RUN_ARTIFACT_VERSION, str(root), kind=SkyRLRun)
        for i, root in enumerate(sources)
    )

    def stored():
        return {row["attempt_id"]: row for row in client.get(f"/a/{APPLET_ID}/api/runs").json()["runs"]}

    def publish(gate_path):
        return publication.publish_runs(
            [str(root) for root in sources],
            preset,
            table=table,
            gate_path=gate_path,
            service_url="http://localhost",
        )

    cohort = cohort_from_fixtures(sources[1])

    def saved_gate(requirements=()):
        report = run(
            gate.gate_step(
                preset,
                handles,
                cohort=cohort,
                group_name="tests/publication",
                inputs=RunInputs(preset_name=preset.name, requirements=requirements),
            )
        )[0]
        return report, str(report.path)

    report, first_gate = saved_gate()
    interrupted = {"once": True}
    logical_id = "snowball-learning-medium-tester-1234abcd"

    def unavailable_readback(origin, method, path, *, json_body=None, **kwargs):
        response = request(origin, method, path, json_body=json_body)
        if method == "POST" and path.endswith("/api/runs") and interrupted["once"]:
            interrupted["once"] = False
            raise RuntimeError("Response unavailable after insert")
        return response

    monkeypatch.setattr("marina.client.marina_request", unavailable_readback)
    for index, source in enumerate(sources):
        attempt_id = f"native-{index}"
        attempt = read_stored_skyrl_attempts(str(source))[1][attempt_id]
        write_attempt(
            str(source / "exports"),
            AttemptRecord(
                run_id=attempt.run_id,
                attempt_id=attempt_id,
                issue=73 if index == 0 else None,
                index=(
                    RunIndexIdentity(
                        run_id=logical_id,
                        model=table.model,
                        preset=preset.name,
                        preset_version="1234abcd",
                        user="tester",
                        branch="fixture",
                        source_revision="fixture",
                        base_preset_version="1234abcd",
                        dirty=False,
                    )
                    if index == 0
                    else None
                ),
            ),
        )
    input_record = sources[0] / RUN_INPUTS_FILENAME
    declared_inputs = RunInputs.model_validate_json(input_record.read_text())
    sentinel = "fixture-private-setting-value"
    input_record.write_text(
        declared_inputs.model_copy(update={"settings": (f'trainer.api_key="{sentinel}"',)}).model_dump_json()
    )
    first = publish(first_gate)
    assert len(first.served_records) == 1 and first.errors == ["Response unavailable after insert"]
    persisted = stored()
    assert sentinel not in persisted["native-0"]["label"]
    assert "trainer.api_key=[REDACTED]" in persisted["native-0"]["label"]
    input_record.write_text(declared_inputs.model_dump_json())
    assert set(persisted) == {"native-0", "native-1"} and persisted["native-0"]["issue"] == 73
    # A numeric bound is stored only as an invariant, with distinct replicate outcomes.
    relative_key = "eval_lite/aai_category/coding/mean"
    report, second_gate = saved_gate(("policy/entropy<=0", "policy/entropy>=0", f"{relative_key}>=control"))
    assert report.verdict == "fail", report.reason
    with edit_receipt(sources[0]) as receipt:
        receipt["result"]["finished_at"] = (
            datetime.fromisoformat(receipt["result"]["started_at"]) + timedelta(hours=2)
        ).isoformat()

    repeated = publish(second_gate)
    assert repeated.unserved_records == [] and repeated.errors == []
    payload = client.get(f"/a/{APPLET_ID}/api/runs").json()
    assert payload["skipped_records"] == [] and len(payload["runs"]) == 2
    rows = {row["attempt_id"]: row for row in payload["runs"]}
    assert rows["native-0"]["evaluation_data_identity"].endswith("/full_validation.parquet")
    assert repeated.urls[str(sources[0])].endswith(f"#run={logical_id}&attempt=native-0")

    def wrong_returned_key(origin, method, path, *, json_body=None, **kwargs):
        response = request(origin, method, path, json_body=json_body)
        if method == "POST" and path.endswith("/api/runs"):
            return response | {"attempt_id": "wrong-returned-attempt"}
        return response

    monkeypatch.setattr("marina.client.marina_request", wrong_returned_key)
    returned_row = {key: value for key, value in rows["native-0"].items() if key != "has_grader_report"}
    with pytest.raises(RuntimeError, match="different returned identity"):
        publication.publish_result(
            AttemptSummary.model_validate_json(json.dumps(returned_row), strict=True), "http://localhost"
        )
    monkeypatch.setattr("marina.client.marina_request", request)
    expected_metrics = {
        "eval_lite/family_mean": "objective",
        "eval_lite/family_mean/gain_per_gpu_hour": "objective",
        "policy/entropy": "guard",
        relative_key: "requirement",
        "eval_lite/mean": "slice",
        "eval_lite/family/small/mean": "slice",
        "eval_lite/family/large/mean": "slice",
    }
    expected = report.model_dump(mode="json", include={"verdict", "effect", "reason", "invariants"}) | {
        "comparisons": [
            row.model_dump(mode="json") for row in report.comparisons if expected_metrics.get(row.metric) == row.role
        ],
        "attempt_ids": ["native-0", "native-1"],
        "replicates": 2,
    }
    assert rows["native-0"]["gate_result"] == rows["native-1"]["gate_result"] == expected
    assert {item["metric"]: item["role"] for item in rows["native-0"]["gate_result"]["comparisons"]} == expected_metrics

    for index in range(2):

        @click.command()
        def republish_current(index=index):
            raise click.exceptions.Exit(
                runner.main(
                    table,
                    argv=[
                        "publish",
                        "--preset",
                        preset.name,
                        "--marina-url",
                        "http://localhost",
                        "--run-output",
                        str(sources[index]),
                        "--attempt",
                        f"native-{index}",
                        "--issue",
                        str(73 + index),
                    ],
                    init=ArtifactHfModel(
                        ArtifactStep.adopt("tests/unbuilt-init", "2026.11.17", "s3://unused/init", kind=Artifact)
                    ),
                )
            )

        republished = CliRunner().invoke(republish_current)
        assert republished.exit_code == 0, (republished.output, republished.exception)
        retained = stored()[f"native-{index}"]
        assert retained["gate_result"] == expected and retained["gate"] == rows[f"native-{index}"]["gate"]
        assert retained["issue"] == 73 + index
        assert publication.lineage_issue(str(sources[index] / "exports")) == 73 + index
        assert retained["wandb_url"] == rows[f"native-{index}"]["wandb_url"]
        assert (retained["status"], retained["failed_facts"]) == (
            rows[f"native-{index}"]["status"],
            rows[f"native-{index}"]["failed_facts"],
        )
    assert rows["native-0"]["status"] == "succeeded" and rows["native-0"]["failed_facts"] == []
    assert rows["native-1"]["status"] == "failed" and rows["native-1"]["failed_facts"] == ["require/policy/entropy"]
    assert rows["native-0"]["elapsed_seconds"] == 7200 and rows["native-0"]["issue"] == 73
    conflict = CliRunner().invoke(
        publication.command(table),
        [
            "--preset",
            preset.name,
            "--marina-url",
            "http://localhost",
            "--run-output",
            str(sources[0]),
            "--attempt",
            "native-0",
            "--issue",
            "74",
        ],
    )
    assert conflict.exit_code == 1 and "conflicting lineage issues" in conflict.output
    assert publication.lineage_issue(str(sources[0] / "exports")) == 73
    grafana = urlparse(rows["native-0"]["grafana_url"])
    assert grafana.path == "/d/marin-rl-runs/rl-post-training"
    query = parse_qs(grafana.query)
    assert query["var-run"] == [logical_id]
    assert query["var-cluster"] == [receipt["config"]["iris"]["cluster"]]
    assert query["to"] == ["1793997600000"]
    evaluation = {item["metric"]: item for item in rows["native-0"]["evaluations"]}
    assert evaluation["eval_lite@25/family/small/mean"]["run"] == 0.5
    assert evaluation["eval_lite@25/aai_category/format/mean"]["run"] == 0.5
    assert evaluation["eval_lite@25/difficulty/untagged/mean"]["run"] == 0.5
    assert evaluation["eval_lite@25/attempted"]["run"] == evaluation["eval_lite@25/scored"]["run"] == 20
    assert repeated.urls[str(sources[0])].endswith("&attempt=native-0")
    windowless = deepcopy(receipt)
    windowless["result"]["finished_at"] = None
    write_receipt(sources[0], windowless)
    assert publish(second_gate).unserved_records == []
    assert stored()["native-0"]["elapsed_seconds"] is None
    assert any("Checkpoint lineage timing is unknown" in note for note in stored()["native-0"]["evidence"])
    write_receipt(sources[0], receipt)
    # A changed replicate invalidates the entire aggregate, including the unchanged replicate.
    retry_metrics = read_skyrl_metrics(str(sources[0]))
    receipt["config"]["run"]["attempt_id"] = "native-retry"
    write_receipt(sources[0], receipt)
    copy_evaluations(sources[0], "native-0", "native-retry")
    for values in retry_metrics.values():
        values["attempt_id"] = "native-retry"
    append_metrics(sources[0], retry_metrics)
    stale = publish(second_gate)
    assert len(stale.unserved_records) == 2 and all("current output group" in error for error in stale.errors)
    historical = stored()["native-0"]
    assert historical["gate_result"] == expected and historical["optimizer_steps"] == preset.updates
    fresh_report, fresh_gate = saved_gate()
    fresh = publish(fresh_gate)
    assert fresh.unserved_records == [] and fresh.errors == []
    assert fresh_report.run.attempt_ids == ("native-retry", "native-1")
    historical = stored()["native-0"]
    assert historical["issue"] == 73
    saved_evaluations = {}
    for root in sources:
        current_id, _ = read_stored_skyrl_attempts(str(root))
        export_root = str(root / "exports")
        rows = read_evaluations(export_root, attempt_id=current_id, evaluation="eval/")
        saved_evaluations[export_root] = rows
        write_evaluations(
            export_root,
            [
                (
                    row.model_copy(
                        update={
                            "answers": [
                                answer.model_copy(update={"verifier_score": 0.0, "passed": False})
                                for answer in row.answers
                            ]
                        }
                    )
                    if row.step == preset.updates
                    else row
                )
                for row in rows
            ],
        )
    _, guard_gate = saved_gate(("policy/entropy>=0",))
    guard_result = publish(guard_gate)
    assert guard_result.unserved_records == [] and guard_result.errors == []
    guard_rows = [
        row
        for row in client.get(f"/a/{APPLET_ID}/api/runs").json()["runs"]
        if row["attempt_id"] in {"native-retry", "native-1"}
    ]
    assert len(guard_rows) == 2
    assert all(
        row["gate_result"]["verdict"] == "fail" and row["status"] == "succeeded" and row["failed_facts"] == []
        for row in guard_rows
    )
    for export_root, rows in saved_evaluations.items():
        write_evaluations(export_root, rows)
    second_receipt = json.loads((sources[1] / "terminal.json").read_text())
    failed_second = deepcopy(second_receipt)
    failed_second["result"]["state"] = "failed"
    write_receipt(sources[1], failed_second)
    (sources[1] / "terminal.json").unlink()
    filesystem = type(url_to_fs(fresh_gate, use_listings_cache=False)[0])
    stored_exists = filesystem.exists

    def gate_marker_unavailable(self, path, **kwargs):
        if str(path) == str(StoragePath(fresh_gate) / RECORD_FILENAME):
            raise OSError("Gate storage unavailable")
        return stored_exists(self, path, **kwargs)

    with monkeypatch.context() as storage_fault:
        storage_fault.setattr(filesystem, "exists", gate_marker_unavailable)
        blocked = publish(fresh_gate)
    assert blocked.unserved_records == [(receipt["config"]["run"]["id"], "native-retry")]
    assert blocked.errors == ["Gate storage unavailable"]
    assert set(blocked.served_records) == {
        (logical_id, "native-0"),
        (second_receipt["config"]["run"]["id"], "native-1"),
    }
    failed_row = stored()["native-1"]
    assert failed_row["gate_result"]["verdict"] == "fail" and "native_attempt" in failed_row["failed_facts"]
    write_receipt(sources[1], second_receipt)
    # Missing provenance produces an attributable INVALID through the actual gate.
    receipt["result"]["provenance"] = None
    write_receipt(sources[0], receipt)
    invalid_report, invalid_gate = saved_gate(("policy/entropy<=2",))
    assert invalid_report.verdict == "invalid" and invalid_report.run.attempt_ids == ("native-retry", "native-1")
    invalid = publish(invalid_gate)
    assert invalid.unserved_records == []
    invalid_row = stored()["native-retry"]
    assert invalid_row["gate_result"]["verdict"] == "invalid" and invalid_row["gate_result"]["reason"]
    (sources[1] / "terminal.json").unlink()
    (sources[1] / "resolved-launch.yaml").unlink()
    missing_report, missing_gate = saved_gate()
    assert missing_report.verdict == "invalid" and missing_report.run.attempt_ids == ("native-retry",)
    assert missing_report.run.replicates == 2
    missing = publish(missing_gate)
    assert str(sources[1]) in missing.unserved_records and any(
        "No stored native receipt" in error for error in missing.errors
    )
    latest_peer_gate = stored()["native-1"]["gate_result"]
    write_receipt(sources[1], second_receipt)
    (sources[1] / "terminal.json").unlink()
    missing_current = deepcopy(second_receipt["config"])
    missing_current["run"]["attempt_id"] = "missing-latest"
    write_json(sources[1] / "resolved-launch.yaml", {"config": missing_current})
    recovered_history = publish(missing_gate)
    assert recovered_history.unserved_records == [str(sources[1])]
    assert len(recovered_history.errors) == 1 and "missing-latest" in recovered_history.errors[0]
    assert (second_receipt["config"]["run"]["id"], "native-1") in recovered_history.served_records
    historical_row = stored()["native-1"]
    assert historical_row["gate_result"] == latest_peer_gate and historical_row["optimizer_steps"] == preset.updates
    before = stored()
    command = CliRunner().invoke(
        publication.command(table),
        [
            "--preset",
            preset.name,
            "--marina-url",
            "http://localhost",
            "--run-output",
            str(sources[0]),
            "--attempt",
            "native-0",
        ],
    )
    assert command.exit_code == 0, (command.output, command.exception)
    assert json.loads(command.output)["served_records"] == [[logical_id, "native-0"]]
    after = stored()
    assert after["native-retry"] == before["native-retry"] and after["native-1"] == before["native-1"]
    assert after["native-0"]["evaluations"] == before["native-0"]["evaluations"]
    assert after["native-0"]["gate_result"] == before["native-0"]["gate_result"]
    for root in controls:
        attempt = read_stored_skyrl_attempts(str(root))[0]
        command = CliRunner().invoke(
            publication.command(table),
            [
                "--preset",
                preset.name,
                "--marina-url",
                "http://localhost",
                "--run-output",
                str(root),
                "--attempt",
                attempt,
            ],
        )
        assert command.exit_code == 0, (command.output, command.exception)
    served_controls = [row for row in stored().values() if row.get("is_control") is True]
    assert len(served_controls) == gate.CONTROL_COUNT
    assert all(
        row["branch"] == "main" and row["preset_version"] == report.control.version and row["gate_result"] is None
        for row in served_controls
    )

    assert len({row["native_run_id"] for row in served_controls}) == gate.CONTROL_COUNT
    assert {row["run_id"] for row in served_controls} == {"nightly-batch"}
    control_index = read_stored_skyrl_attempts(str(controls[0]))[1]["control-r0"].index
    assert control_index is not None
    request(
        "http://localhost",
        "POST",
        f"/a/{APPLET_ID}/query",
        json_body={
            "sql": "INSERT INTO rl_runs (run_id, attempt_id, record) VALUES ('invalid-old', 'attempt', '{}')",
            "parameters": {},
        },
    )
    listing = client.get(f"/a/{APPLET_ID}/api/runs").json()
    assert [row["run_id"] for row in listing["skipped_records"]] == ["invalid-old"]
    monkeypatch.setattr(rl_index, "list_attempts", read_progress_index)
    selected = gate.select_controls(control_index, service_url="http://localhost")
    assert {member.attempt_id for member in selected.attempts} == {row["attempt_id"] for row in served_controls}
