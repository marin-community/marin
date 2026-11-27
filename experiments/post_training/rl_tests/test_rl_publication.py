# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Barrier, local

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from finestore.rl.rl_progress import Verdict
from finestore.rl.run_records import (
    AttemptRecord,
    CheckpointReference,
    LaunchState,
    read_evaluations,
    run_records_root,
    write_attempt,
    write_evaluations,
)
from fsspec.core import url_to_fs
from iris.client.workload import Device, DeviceKind
from iris.resources.state import JobState, TaskState
from marin.execution.artifact import RECORD_FILENAME
from marin.rl.skyrl_records import read_stored_skyrl_attempts, recorded_launch

from experiments.post_training import rl as runner
from experiments.post_training import rl_compose as recipe
from experiments.post_training import rl_gate as gate
from experiments.post_training import rl_index
from experiments.post_training import rl_nightly_controls as nightly
from experiments.post_training import rl_publication as publication
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_index import APPLET_ID
from experiments.post_training.rl_index import list_attempts as read_progress_index
from experiments.post_training.rl_measurements import lineage_gpu_hours
from experiments.post_training.rl_tests.fixtures import (
    FakeIris,
    FakeLauncher,
    FakeMarina,
    nightly_tick,
    replace_evaluations,
    write_artifact_record,
    write_json,
    write_metrics,
)
from experiments.post_training.rl_tests.fixtures import controls as write_controls
from experiments.post_training.skyrl_test_support import read_skyrl_metrics, selection_summary
from infra.marina.tests.rl_progress_fixtures import grader_report

pytestmark = pytest.mark.usefixtures("source_provenance")


@pytest.mark.parametrize(
    "case, steps, elapsed, gpu_hours, failed",
    [
        ("composed", 2, 3600, 40, []),
        ("missing-evaluation", None, 3600, 40, ["evaluation_records_available"]),
        ("lineage", 2, 7200, 80, []),
        ("unknown-grandparent", None, None, None, []),
        ("unknown-allocation", 2, None, None, []),
        ("untracked", None, 3600, 40, ["eval_lite/grader_error_fraction", "eval_lite/initial/grader_error_fraction"]),
    ],
)
def test_publish_cli_rebuilds_stored_measurements(
    preset_table, recorded_runs, monkeypatch, case, steps, elapsed, gpu_hours, failed
):
    preset = replace(snowball_model.SNOWBALL.by_name["learning-medium"], updates=2, measured=range(1, 3))
    table = preset_table(preset)
    untracked = case in {"missing-evaluation", "untracked"}
    fields = {"config.skyrl.trainer.logger": ["console", "files"]} if untracked else {}
    path = recorded_runs(preset.name, "tests/stored-progress", attempt_id="native-attempt", table=table, receipt=fields)
    export = str(path / "exports")
    record = read_stored_skyrl_attempts(str(path))[1]["native-attempt"]
    if case in {"lineage", "unknown-grandparent", "unknown-allocation"}:
        config = deepcopy(record.native.config)
        config["run"]["attempt_id"] = "ancestor"
        if case == "unknown-allocation":
            config["iris"]["allocation"].pop("gpus_per_node")
        result = record.native.result.model_copy(
            update={
                "state": LaunchState.FAILED,
                "started_at": "2026-11-06T16:40:00+00:00",
                "finished_at": "2026-11-06T17:40:00+00:00",
            }
        )
        ancestor = AttemptRecord(
            run_id=record.run_id,
            attempt_id="ancestor",
            config=config,
            result=result,
            metrics_started_at=result.started_at,
            elapsed_excluding_eval_seconds=123,
        )
        if case == "unknown-grandparent":
            parent = CheckpointReference(export_root=str(path / "unavailable"), attempt_id="grandparent", step=1)
            ancestor = ancestor.model_copy(update={"resumed_step": 1, "parent": parent})
        write_attempt(export, ancestor)
        write_attempt(
            export,
            AttemptRecord(
                run_id=record.run_id,
                attempt_id=record.attempt_id,
                resumed_step=1,
                elapsed_excluding_eval_seconds=321,
                parent=CheckpointReference(export_root=export, attempt_id=ancestor.attempt_id, step=1),
            ),
        )
        metrics = read_skyrl_metrics(str(path))
        metrics[1]["attempt_id"] = ancestor.attempt_id
        write_metrics(path, metrics)
    if untracked:
        write_metrics(path, {})
        write_attempt(export, record.model_copy(update={"metrics_started_at": None}))
    if case == "missing-evaluation":
        replace_evaluations(path, [])
    if case == "composed":
        write_json(path / "exports/grader_reports/native-attempt/grader_report.json", grader_report())
    marina = FakeMarina(monkeypatch)
    result = _publish_cli(table, preset.name, path, "native-attempt")
    assert result.exit_code == 0, result.output
    row = marina.by_attempt("native-attempt")
    assert (row.optimizer_steps, row.elapsed_seconds, row.failed_facts) == (steps, elapsed, failed)
    assert row.status == ("failed" if failed else "succeeded")
    if case in {"lineage", "unknown-grandparent", "unknown-allocation"}:
        attempt = read_stored_skyrl_attempts(str(path))[1]["native-attempt"]
        assert lineage_gpu_hours(attempt, recorded_launch(attempt)) == gpu_hours
    if case == "lineage":
        assert row.elapsed_excluding_eval_seconds == 444
    if case == "composed":
        assert (
            row.engine["vllm_version"] == "fixture-vllm" and row.grader_report.model_dump(mode="json") == grader_report()
        )


def _publish_cli(table, preset, output, attempt, *options):
    return CliRunner().invoke(
        publication.command(table),
        [
            "--preset",
            preset,
            "--run-output",
            str(output),
            "--attempt",
            attempt,
            *options,
        ],
    )


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
    row = publication.build_record(str(source), newer, table=preset_table(newer), attempt=attempt)
    assert row.status == "succeeded" and row.failed_facts == [], (row.failed_facts, row.evidence)
    native = recorded_launch(attempt)
    (selection,) = native.validation_data
    expected_source = f"{selection.identity}/{selection.relative_path}"
    assert row.evaluation_data_identity == baseline.evaluation_data_identity == expected_source
    assert {item.evaluation_score.profile for item in row.evaluations if item.evaluation_score} == {"eval_lite"}
    assert {
        item.evaluation_score.step for item in row.evaluations if item.evaluation_score and item.evaluation_score.final
    } == {preset.updates}

    def family_identity(summary):
        return next(
            score.comparability
            for entry in summary.evaluations
            if (score := entry.evaluation_score) and score.final and score.statistic == "family_mean"
        )

    identity = family_identity(baseline)
    assert identity is not None and family_identity(row) == identity
    outputs = read_evaluations(native.export_root, attempt_id=attempt.attempt_id, evaluation="eval/")
    item = next(item for item in outputs if item.step == preset.updates)
    unanswered = item.answers[0].model_copy(update={"verification_status": "unavailable", "verifier_score": None})

    def score_identity(record=attempt):
        return family_identity(publication.build_record(str(source), preset, table=table, attempt=record))

    write_evaluations(native.export_root, [item.model_copy(update={"answers": [unanswered, *item.answers[1:]]})])
    assert score_identity() != identity
    write_evaluations(
        native.export_root,
        [
            item.model_copy(update={"strata": item.strata.model_copy(update={"family": "alternate-family"})})
            for item in outputs
            if item.task_id == outputs[0].task_id
        ],
    )
    grouped = score_identity()
    assert grouped != identity

    source_path = Path(selection.uri) / selection.relative_path
    samples = pq.read_table(source_path).to_pylist()
    task = json.loads(samples[0]["task_spec"])
    task.setdefault("labels", {})["comparison_fixture"] = "changed-content-with-same-id"
    samples[0]["task_spec"] = json.dumps(task)
    pq.write_table(pa.Table.from_pylist(samples), source_path)
    changed_tasks = score_identity()
    assert changed_tasks != grouped
    altered = deepcopy(attempt.native.config)
    altered["skyrl"]["generator"]["eval_sampling_params"]["temperature"] = 0.314
    assert score_identity(attempt.model_copy(update={"config": altered})) != changed_tasks


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


def test_nightly_retries_publication_and_refills_deficit(
    recorded_runs, monkeypatch, tmp_path, cache_native_compositions
):
    table = snowball_model.SNOWBALL
    preset = table.by_name["speed-short"]
    state_dir = tmp_path / "pending"
    state_dir.mkdir()
    pending = state_dir / f"snowball-{preset.name}.json"
    for run in ("old-1", "old-2"):
        recorded_runs(preset.name, run, control=True)
    marina = FakeMarina(monkeypatch, initial=rl_index.list_attempts())
    launcher = FakeLauncher(monkeypatch, table=table)

    submissions = iter((("first", "3"), ("refill", "2")))

    def observe_submit(job, request):
        batch, count = next(submissions)
        assert request["entrypoint"].command == [
            "python",
            "-m",
            snowball_model.__name__,
            "--preset",
            preset.name,
            "--control",
            batch,
            "--replicates",
            count,
            "--version",
            recipe.RUN_ARTIFACT_VERSION,
            "--run",
        ]
        assert not pending.exists()
        assert publication.CONTROL_SUBMISSION_ENV not in request["environment"].env_vars

    iris = FakeIris(monkeypatch, on_submit=observe_submit)

    def tick(batch):
        return nightly_tick(monkeypatch, state_dir, batch, f"snowball/{preset.name}")

    assert tick("first") == 0
    handoff = publication.ControlSubmission.model_validate_json(pending.read_text())
    assert launcher.requests == {(preset.name, "first"): 3} and len(handoff.outputs) == 3
    root = str(iris.set_job(handoff.job, JobState.RUNNING))

    def write(output, attempt, state, **receipt):
        name = Path(output).parent.relative_to(tmp_path).as_posix()
        return recorded_runs(
            preset.name, name, control=True, coordinator=root, attempt_id=attempt, state=state, receipt=receipt
        )

    accepted, lost, _ = handoff.outputs
    write(accepted, "accepted", "submitted")
    assert tick("running") == 0
    assert marina.by_attempt("accepted").status == "running" and len(launcher.requests) == 1
    iris.set_job(root, JobState.FAILED)
    write(accepted, "accepted", "succeeded")
    write(lost, "preempted", "failed", **{"result.iris_job_state": "worker_failed"})
    iris.set_job(root + "/training", JobState.WORKER_FAILED, reason="Evicted")
    marina.fail_once.add("accepted")
    assert tick("publication-error") == 1
    assert publication.ControlSubmission.model_validate_json(pending.read_text()) == handoff
    assert marina.by_attempt("preempted").status == "failed" and len(launcher.requests) == 1
    assert tick("refill") == 0
    assert marina.by_attempt("accepted").status == "succeeded"
    assert launcher.requests == {(preset.name, "first"): 3, (preset.name, "refill"): 2}
    assert publication.ControlSubmission.model_validate_json(pending.read_text()).job != root


def test_nightly_distinguishes_interruptions_and_failures(
    recorded_runs, monkeypatch, tmp_path, subtests, cache_native_compositions
):
    scenarios = (
        ("evicted", JobState.WORKER_FAILED, {}, "Evicted to make room for higher-priority work", 0),
        ("killed-after-preparing", JobState.KILLED, {}, "", 0),
        ("native-failed", JobState.FAILED, {}, "training failed", 1),
        ("worker-with-failed-task", JobState.WORKER_FAILED, {TaskState.FAILED: 1}, "worker failed", 1),
        ("oom", JobState.WORKER_FAILED, {}, "OOMKilled", 1),
        ("host-loss", JobState.WORKER_FAILED, {}, "host crash", 1),
        ("exhausted-memory", JobState.WORKER_FAILED, {}, "out of memory", 1),
        ("failed-reconcile", JobState.WORKER_FAILED, {}, "worker reconcile failure threshold exceeded", 1),
        ("unhealthy-worker", JobState.WORKER_FAILED, {}, "unhealthy worker failed", 1),
        ("failed-before-preparing", JobState.FAILED, None, "", 1),
        ("lost-before-preparing", None, None, "", 0),
    )
    for scenario, state, tasks, reason, expected in scenarios:
        with subtests.test(scenario=scenario), monkeypatch.context() as patch:
            directory = tmp_path / scenario
            directory.mkdir()
            marina, iris = FakeMarina(patch), FakeIris(patch)
            FakeLauncher(patch)
            root = f"/runner/checkpoints-snowball-rl-controls-speed-medium-{scenario}-coordinator"
            output = tmp_path / scenario / recipe.RUN_ARTIFACT_VERSION
            if tasks is not None:
                output = recorded_runs(
                    "speed-medium",
                    scenario,
                    state="submitted" if state == JobState.KILLED else "failed",
                    control=True,
                    coordinator=root,
                    receipt={"result.iris_job_state": None if state == JobState.KILLED else state},
                )
                message = {"message": reason} if scenario == "host-loss" else {"reason": reason}
                iris.set_job(root + "/training", state, tasks=tasks, **message)
            if state is not None:
                iris.set_job(root, JobState.KILLED if state == JobState.KILLED else JobState.FAILED)
            pending = directory / (
                "snowball-speed-medium.migration.json" if scenario == "evicted" else "snowball-speed-medium.json"
            )
            pending.write_text(publication.ControlSubmission(job=root, outputs=(str(output),)).model_dump_json())
            assert nightly_tick(patch, directory, "reconcile", "snowball/learning-medium") == expected
            assert not pending.exists()
            assert len(marina.rows) == int(tasks is not None)


def test_nightly_reserves_mochi_racks_and_skips_duplicate_launches(
    monkeypatch, tmp_path, capsys, cache_native_compositions
):
    marina, iris = FakeMarina(monkeypatch), FakeIris(monkeypatch)
    launcher = FakeLauncher(monkeypatch)
    assert nightly_tick(monkeypatch, tmp_path, "fresh", "mochi") == 0
    assert sorted(launcher.requests.values()) == [1, 2]
    launcher.requests.clear()
    speed = "/runner/checkpoints-mochi-rl-controls-speed-medium-running-coordinator"
    iris.set_job(speed, JobState.RUNNING)
    gb200 = Device(DeviceKind.GPU, "GB200", 4)
    iris.set_job(speed + "/training", JobState.RUNNING, device=gb200, nodes=15)
    pending = tmp_path / "mochi-speed-medium.json"
    pending.write_text(
        publication.ControlSubmission(
            job=speed, outputs=tuple(str(tmp_path / f"unprepared-{n}") for n in range(2))
        ).model_dump_json()
    )
    iris.set_job("/other/users-test", JobState.RUNNING, device=gb200, nodes=54)
    assert nightly_tick(monkeypatch, tmp_path, "limited", "mochi snowball/speed-medium") == 0
    assert launcher.requests == {("learning-medium", "limited"): 1}
    assert "Skipping mochi/speed-medium: pending record" in capsys.readouterr().out

    multi = iris.set_job("/atqamar/custom-gpu-experiment", JobState.PENDING, device=gb200, nodes=32)
    assert nightly_tick(monkeypatch, tmp_path, "external", "mochi") == 0
    assert "Skipping mochi/learning-medium: no free rack" in capsys.readouterr().out
    assert launcher.requests == {("learning-medium", "limited"): 1}

    pending.unlink()
    iris.set_job(speed, JobState.SUCCEEDED)
    iris.set_job(speed + "/training", JobState.SUCCEEDED)
    iris.set_job(str(multi), JobState.BUILDING, device=gb200, nodes=32)
    iris.set_job("/runner/another-experiment/worker", JobState.RUNNING, device=gb200, nodes=17)
    assert nightly_tick(monkeypatch, tmp_path, "both-owners", "mochi") == 0
    assert launcher.requests == {("learning-medium", "limited"): 1}

    def unavailable(*, state, prefix):
        assert prefix in {"/runner/", "/atqamar/"} and state in nightly._ACTIVE_STATES
        raise ConnectError(Code.OUT_OF_RANGE, "deep offset")

    monkeypatch.setattr(iris, "list_jobs", unavailable)
    iris.set_job(speed, JobState.RUNNING)
    pending.write_text(
        publication.ControlSubmission(
            job=speed, outputs=tuple(str(tmp_path / str(i)) for i in range(3))
        ).model_dump_json()
    )
    assert nightly_tick(monkeypatch, tmp_path, "full", "mochi") == 0
    assert "Mochi rack scan failed" not in capsys.readouterr().err
    pending.unlink()
    iris.set_job(speed, JobState.SUCCEEDED)

    duplicate = "/runner/checkpoints-snowball-rl-controls-speed-medium-lost-reply-coordinator"
    iris.set_job(duplicate, JobState.BUILDING)
    assert nightly_tick(monkeypatch, tmp_path, "lost-reply", "snowball/speed-medium") == 0
    assert "Skipping snowball/speed-medium: active coordinator" in capsys.readouterr().out
    iris.set_job(duplicate, JobState.SUCCEEDED)
    assert nightly_tick(monkeypatch, tmp_path, "scan-failed", "mochi snowball/speed-medium") == 0
    assert "Mochi rack scan failed:" in capsys.readouterr().err
    assert launcher.requests == {("learning-medium", "limited"): 1, ("speed-medium", "scan-failed"): 1}
    assert not marina.rows


def test_nightly_reserves_snowball_budget_across_ticks(monkeypatch, tmp_path, cache_native_compositions):
    FakeMarina(monkeypatch)
    iris, launcher = FakeIris(monkeypatch), FakeLauncher(monkeypatch)
    models = nightly.select_presets("snowball")
    assert {p.name: nightly._attempt_cost(model, p) for model, p in models} == {
        "speed-short": 9_996,
        "data-short": 9_996,
        "speed-medium": 49_836,
        "speed-long": 49_836,
        "learning-medium": 49_836,
    }
    assert nightly_tick(monkeypatch, tmp_path, "fresh", "snowball/speed-short snowball/data-short") == 0
    assert sorted(launcher.requests.values()) == [1, 5]
    launcher.requests.clear()
    running = "/runner/checkpoints-snowball-rl-controls-speed-long-running-coordinator"
    iris.set_job(running, JobState.PENDING)
    path = tmp_path / "snowball-speed-long.json"
    path.write_text(publication.ControlSubmission(job=running, outputs=(str(tmp_path / "queued"),)).model_dump_json())
    assert nightly_tick(monkeypatch, tmp_path, "reserved", "snowball/speed-medium snowball/speed-short mochi") == 0
    assert launcher.requests == {("speed-short", "reserved"): 1}
    iris.set_job(running, JobState.KILLED)
    assert nightly_tick(monkeypatch, tmp_path, "refill", "snowball/speed-medium") == 0
    assert launcher.requests[("speed-medium", "refill")] == 1
    assert not path.exists()

    mochi = "/runner/checkpoints-mochi-rl-controls-speed-medium-untracked-coordinator"
    iris.set_job(mochi, JobState.BUILDING)
    assert nightly_tick(monkeypatch, tmp_path, "shared", "snowball/speed-medium") == 0
    assert ("speed-medium", "shared") not in launcher.requests


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
    write_json(
        event, {"issue": {"number": 14, "pull_request": {}}, "comment": {"body": body, "user": {"login": "contributor"}}}
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
    monkeypatch.setattr("sys.argv", ["nightly", "--comment-event", str(event)])
    assert nightly.main() == 0
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
        assert "repos/example/marin/actions/workflows/marin-skyrl-nightly.yaml/dispatches" in dispatch
        assert json.loads(payload) == {"ref": "main", "inputs": {"controls": "true", "presets": selection}}
        assert comments in reply
        assert reply[-1] == "body=🤖 [RL control run](https://github.com/example/marin/actions/runs/145)"


def test_publication_qualifies_terminal_evidence(recorded_runs, monkeypatch, tmp_path, subtests):
    table = snowball_model.SNOWBALL
    preset = table.by_name["speed-medium"]
    write_controls(recorded_runs, preset.name)
    control_rows = rl_index.list_attempts()
    marina = FakeMarina(monkeypatch)
    root = tmp_path / "report"

    def save_report(attempt):
        report = gate.GateReport(
            preset=preset.name,
            verdict=Verdict.FAIL,
            effect=None,
            reason="control guard failed",
            run=gate.RunMetadata(preset_name=preset.name, replicates=1, attempt_ids=(attempt,)),
            control=gate.ControlMetadata(version="1234abcd", attempts=()),
            invariants=(),
            comparisons=(),
            gaps=(),
        )
        write_artifact_record(root, "report", gate.GATE_REVISION, gate.GateReport, result=report.result_payload())

    for case in (
        "stale-gate",
        "guard-fail",
        "storage-fault",
        "invalid-provenance",
        "missing-receipt",
        "issue-conflict",
        "interrupted-without-report",
    ):
        with subtests.test(case=case), monkeypatch.context() as patch:
            marina.rows.clear()
            fields = {"result.provenance": None} if case == "invalid-provenance" else {}
            path = recorded_runs(preset.name, f"tests/{case}", attempt_id=case, receipt=fields)
            save_report("different-attempt" if case == "stale-gate" else case)
            gate_path = root
            if case == "invalid-provenance":
                for control in control_rows:
                    marina.publish(control, None)
                result = CliRunner().invoke(
                    gate.command(table), ["--preset", preset.name, "--candidate", f"tests/{case}"]
                )
                assert "no recorded source provenance" in result.output, result.output
                gate_path = Path(json.loads(result.stdout.splitlines()[-1])["report"]).parent
                marina.rows.clear()
            if case == "missing-receipt":
                (path / "terminal.json").unlink()
                (path / "resolved-launch.yaml").unlink()
                shutil.rmtree(run_records_root(str(path / "exports")))
            if case == "storage-fault":
                filesystem = type(url_to_fs(str(root))[0])
                exists = filesystem.exists

                def read_with_fault(self, key, exists=exists, **kwargs):
                    if str(key) == str(root / RECORD_FILENAME):
                        raise OSError("Gate storage unavailable")
                    return exists(self, key, **kwargs)

                patch.setattr(filesystem, "exists", read_with_fault)
            if case == "issue-conflict":
                assert _publish_cli(table, preset.name, path, case, "--issue", "73").exit_code == 0
                assert _publish_cli(table, preset.name, path, case, "--issue", "74").exit_code == 1
                assert marina.by_attempt(case).issue == 73
                continue
            interrupted = case == "interrupted-without-report"
            if interrupted:
                gate_path = tmp_path / "unwritten-report"
            outcome = publication.publish_runs(
                [str(path)], preset, table=table, gate_path=str(gate_path), interrupted=interrupted
            )
            if case in {"stale-gate", "storage-fault", "missing-receipt"}:
                assert outcome.unserved_records and not marina.rows
            else:
                assert not outcome.unserved_records
                row = marina.by_attempt(case)
                if interrupted:
                    assert row.status == "failed" and row.failed_facts == ["required_gate_report"]
                    continue
                assert row.status == "succeeded" and row.failed_facts == []
                assert row.gate_result.verdict == ("invalid" if case == "invalid-provenance" else "fail")


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_snowball_publication_updates_attempts_and_publishes_controls(rl_progress_client, recorded_runs, monkeypatch):
    client, request = rl_progress_client
    table = snowball_model.SNOWBALL
    preset = table.by_name["speed-medium"]
    outputs = [str(path) for path in write_controls(recorded_runs, preset.name, cohort_run_id="nightly-batch")]
    unavailable = True

    def lost_response(origin, method, path, **kwargs):
        nonlocal unavailable
        response = request(origin, method, path, **kwargs)
        if method == "POST" and path.endswith("/api/runs") and unavailable:
            unavailable = False
            raise RuntimeError("Response unavailable after insert")
        return response

    monkeypatch.setattr("marina.client.marina_request", lost_response)

    def publish():
        return publication.publish_runs(outputs, preset, table=table, service_url="http://localhost")

    assert len(publish().unserved_records) == 1
    assert not publish().unserved_records
    rows = client.get(f"/a/{APPLET_ID}/api/runs").json()["runs"]
    assert len(rows) == gate.CONTROL_COUNT
    assert len({row["native_run_id"] for row in rows}) == gate.CONTROL_COUNT
    assert {row["run_id"] for row in rows} == {"nightly-batch"}

    def wrong_identity(origin, method, path, **kwargs):
        response = request(origin, method, path, **kwargs)
        return response | {"attempt_id": "wrong"} if method == "POST" else response

    monkeypatch.setattr("marina.client.marina_request", wrong_identity)
    assert len(publish().unserved_records) == gate.CONTROL_COUNT
    monkeypatch.setattr("marina.client.marina_request", request)
    monkeypatch.setattr(rl_index, "list_attempts", read_progress_index)
    index = read_stored_skyrl_attempts(outputs[0])[1]["control-r0"].index
    controls = gate.select_controls(index, service_url="http://localhost")
    assert {attempt.attempt_id for attempt in controls.attempts} == {row["attempt_id"] for row in rows}
