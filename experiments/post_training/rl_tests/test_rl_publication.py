# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import click
import pytest
from click.testing import CliRunner
from finestore.rl.rl_progress import AttemptSummary
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
from marin.execution.artifact import RECORD_FILENAME, Artifact
from marin.execution.lazy import ArtifactStep, run
from marin.rl.skyrl import ArtifactHfModel, SkyRLRun
from marin.rl.skyrl_records import read_skyrl_metrics, read_stored_skyrl_attempts, recorded_launch
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
from experiments.post_training.rl_measurements import RunInputs, lineage_gpu_hours
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
from experiments.post_training.skyrl_test_support import selection_summary
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
    inputs = RunInputs(
        preset_name=name,
        settings=("trainer.policy.optimizer_config.lr=2e-06", "trainer.policy.optimizer_config.weight_decay=0.01")
        + ((f"trainer.max_steps={expected_updates}",) if name == "train-long" else ()),
        tasks=f"snowball-100k@{snowball_model.DEFAULT_SELECTION_VERSION}",
        init=recipe.stored_initializer(recipe.published_model("weights@2026.11.10", preset=preset), preset=preset),
    )
    native_table = table
    if case == "native-no-evaluation":
        without_evaluation = replace(preset, evaluations=(), checkpoints="none", checkpoint_every=None, export="none")
        native_table = replace(
            table, presets=tuple(without_evaluation if row.name == name else row for row in table.presets)
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
    preset_label, *settings = row["label"].split()
    assert preset_label == name
    assert set(settings) == {*inputs.settings, f"tasks={inputs.tasks}", "init=weights@2026.11.10"}
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


def test_stored_evaluation_keeps_its_original_selection(preset_table, recorded_runs):
    preset = replace(snowball_model.SNOWBALL.by_name["learning-medium"], updates=2, measured=range(1, 3))
    table = preset_table(preset)
    source = recorded_runs(preset.name, "tests/eval-coverage", attempt_id="recorded")
    attempt = read_stored_skyrl_attempts(str(source))[1]["recorded"]
    baseline = publication.build_record(str(source), preset, table=table, attempt=attempt)
    newer = replace(preset, held_out_tasks="eval-lite@2026.11.24.99")
    write_json(Path(recipe.selection_artifact(newer.held_out_tasks).path()) / "summary.json", selection_summary(40))
    row = publication.build_record(str(source), newer, table=preset_table(newer), attempt=attempt)
    assert row.status == "succeeded" and row.failed_facts == []
    assert row.evaluation_data_identity == baseline.evaluation_data_identity
    assert row.evaluation_task_set == baseline.evaluation_task_set


def test_one_issue_is_reserved_when_publishers_disagree(tmp_path, monkeypatch):
    export_root = str(tmp_path / "exports")
    conditional_object = publication.conditional_object
    rendezvous = Barrier(2, timeout=10)

    def concurrent_object(path):
        object_store = conditional_object(path)

        def write(data, *, expected_version):
            rendezvous.wait()
            return object_store.write(data, expected_version=expected_version)

        return SimpleNamespace(read=object_store.read, write=write)

    monkeypatch.setattr(publication, "conditional_object", concurrent_object)

    def claim(issue):
        try:
            return publication.reserve_issue(export_root, issue)
        except publication.IssueConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        assigned = [issue for issue in executor.map(claim, (73, 74)) if issue is not None]

    [winner] = assigned
    assert publication.lineage_issue(export_root, None) == winner


def test_nightly_fills_only_the_current_main_control_cohort(monkeypatch):
    launch = {"config": "composed"}
    monkeypatch.setattr(nightly, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(gate, "compose_preset", lambda preset: launch)
    monkeypatch.setattr(gate, "config_version", lambda config: "current")
    branch = SimpleNamespace(branch="main", dirty=False)
    monkeypatch.setattr(nightly.Provenance, "capture", lambda *_: branch)
    monkeypatch.setattr(
        "sys.argv",
        ["rl_nightly_controls", "--model", "snowball", "--preset", "learning-medium", "--batch", "123-1"],
    )
    base = AttemptSummary.model_validate_json(
        json.dumps(
            progress_record()
            | {
                "model": "snowball",
                "preset_version": "current",
                "is_control": True,
                "dirty": False,
                "branch": "main",
                "commit": "main-source",
                "base_commit": "main-source",
            }
        )
    )
    controls = [base.model_copy(update={"native_run_id": f"control-{i}", "attempt_id": f"first-{i}"}) for i in range(5)]
    retry = controls[0].model_copy(update={"attempt_id": "retry", "completed_at": "2026-11-02T02:00:00Z"})
    records = [
        *controls[:3],
        retry.model_copy(update={"status": "running", "completed_at": None, "elapsed_seconds": None}),
        base.model_copy(update={"run_id": "branch-control", "branch": "pull-branch"}),
        base.model_copy(update={"run_id": "old-control", "preset_version": "old"}),
        base.model_copy(update={"run_id": "failed-control", "status": "failed"}),
    ]
    reads = []

    def read_attempts(_service_url=None):
        reads.append(True)
        return records

    monkeypatch.setattr(nightly.rl_index, "list_attempts", read_attempts)
    native = SimpleNamespace(result=SimpleNamespace(state=SimpleNamespace(is_terminal=True)))
    monkeypatch.setattr(
        nightly, "read_stored_skyrl_attempts", lambda uri: ("retry", {"retry": SimpleNamespace(native=native)})
    )
    recovered = []

    def recover(paths, preset, *, table, attempt_id):
        assert paths == [retry.artifact_uri] and attempt_id == retry.attempt_id
        assert table.model == retry.model and preset.name == retry.series
        recovered.append(attempt_id)
        records[3] = retry
        return publication.PublicationOutcome(served_records=[(retry.run_id, attempt_id)])

    monkeypatch.setattr(publication, "publish_runs", recover)
    submissions = []

    def submit(command, *, check):
        submissions.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nightly.subprocess, "run", submit)

    assert nightly.main() == 0
    records.extend(controls[3:])
    assert nightly.main() == 0
    branch.branch = "pull-branch"
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    assert nightly.main() == 0

    assert len(reads) == 3 and recovered == ["retry"]
    assert [command[command.index("--replicates") + 1] for command in submissions] == ["2", "1", "5"]
    for command in submissions:
        assert command[1:5] == ["-m", "experiments.post_training.snowball_rl", "--preset", "learning-medium"]
        assert command[command.index("--control") + 1] == "123-1"
        assert command[-2:] == ["--run", "--wait"]


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_snowball_publication_updates_attempts_and_publishes_controls(
    preset_table, rl_progress_client, recorded_runs, monkeypatch, cache_native_compositions
):
    cache_native_compositions()
    client, request, _ = rl_progress_client
    preset = replace(
        snowball_model.SNOWBALL.by_name["learning-medium"],
        guards=(*snowball_model.SNOWBALL.by_name["learning-medium"].guards, presets.Metric("policy/entropy", True)),
    )
    table = preset_table(preset)
    monkeypatch.setattr(publication, "configure_coreweave_s3", lambda: None)
    controls = write_controls(recorded_runs, preset.name, cohort_run_id="nightly-batch")
    for index, root in enumerate(controls):
        metrics = read_skyrl_metrics(str(root))
        for row in metrics.values():
            row["policy/entropy"] = 0.48 + index * 0.01
        append_metrics(root, metrics)
    sources = [
        recorded_runs(
            preset.name, f"tests/publication-{i}", metrics={"policy/entropy": float(i)}, attempt_id=f"native-{i}"
        )
        for i in range(2)
    ]
    handles = tuple(
        ArtifactStep.adopt(f"tests/publication-{i}", recipe.RUN_ARTIFACT_VERSION, str(root), kind=SkyRLRun)
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
    first = publish(first_gate)
    assert len(first.served_records) == 1 and first.errors == ["Response unavailable after insert"]
    persisted = stored()
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
        "control_version": report.control.version,
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
    served_controls = [row for row in stored().values() if row["is_control"]]
    assert len(served_controls) == gate.CONTROL_COUNT
    assert all(
        row["branch"] == "main" and row["preset_version"] == report.control.version and row["gate_result"] is None
        for row in served_controls
    )

    assert len({row["native_run_id"] for row in served_controls}) == gate.CONTROL_COUNT
    assert {row["run_id"] for row in served_controls} == {"nightly-batch"}
    control_index = read_stored_skyrl_attempts(str(controls[0]))[1]["control-r0"].index
    assert control_index is not None
    monkeypatch.setattr(rl_index, "list_attempts", read_progress_index)
    selected = gate.select_controls(control_index, service_url="http://localhost")
    assert {member.attempt_id for member in selected.attempts} == {row["attempt_id"] for row in served_controls}
