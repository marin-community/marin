# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import click
import pyarrow as pa
import pyarrow.parquet as pq
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
from experiments.post_training import rl_presets as presets
from experiments.post_training import rl_publication as publication
from experiments.post_training import snowball_rl as snowball_model
from experiments.post_training.rl_measurements import RUN_INPUTS_FILENAME, RunInputs, lineage_gpu_hours
from experiments.post_training.rl_tests.fixtures import (
    append_metrics,
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
from infra.marina.tests.rl_progress_fixtures import grader_report

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
    assert row.status == "succeeded" and row.failed_facts == []
    native = recorded_launch(attempt)
    (selection,) = native.validation_data
    expected_source = f"{selection.identity}/{selection.relative_path}"
    assert row.evaluation_data_identity == baseline.evaluation_data_identity == expected_source
    assert {item.evaluation_score.profile for item in row.evaluations if item.evaluation_score} == {"eval_lite"}
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


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_snowball_publication_updates_attempts_and_publishes_controls(
    preset_table, rl_progress_client, recorded_runs, monkeypatch
):
    client, request = rl_progress_client
    preset = replace(
        snowball_model.SNOWBALL.by_name["learning-medium"],
        guards=(*snowball_model.SNOWBALL.by_name["learning-medium"].guards, presets.Metric("policy/entropy", True)),
    )
    table = preset_table(preset)
    monkeypatch.setattr(publication, "configure_coreweave_s3", lambda: None)
    pin = gate.control_pin(preset, table=table)

    def write_control(name, artifact_name, **kwargs):
        return recorded_runs(
            name,
            artifact_name,
            attempt_id="pinned-" + artifact_name.rsplit("-r", 1)[1],
            metrics={"policy/entropy": kwargs["score"]},
            **kwargs,
        )

    controls = write_controls(write_control, preset.name)
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
        return {row["attempt_id"]: row for row in client.get(f"/a/{publication.APPLET_ID}/api/runs").json()["runs"]}

    def publish(gate_path):
        return publication.publish_runs(
            [str(root) for root in sources],
            preset,
            table=table,
            gate_path=gate_path,
            service_url="http://localhost",
        )

    def saved_gate(requirements=()):
        report = run(
            gate.gate_step(
                preset,
                handles,
                table=table,
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
    payload = client.get(f"/a/{publication.APPLET_ID}/api/runs").json()
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
        for row in client.get(f"/a/{publication.APPLET_ID}/api/runs").json()["runs"]
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
        (root / RUN_INPUTS_FILENAME).unlink()
    with edit_receipt(controls[0]) as receipt:
        receipt["result"]["provenance"].update(branch="main", dirty=False, base_commit=pin.commit)
    command = CliRunner().invoke(
        publication.command(table), ["--preset", "data-medium", "--marina-url", "http://localhost"]
    )
    assert command.exit_code == 0, (command.output, command.exception)
    assert json.loads(command.output)["unserved_records"] == []
    served_controls = [
        row
        for row in client.get(f"/a/{publication.APPLET_ID}/api/runs").json()["runs"]
        if row["attempt_id"].startswith("pinned-")
    ]
    assert len(served_controls) == pin.replicates
    assert all(row["is_control"] for row in served_controls)
    [main_control] = [row for row in served_controls if row.get("base_commit") is not None]
    assert (main_control["branch"], main_control["base_commit"], main_control["commit"]) == (
        "main",
        pin.commit,
        pin.commit,
    )
    assert all(
        row["series"] == row["label"] == "learning-medium" and row["gate_result"] is None for row in served_controls
    )
    before_ids = {
        (row["run_id"], row["attempt_id"]) for row in client.get(f"/a/{publication.APPLET_ID}/api/runs").json()["runs"]
    }
    stale_table = replace(table, control_pins={**table.control_pins, pin.preset: replace(pin, version="unavailable")})
    missing_controls = publication.publish_controls(stale_table, preset.name, service_url="http://localhost")
    assert not missing_controls.served_records and len(missing_controls.unserved_records) == pin.replicates
    after_ids = {
        (row["run_id"], row["attempt_id"]) for row in client.get(f"/a/{publication.APPLET_ID}/api/runs").json()["runs"]
    }
    assert after_ids == before_ids
