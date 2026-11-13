# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import click
import pytest
from click.testing import CliRunner
from finestore.rl.run_records import (
    AttemptRecord,
    CheckpointReference,
    LaunchReceipt,
    read_attempts,
    run_records_root,
    update_attempt,
)
from finestore.store import DataStore
from fsspec.core import url_to_fs
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, run
from marin.rl.rl_progress import RunSummary
from marin.rl.skyrl import SkyRLRun
from marin.rl.skyrl_records import read_skyrl_metrics, read_stored_skyrl_attempts
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import publish_rl_progress as publisher
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_publication as publication
from experiments.post_training.snowball_tests.fixtures import append_metrics, write_json, write_metrics, write_receipt

pytestmark = pytest.mark.usefixtures("source_provenance")


@pytest.mark.parametrize(
    "case, expected_updates, expected_time, expected_facts",
    [
        ("native-composed", 2, 3600, []),
        ("native-failed", 0, 3600, ["native_attempt", "planned_updates"]),
        ("invariant-failed", 2, 3600, ["finite_losses_and_gradients"]),
        ("checkpoint-lineage", 2, 7200, []),
        ("lineage-missing-window", 2, None, []),
        ("lineage-unknown-parent", None, None, []),
        ("lineage-unknown-grandparent", None, None, []),
        ("canonical-only", None, None, []),
        ("historical-retry", 2, 3600, ["native_attempt"]),
        ("unrelated-attempt", 2, 3600, []),
        ("missing-startup", 2, 3600, []),
        ("untracked-native", None, None, []),
        ("train-long", 2, 3600, []),
        ("updates-override", 1, 3600, []),
        ("negative-duration", None, None, []),
    ],
)
@pytest.mark.timeout(180)
def test_native_attempt_records_keep_identity_measurements_and_failures(
    recorded_runs, monkeypatch, case, expected_updates, expected_time, expected_facts
):
    def forbid_live(*args, **kwargs):
        pytest.fail("Host record builder requested live native execution/state")

    monkeypatch.setattr("marin.rl.skyrl.run_skyrl", forbid_live)
    monkeypatch.setattr("marin.rl.skyrl.read_skyrl_attempt", forbid_live)
    name = "train-long" if case == "train-long" else "learning-medium"
    preset = replace(presets.PRESETS[name], updates=2, measured=range(1, 3))
    monkeypatch.setattr(presets, "PRESETS", dict(presets.PRESETS) | {name: preset})
    weights = Path(presets.artifact_prefix(preset.recipe.layout.cluster)) / "weights" / "2026.11.10"
    initializer = SkyRLRun(
        path=str(weights),
        hf_model_uri=str(weights / "policy"),
        global_step=1,
        tokenizer_uri=preset.model.tokenizer_uri,
        tokenizer_revision=preset.model.tokenizer_revision,
        checkpoint_root=str(weights / "checkpoints"),
        draft_checkpoint_root=None,
        terminal_manifest_uri=str(weights / "terminal.json"),
        iris_job_id="/tests/initializer",
    )
    write_json(
        weights / RECORD_FILENAME,
        ArtifactRecord(
            name="weights",
            version="2026.11.10",
            fingerprint="fixture-initializer",
            result_type=result_type_name(SkyRLRun),
            result=initializer.result_payload(),
        ).model_dump(mode="json"),
    )
    inputs = gate.GateInputs(
        preset_name=name,
        settings=("trainer.policy.optimizer_config.lr=2e-06", "trainer.policy.optimizer_config.weight_decay=0.01"),
        tasks=f"snowball-100k@{presets.DEFAULT_SELECTION_VERSION}",
        init="weights@2026.11.10",
    )
    source = recorded_runs(
        name,
        "tests/stored-progress",
        attempt_id="native-attempt",
        inputs=inputs,
        metrics={"attempt_id": "native-attempt"} if case == "historical-retry" else None,
    )
    receipt = json.loads((source / "terminal.json").read_text())
    if case in {"native-failed", "historical-retry"}:
        receipt["result"].update(state="failed", model=None)
    if case == "untracked-native":
        receipt["config"]["skyrl"]["trainer"]["logger"] = ["console", "files"]
    write_receipt(source, receipt)
    metrics = read_skyrl_metrics(str(source))
    records = read_attempts(str(source / "exports"))
    if case in {
        "checkpoint-lineage",
        "lineage-missing-window",
        "lineage-unknown-grandparent",
        "unrelated-attempt",
        "negative-duration",
    }:
        ancestor = deepcopy(receipt)
        ancestor["config"]["run"]["attempt_id"] = "ancestor"
        ancestor["result"].update(
            state="failed",
            started_at="2026-11-06T16:40:00+00:00",
            finished_at=(
                None
                if case == "lineage-missing-window"
                else "2026-11-06T15:40:00+00:00" if case == "negative-duration" else "2026-11-06T17:40:00+00:00"
            ),
        )
        native_ancestor = LaunchReceipt.model_validate(ancestor)
        records["ancestor"] = AttemptRecord(
            run_id=receipt["config"]["run"]["id"],
            attempt_id="ancestor",
            config=native_ancestor.config,
            result=native_ancestor.result,
            metrics_started_at=native_ancestor.result.started_at,
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
        if case != "unrelated-attempt":
            records["native-attempt"] = records["native-attempt"].model_copy(
                update={
                    "resumed_step": 1,
                    "parent": CheckpointReference(export_root=str(source / "exports"), attempt_id="ancestor", step=1),
                }
            )
            metrics[1]["attempt_id"] = "ancestor"
    if case == "lineage-unknown-parent":
        records["native-attempt"] = records["native-attempt"].model_copy(
            update={
                "resumed_step": 1,
                "parent": CheckpointReference(export_root=str(source / "unknown-exports"), attempt_id="unknown", step=1),
            }
        )
    if case == "native-failed":
        metrics = {}
    if case == "untracked-native":
        metrics = {}
        records["native-attempt"] = records["native-attempt"].model_copy(update={"metrics_started_at": None})
    if case == "invariant-failed":
        metrics[1]["policy/raw_grad_norm"] = float("nan")
    if case == "missing-startup":
        records["native-attempt"] = records["native-attempt"].model_copy(
            update={"started_at": None, "vllm_version": None}
        )
    if case == "historical-retry":
        retry = deepcopy(receipt)
        retry["config"]["run"]["attempt_id"] = "retry"
        retry["result"].update(
            iris_job_id="/tests/retry",
            started_at="2026-11-06T20:00:00+00:00",
            finished_at="2026-11-06T20:05:00+00:00",
        )
        native_retry = LaunchReceipt.model_validate(retry)
        records["retry"] = AttemptRecord(
            run_id=retry["config"]["run"]["id"],
            attempt_id="retry",
            config=native_retry.config,
            result=native_retry.result,
            started_at=retry["result"]["started_at"],
        )
        write_json(source / "resolved-launch.yaml", {"config": retry["config"]})
    with DataStore.open(run_records_root(str(source / "exports"))) as store:
        for record in records.values():
            update_attempt(store, record)
    write_metrics(source, metrics)
    if case in {"native-failed", "historical-retry"}:
        (source / "terminal.json").unlink()
    if case == "canonical-only":
        shutil.rmtree(run_records_root(str(source / "exports")))
    if case == "native-composed":
        shutil.rmtree(source / "exports/dumped_evals")
    if case == "updates-override":
        receipt["config"]["skyrl"]["trainer"]["max_steps"] = 1
        write_receipt(source, receipt)
    if case == "negative-duration":
        receipt["result"]["finished_at"] = "2026-11-06T20:40:00+00:00"
        write_receipt(source, receipt)
        with pytest.raises(ValueError):
            publication.build_record(
                str(source), preset, attempt_id="native-attempt", attempts=read_stored_skyrl_attempts(str(source))
            )
        return
    row = publication.build_record(
        str(source),
        presets.PRESETS["data-medium"],
        attempt_id="native-attempt",
        attempts=read_stored_skyrl_attempts(str(source)),
    ).model_dump(mode="json")
    assert row["artifact_uri"] == str(source) and row["attempt_id"] == "native-attempt"
    assert row["iris_job_id"] == "/tests/native-run"
    assert row["run_id"] == receipt["config"]["run"]["id"] and row["series"] == name
    assert (
        row["label"]
        == f"{name} trainer.policy.optimizer_config.lr=2e-06 trainer.policy.optimizer_config.weight_decay=0.01 "
        f"tasks={inputs.tasks} init=weights@2026.11.10"
    )
    assert row["issue"] is None and row["gate_result"] is None
    assert (row["optimizer_steps"], row["elapsed_seconds"]) == (expected_updates, expected_time)
    assert row["timing_scope"] == "checkpoint_lineage_native_jobs"
    assert row["status"] == ("failed" if expected_facts else "succeeded")
    assert row["failed_facts"] == expected_facts
    if case == "missing-startup":
        assert row["wandb_url"] is None and row["engine"] is None
    if case == "native-composed":
        assert row["engine"]["vllm_version"] == "fixture-vllm"


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_snowball_publication_updates_attempts_and_publishes_controls(rl_progress_client, recorded_runs, monkeypatch):
    client, request = rl_progress_client
    preset = replace(
        presets.PRESETS["learning-medium"],
        guards=(*presets.PRESETS["learning-medium"].guards, presets.Metric("policy/entropy", True)),
    )
    monkeypatch.setattr(presets, "PRESETS", {**presets.PRESETS, preset.name: preset})
    pin = gate.control_pins(preset)[0]
    controls = [
        recorded_runs(
            preset.name,
            gate.control_name(pin.preset, pin.commit, i),
            speed=100 + i,
            score=0.49 + i * 0.01,
            attempt_id=f"pinned-{i}",
            metrics={"policy/entropy": 0.49 + i * 0.01},
        )
        for i in range(pin.runs)
    ]
    sources = [
        recorded_runs(
            preset.name, f"tests/publication-{i}", metrics={"policy/entropy": float(i)}, attempt_id=f"native-{i}"
        )
        for i in range(2)
    ]
    handles = tuple(
        ArtifactStep.adopt(f"tests/publication-{i}", preset.revision, str(root), kind=SkyRLRun)
        for i, root in enumerate(sources)
    )

    def stored():
        return {row["attempt_id"]: row for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]}

    def publish(gate_path, issue=None):
        return publication.publish_runs(
            [str(root) for root in sources],
            preset,
            gate_path=gate_path,
            issue=issue,
            service_url="http://localhost",
        )

    def saved_gate(requirements=()):
        report = run(
            gate.gate_step(preset, handles, inputs=gate.GateInputs(preset_name=preset.name, requirements=requirements))
        )[0]
        return report, str(report.path)

    report, first_gate = saved_gate()
    with monkeypatch.context() as boundary:
        boundary.setattr(
            "marina.client.marina_request", lambda *args, **kwargs: pytest.fail("Invalid record reached Marina")
        )
        rejected = publish(first_gate, issue="invalid")
        assert len(rejected.unserved_records) == 2 and all("issue" in error for error in rejected.errors)
    report_file = Path(first_gate) / RECORD_FILENAME
    original = report_file.read_bytes()
    invalid_count = json.loads(original)
    invalid_count["result"]["run"]["replicates"] = 3
    write_json(report_file, invalid_count)
    try:
        rejected = publish(first_gate)
        assert len(rejected.unserved_records) == 2 and all("current output group" in error for error in rejected.errors)
    finally:
        report_file.write_bytes(original)
    interrupted = {"once": True}

    def unavailable_readback(origin, method, path, *, json_body=None, **kwargs):
        response = request(origin, method, path, json_body=json_body)
        if method == "POST" and path.endswith("/api/runs") and interrupted["once"]:
            interrupted["once"] = False
            raise RuntimeError("Response unavailable after insert")
        return response

    monkeypatch.setattr("marina.client.marina_request", unavailable_readback)
    first = publish(first_gate, issue=73)
    assert len(first.served_records) == 1 and first.errors == ["Response unavailable after insert"]
    persisted = stored()
    assert set(persisted) == {"native-0", "native-1"} and persisted["native-0"]["issue"] == 73
    # A numeric bound is stored only as an invariant, with distinct replicate outcomes.
    relative_key = "eval_lite/aaii_category/coding/mean"
    report, second_gate = saved_gate(("policy/entropy<=0", "policy/entropy>=0", f"{relative_key}>=control"))
    assert report.verdict == "fail", report.reason
    receipt = json.loads((sources[0] / "terminal.json").read_text())
    receipt["result"]["finished_at"] = (
        datetime.fromisoformat(receipt["result"]["started_at"]) + timedelta(hours=2)
    ).isoformat()
    write_receipt(sources[0], receipt)
    monkeypatch.setattr("marina.client.marina_request", request)
    repeated = publish(second_gate, issue=74)
    assert repeated.unserved_records == [] and repeated.errors == []
    payload = client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()
    assert payload["skipped_records"] == [] and len(payload["runs"]) == 2
    rows = {row["attempt_id"]: row for row in payload["runs"]}
    assert repeated.urls[str(sources[0])].endswith(f"#run=tests%2Fpublication-0-{preset.revision}&attempt=native-0")

    def wrong_returned_key(origin, method, path, *, json_body=None, **kwargs):
        response = request(origin, method, path, json_body=json_body)
        if method == "POST" and path.endswith("/api/runs"):
            return response | {"attempt_id": "wrong-returned-attempt"}
        return response

    monkeypatch.setattr("marina.client.marina_request", wrong_returned_key)
    with pytest.raises(RuntimeError, match="different returned identity"):
        publisher.publish_result(
            RunSummary.model_validate_json(json.dumps(rows["native-0"]), strict=True), "http://localhost"
        )
    monkeypatch.setattr("marina.client.marina_request", request)
    expected = report.model_dump(mode="json", include={"verdict", "label", "reason", "invariants"}) | {
        "comparisons": [
            row.model_dump(mode="json")
            for row in report.comparisons
            if row.role in ("objective", "constraint", "requirement")
        ],
        "attempt_ids": ["native-0", "native-1"],
        "replicates": 2,
    }
    assert rows["native-0"]["gate_result"] == rows["native-1"]["gate_result"] == expected
    assert {(item["metric"], item["role"]) for item in rows["native-0"]["gate_result"]["comparisons"]} == {
        ("eval_lite/family_mean", "objective"),
        ("eval_lite/family_mean/gain_per_h100_hour", "objective"),
        ("policy/entropy", "constraint"),
        (relative_key, "requirement"),
    }
    assert rows["native-0"]["status"] == "succeeded" and rows["native-0"]["failed_facts"] == []
    assert rows["native-1"]["status"] == "failed" and rows["native-1"]["failed_facts"] == ["require/policy/entropy"]
    assert rows["native-0"]["elapsed_seconds"] == 7200 and rows["native-0"]["issue"] == 74
    assert repeated.urls[str(sources[0])].endswith("&attempt=native-0")
    windowless = deepcopy(receipt)
    windowless["result"]["finished_at"] = None
    write_receipt(sources[0], windowless)
    assert publish(second_gate).unserved_records == []
    assert stored()["native-0"]["elapsed_seconds"] is None
    assert any("Checkpoint lineage timing is unknown" in note for note in stored()["native-0"]["evidence"])
    write_receipt(sources[0], receipt)
    report_file = Path(second_gate) / RECORD_FILENAME
    original = report_file.read_bytes()
    legacy = json.loads(original)
    legacy["result"]["invariants"][0].pop("comparator")
    write_json(report_file, legacy)
    try:
        assert publish(second_gate).unserved_records == []
        assert stored()["native-0"]["gate_result"]["invariants"][0].get("comparator") is None
    finally:
        report_file.write_bytes(original)
    # A changed replicate invalidates the entire aggregate, including the unchanged replicate.
    write_receipt(sources[0], receipt)
    retry_metrics = read_skyrl_metrics(str(sources[0]))
    receipt["config"]["run"]["attempt_id"] = "native-retry"
    write_receipt(sources[0], receipt)
    for values in retry_metrics.values():
        values["attempt_id"] = "native-retry"
    append_metrics(sources[0], retry_metrics)
    stale = publish(second_gate)
    assert len(stale.unserved_records) == 2 and all("current output group" in error for error in stale.errors)
    historical = stored()["native-0"]
    assert historical["gate_result"] is None and historical["optimizer_steps"] == preset.updates
    fresh_report, fresh_gate = saved_gate()
    fresh = publish(fresh_gate)
    assert fresh.unserved_records == [] and fresh.errors == []
    assert fresh_report.run.attempt_ids == ("native-retry", "native-1")
    historical = stored()["native-0"]
    assert historical["issue"] == 74
    saved_answers = {}
    for root in sources:
        for path in (root / f"exports/dumped_evals/global_step_{preset.updates}_evals").glob("*.jsonl"):
            saved_answers[path] = path.read_bytes()
            with path.open() as stream:
                answers = [json.loads(line) for line in stream]
            for answer in answers:
                answer["verification"]["score"] = 0.0
            path.write_text("".join(json.dumps(answer) + "\n" for answer in answers))
    _, guard_gate = saved_gate(("policy/entropy>=0",))
    guard_result = publish(guard_gate)
    assert guard_result.unserved_records == [] and guard_result.errors == []
    guard_rows = [
        row
        for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]
        if row["attempt_id"] in {"native-retry", "native-1"}
    ]
    assert len(guard_rows) == 2
    assert all(
        row["gate_result"]["verdict"] == "fail" and row["status"] == "succeeded" and row["failed_facts"] == []
        for row in guard_rows
    )
    for path, data in saved_answers.items():
        path.write_bytes(data)
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
        (receipt["config"]["run"]["id"], "native-0"),
        (second_receipt["config"]["run"]["id"], "native-1"),
    }
    failed_row = stored()["native-1"]
    assert failed_row["gate_result"] is None and "native_attempt" in failed_row["failed_facts"]
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
    assert historical_row["gate_result"] is None and historical_row["optimizer_steps"] == preset.updates
    for root in controls:
        (root / gate.GATE_INPUTS_FILENAME).unlink()
    invalid_command = CliRunner().invoke(publication.main, ["--preset", "train-long"])
    assert invalid_command.exit_code == 2
    with pytest.raises(click.UsageError):
        publication.publish_controls("train-long", service_url="http://localhost")
    command = CliRunner().invoke(publication.main, ["--preset", "data-medium", "--url", "http://localhost"])
    assert command.exit_code == 0, (command.output, command.exception)
    assert json.loads(command.output)["unserved_records"] == []
    served_controls = [
        row
        for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]
        if row["attempt_id"].startswith("pinned-")
    ]
    assert len(served_controls) == pin.runs
    assert all(
        row["series"] == row["label"] == "learning-medium" and row["gate_result"] is None for row in served_controls
    )
    before_ids = {
        (row["run_id"], row["attempt_id"]) for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]
    }
    monkeypatch.setattr(presets, "PRESETS", {**presets.PRESETS, preset.name: replace(preset, revision="2026.11.12.2")})
    with pytest.raises(click.UsageError, match="control"):
        publication.publish_controls(preset.name, service_url="http://localhost")
    after_ids = {
        (row["run_id"], row["attempt_id"]) for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]
    }
    assert after_ids == before_ids
