# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta

import click
import pytest
import yaml
from click.testing import CliRunner
from marin.execution.artifact import RECORD_FILENAME
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import publish_rl_progress as publisher
from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_publication as publication
from experiments.post_training.snowball_tests.fixtures import write_json, write_receipt

pytestmark = pytest.mark.usefixtures("source_provenance")


@pytest.mark.parametrize(
    "case",
    [
        "native-composed",
        "native-failed",
        "invariant-failed",
        "legacy-lineage",
        "lineage-missing-window",
        "historical-unknown",
        "current-lineage-unknown",
        "missing-tracking",
        "train-long",
    ],
)
@pytest.mark.timeout(180)
def test_native_attempt_records_keep_identity_measurements_and_failures(recorded_runs, monkeypatch, native_json, case):
    name = "train-long" if case == "train-long" else "learning-medium"
    preset = replace(presets.PRESETS[name], updates=2, measured=range(1, 3))
    monkeypatch.setattr(presets, "PRESETS", dict(presets.PRESETS) | {name: preset})
    inputs = gate.GateInputs(
        preset_name=name,
        settings=("trainer.policy.optimizer_config.lr=2e-06", "trainer.ref.optimizer_config.lr=0.01"),
        tasks=f"snowball-100k@{presets.DEFAULT_SELECTION_VERSION}",
        init="weights@pinned",
    )
    source = recorded_runs(name, "tests/stored-progress", attempt_id="native-attempt", inputs=inputs)
    receipt = json.loads((source / "terminal.json").read_text())
    if case == "native-composed":
        training = snowball.build_run(
            preset, settings=(presets.parse_setting("trainer.policy.optimizer_config.lr=2.0e-6"),)
        ).runs[0]
        context = StepContext.for_run(
            str(source), str(source.parent), runtime_args=training.runtime_args, deps=training.deps
        )
        raw = yaml.safe_load(training.build_config(context).launch_config_yaml)
        raw["run"]["attempt_id"] = "native-attempt"
        raw["artifacts"]["export_root"] = str(source / "exports")
        path = source / "native-source-launch.yaml"
        path.write_text(yaml.safe_dump(raw))
        receipt["config"] = native_json(
            """
import json, sys
from pathlib import Path
from cloud.iris.launch_config import load_launch_config
from omegaconf import OmegaConf
print(json.dumps(OmegaConf.to_container(load_launch_config(Path(sys.argv[1])), resolve=True)))
""",
            [str(path)],
        )
    if case in {"legacy-lineage", "lineage-missing-window", "historical-unknown", "current-lineage-unknown"}:
        ancestor = deepcopy(receipt)
        ancestor["config"]["run"]["attempt_id"] = "ancestor"
        ancestor["result"].update(
            state="failed",
            started_at="2026-11-06T16:40:00+00:00",
            finished_at=None if case == "lineage-missing-window" else "2026-11-06T17:40:00+00:00",
        )
        write_json(source / "exports/attempts/ancestor/receipt.json", ancestor)
        if case in {"legacy-lineage", "lineage-missing-window"}:
            for path in (source / "exports/metrics").glob("step-*.json"):
                values = json.loads(path.read_text())
                values["attempt_id"] = "ancestor" if values["step"] < 2 else "native-attempt"
                write_json(path, values)
    if case == "native-failed":
        receipt["result"]["state"] = "failed"
        shutil.rmtree(source / "exports/metrics")
    if case == "invariant-failed":
        path = source / "exports/metrics/step-000001.json"
        write_json(path, json.loads(path.read_text()) | {"policy/raw_grad_norm": float("nan")})
    if case == "missing-tracking":
        (source / "exports/attempts/native-attempt/tracking.json").unlink()
    if case == "native-failed":
        (source / "terminal.json").unlink()
        write_json(source / "exports/attempts/native-attempt/receipt.json", receipt)
        write_json(source / "resolved-launch.yaml", {"config": receipt["config"]})
    else:
        write_receipt(source, receipt)
    if case == "native-composed":
        shutil.rmtree(source / "exports/dumped_evals")

        def forbid_live(*args, **kwargs):
            pytest.fail("Host record builder requested live native execution/state")

        monkeypatch.setattr("marin.rl.skyrl.run_skyrl", forbid_live)
        monkeypatch.setattr("marin.rl.skyrl.read_skyrl_launch_receipt", forbid_live)
    selected = "ancestor" if case == "historical-unknown" else "native-attempt"
    row = publication.build_record(str(source), presets.PRESETS["data-medium"], attempt_id=selected)
    assert row["artifact_uri"] == str(source) and row["attempt_id"] == selected
    assert row["run_id"] == receipt["config"]["run"]["id"]
    assert row["series"] == name
    assert (
        row["label"] == f"{name} trainer.policy.optimizer_config.lr=2e-06 trainer.ref.optimizer_config.lr=0.01 "
        f"tasks={inputs.tasks} init=weights@pinned"
    )
    assert row["issue"] is None and row["verdict"] is None and row["gate_result"] is None
    assert row["evaluation"] is None and row["baseline"] is None and row["final"] is None
    expected_time = (
        None
        if case in {"lineage-missing-window", "historical-unknown", "current-lineage-unknown"}
        else 7200 if case == "legacy-lineage" else 3600
    )
    assert row["elapsed_seconds"] == expected_time and row["timing_scope"] == "checkpoint_lineage_native_jobs"
    assert row["optimizer_steps"] == (None if case == "historical-unknown" else 0 if case == "native-failed" else 2)
    if case in {"native-failed", "invariant-failed", "historical-unknown"}:
        assert row["status"] == "failed"
        assert ("finite_losses_and_gradients" if case == "invariant-failed" else "native_attempt") in row["failed_facts"]
    else:
        assert row["status"] == "succeeded" and row["failed_facts"] == []
    if case == "legacy-lineage":
        historical = publication.build_record(str(source), preset, attempt_id="ancestor")
        assert historical["optimizer_steps"] == 1 and historical["elapsed_seconds"] == 3600
    if case == "missing-tracking":
        assert row["wandb_url"] is None and row["engine"] is None
    if case == "native-composed":
        assert row["engine"]["vllm_version"] == "fixture-vllm"


@pytest.mark.integration
@pytest.mark.timeout(180)
def test_snowball_publication_updates_attempts_and_publishes_controls(rl_progress_client, recorded_runs, monkeypatch):
    client, request = rl_progress_client
    preset = presets.PRESETS["learning-medium"]
    pin = gate.control_pins(preset.name)[0]
    controls = [
        recorded_runs(
            preset.name,
            presets.control_name(pin.preset, pin.commit, i),
            speed=100 + i,
            score=0.49 + i * 0.01,
            attempt_id=f"pinned-{i}",
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
        ArtifactStep.adopt(f"tests/publication-{i}", presets.PRESET_REVISION, str(root), kind=SkyRLRun)
        for i, root in enumerate(sources)
    )

    def saved_gate(requirements=()):
        report = run(
            gate.gate_step(preset, handles, inputs=gate.GateInputs(preset_name=preset.name, requirements=requirements))
        )[0]
        return report, str(report.path)

    report, first_gate = saved_gate()
    assert report.comparisons and report.invariants and report.run.attempt_ids == ("native-0", "native-1")
    interrupted = {"once": True}

    def unavailable_readback(origin, method, path, *, json_body=None, **kwargs):
        response = request(origin, method, path, json_body=json_body)
        if method == "POST" and path.endswith("/api/runs") and interrupted["once"]:
            interrupted["once"] = False
            raise RuntimeError("Response unavailable after insert")
        return response

    monkeypatch.setattr("marina.client.marina_request", unavailable_readback)
    first = publication.publish_runs(
        [str(root) for root in sources],
        preset,
        gate_path=first_gate,
        gate_required=True,
        issue=73,
        service_url="http://localhost",
    )
    assert len(first.served_records) == 1 and first.errors == ["Response unavailable after insert"]
    persisted = {row["attempt_id"]: row for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]}
    assert set(persisted) == {"native-0", "native-1"} and persisted["native-0"]["issue"] == 73
    # A numeric bound is stored only as an invariant, with distinct replicate outcomes.
    relative_key = "test_panel/aaii_category/coding/mean"
    report, second_gate = saved_gate(("policy/entropy<=0", f"{relative_key}>=control"))
    assert [
        (row.metric, row.run, row.control) for row in report.comparisons if row.role == gate.ComparisonRole.REQUIREMENT
    ] == [(relative_key, pytest.approx(0.5), pytest.approx(0.505))]
    assert report.verdict == "fail" and all(row.passes is not False for row in report.comparisons)
    assert [(fact.replicate, fact.holds) for fact in report.invariants if fact.name == "require/policy/entropy"] == [
        (0, True),
        (1, False),
    ]
    receipt = json.loads((sources[0] / "terminal.json").read_text())
    receipt["result"]["finished_at"] = (
        datetime.fromisoformat(receipt["result"]["started_at"]) + timedelta(hours=2)
    ).isoformat()
    write_receipt(sources[0], receipt)
    monkeypatch.setattr("marina.client.marina_request", request)
    repeated = publication.publish_runs(
        [str(root) for root in sources],
        preset,
        gate_path=second_gate,
        gate_required=True,
        issue=74,
        service_url="http://localhost",
    )
    assert repeated.unserved_records == [] and repeated.errors == []
    payload = client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()
    assert payload["skipped_records"] == [] and len(payload["runs"]) == 2
    rows = {row["attempt_id"]: row for row in payload["runs"]}
    assert repeated.urls[str(sources[0])].endswith(
        f"#run=tests%2Fpublication-0-{presets.PRESET_REVISION}&attempt=native-0"
    )

    def wrong_returned_key(origin, method, path, *, json_body=None, **kwargs):
        response = request(origin, method, path, json_body=json_body)
        if method == "POST" and path.endswith("/api/runs"):
            return response | {"attempt_id": "wrong-returned-attempt"}
        return response

    monkeypatch.setattr("marina.client.marina_request", wrong_returned_key)
    with pytest.raises(RuntimeError, match="different returned identity"):
        publisher.publish_result(rows["native-0"], "http://localhost")
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
    assert rows["native-0"]["status"] == "succeeded" and rows["native-0"]["failed_facts"] == []
    assert rows["native-1"]["status"] == "failed" and rows["native-1"]["failed_facts"] == ["require/policy/entropy"]
    assert rows["native-0"]["elapsed_seconds"] == 7200 and rows["native-0"]["issue"] == 74
    assert repeated.urls[str(sources[0])].endswith("&attempt=native-0")
    # A changed replicate invalidates the entire aggregate, including the unchanged replicate.
    write_json(sources[0] / "exports/attempts/native-0/receipt.json", receipt)
    receipt["config"]["run"]["attempt_id"] = "native-retry"
    write_receipt(sources[0], receipt)
    stale = publication.publish_runs(
        [str(root) for root in sources],
        preset,
        gate_path=second_gate,
        gate_required=True,
        service_url="http://localhost",
    )
    assert len(stale.unserved_records) == 2 and all("current output group" in error for error in stale.errors)
    historical = next(
        row for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"] if row["attempt_id"] == "native-0"
    )
    assert historical["gate_result"] is None and historical["verdict"] is None and historical["optimizer_steps"] is None
    fresh_report, fresh_gate = saved_gate()
    fresh = publication.publish_runs(
        [str(root) for root in sources], preset, gate_path=fresh_gate, gate_required=True, service_url="http://localhost"
    )
    assert fresh.unserved_records == [] and fresh.errors == []
    assert fresh_report.run.attempt_ids == ("native-retry", "native-1")
    historical = next(
        row for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"] if row["attempt_id"] == "native-0"
    )
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
    guard_report, guard_gate = saved_gate(("policy/entropy>=0",))
    assert guard_report.verdict == "fail" and any(row.passes is False for row in guard_report.comparisons)
    assert all(fact.holds for fact in guard_report.invariants)
    guard_result = publication.publish_runs(
        [str(root) for root in sources],
        preset,
        gate_path=guard_gate,
        gate_required=True,
        service_url="http://localhost",
    )
    assert guard_result.unserved_records == [] and guard_result.errors == []
    guard_rows = [
        row
        for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]
        if row["attempt_id"] in {"native-retry", "native-1"}
    ]
    assert len(guard_rows) == 2
    assert all(
        row["verdict"] == "fail" and row["status"] == "succeeded" and row["failed_facts"] == [] for row in guard_rows
    )
    for path, data in saved_answers.items():
        path.write_bytes(data)
    second_receipt = json.loads((sources[1] / "terminal.json").read_text())
    failed_second = deepcopy(second_receipt)
    failed_second["result"]["state"] = "failed"
    failed_receipt_path = sources[1] / "exports/attempts/native-1/receipt.json"
    write_json(failed_receipt_path, failed_second)
    (sources[1] / "terminal.json").unlink()
    stored_exists = StoragePath.exists

    def gate_marker_unavailable(path):
        if str(path) == str(StoragePath(fresh_gate) / RECORD_FILENAME):
            raise OSError("Gate storage unavailable")
        return stored_exists(path)

    with monkeypatch.context() as storage_fault:
        storage_fault.setattr(StoragePath, "exists", gate_marker_unavailable)
        blocked = publication.publish_runs(
            [str(root) for root in sources],
            preset,
            gate_path=fresh_gate,
            gate_required=True,
            service_url="http://localhost",
        )
    assert blocked.unserved_records == [str(sources[0] / "exports/attempts/native-retry")]
    assert blocked.errors == ["Gate storage unavailable"]
    assert set(blocked.served_records) == {
        str(sources[0] / "exports/attempts/native-0"),
        str(sources[1] / "exports/attempts/native-1"),
    }
    failed_row = next(
        row for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"] if row["attempt_id"] == "native-1"
    )
    assert failed_row["gate_result"] is None and "native_attempt" in failed_row["failed_facts"]
    failed_receipt_path.unlink()
    write_receipt(sources[1], second_receipt)
    # Missing provenance produces an attributable INVALID through the actual gate.
    receipt["result"]["provenance"] = None
    write_receipt(sources[0], receipt)
    invalid_report, invalid_gate = saved_gate(("policy/entropy<=2",))
    assert invalid_report.verdict == "invalid" and invalid_report.run.attempt_ids == ("native-retry", "native-1")
    invalid = publication.publish_runs(
        [str(root) for root in sources],
        preset,
        gate_path=invalid_gate,
        gate_required=True,
        service_url="http://localhost",
    )
    assert invalid.unserved_records == []
    invalid_row = next(
        row
        for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]
        if row["attempt_id"] == "native-retry"
    )
    assert invalid_row["gate_result"]["verdict"] == "invalid" and invalid_row["gate_result"]["reason"]
    (sources[1] / "terminal.json").unlink()
    missing_report, missing_gate = saved_gate()
    assert missing_report.verdict == "invalid" and missing_report.run.attempt_ids == ("native-retry",)
    missing = publication.publish_runs(
        [str(root) for root in sources],
        preset,
        gate_path=missing_gate,
        gate_required=True,
        service_url="http://localhost",
    )
    assert str(sources[1]) in missing.unserved_records and any(
        "No stored native receipt" in error for error in missing.errors
    )
    write_json(sources[1] / "exports/attempts/native-1/receipt.json", second_receipt)
    missing_current = deepcopy(second_receipt["config"])
    missing_current["run"]["attempt_id"] = "missing-latest"
    write_json(sources[1] / "resolved-launch.yaml", {"config": missing_current})
    recovered_history = publication.publish_runs(
        [str(root) for root in sources],
        preset,
        gate_path=missing_gate,
        gate_required=True,
        service_url="http://localhost",
    )
    assert recovered_history.unserved_records == [str(sources[1])]
    assert len(recovered_history.errors) == 1 and "missing-latest" in recovered_history.errors[0]
    assert str(sources[1] / "exports/attempts/native-1") in recovered_history.served_records
    historical_row = next(
        row for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"] if row["attempt_id"] == "native-1"
    )
    assert historical_row["gate_result"] is None and historical_row["optimizer_steps"] is None
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
