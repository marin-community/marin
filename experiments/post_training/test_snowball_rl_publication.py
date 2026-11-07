# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click.testing import CliRunner
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, StepContext, resolve
from marin.rl.skyrl import SkyRLRun, read_skyrl_launch_config

from experiments.post_training import publish_rl_progress as publisher
from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_publication as publication
from experiments.post_training.snowball_rl_test_fixtures import native_config, write_json

pytestmark = pytest.mark.usefixtures("source_provenance", "synthetic_current_controls")


@pytest.mark.parametrize(
    "case",
    [
        "partial",
        "lineage",
        "lineage-missing-window",
        "external-resume",
        "native-failed",
        "builder-error",
        "invariant-failed",
        "train-long",
        "learning-long",
        "unknown-time",
        "missing-tracking",
        "historical",
        "wrong-cohort",
        "native-composed",
        "speed-medium",
        "data-short",
    ],
)
@pytest.mark.timeout(180)
def test_native_attempt_records_keep_identity_measurements_and_failures(recorded_runs, monkeypatch, native_json, case):
    publishing = case not in {"speed-medium", "data-short"}
    original = presets.PRESETS[
        case if case in {"learning-long", "train-long", "speed-medium", "data-short"} else "learning-medium"
    ]
    preset = replace(original, updates=2, measured=range(1, 3))
    monkeypatch.setattr(presets, "PRESETS", dict(presets.PRESETS) | {preset.name: preset})
    monkeypatch.setattr("marin.rl.skyrl.run_skyrl", lambda config: native(config))
    built = snowball.build_run(
        preset,
        settings=(() if not publishing else (presets.parse_setting("trainer.policy.optimizer_config.lr=2.0e-6"),)),
        control=not publishing,
    )
    training = built.runs[0]
    stored_preset = (
        replace(preset, evaluations=(replace(preset.evaluations[0], answers=2), *preset.evaluations[1:]))
        if case == "partial"
        else preset
    )
    source = recorded_runs(
        stored_preset,
        training.name,
        attempt_id="native-attempt",
        initial_score=0.3,
        score=0.5,
        fingerprint=training.fingerprint(),
        unscored=15 if case == "partial" else 0,
        argv=["--preset", preset.name, "--set=+trainer.policy.optimizer_config={lr: 2.0e-6}"],
    )
    tracking_path = source / "exports/attempts/native-attempt/tracking.json"
    tracking = json.loads(tracking_path.read_text())
    tracking.update(
        wandb_url="https://wandb.ai/test/record-builder/runs/native-attempt", engine={"model_dtype": "bfloat16"}
    )
    write_json(tracking_path, tracking)
    launch = read_skyrl_launch_config(str(source))
    attempts = source.parent / "separate-attempt-receipts" / case
    if case == "native-composed":
        context = StepContext.for_run(
            str(source), str(source.parent), runtime_args=training.runtime_args, deps=training.deps
        )
        raw = yaml.safe_load(training.build_config(context).launch_config_yaml)
        raw["run"]["attempt_id"] = "native-attempt"
        raw["runtime"]["task_env"]["RECORD_BUILDER_ENV"] = "native-composition"
        raw["artifacts"]["attempts_root"] = str(attempts)
        path = source / "native-source-launch.yaml"
        path.write_text(yaml.safe_dump(raw))
        launch = native_json(
            """
import json, sys
from pathlib import Path
from cloud.iris.launch_config import load_launch_config
from omegaconf import OmegaConf
print(json.dumps(OmegaConf.to_container(load_launch_config(Path(sys.argv[1])), resolve=True)))
""",
            [str(path)],
        )
    else:
        launch["artifacts"] = {"attempts_root": str(attempts)}
    receipt = json.loads((source / "terminal.json").read_text())
    receipt["config"] = launch
    write_json(source / "terminal.json", receipt)
    (source / "resolved-launch.yaml").write_text(yaml.safe_dump({"config": launch}))
    if case in {"lineage", "lineage-missing-window"}:
        ancestor = json.loads(json.dumps(receipt))
        ancestor["config"]["run"]["attempt_id"] = "ancestor"
        ancestor["result"].update(
            state="failed",
            started_at="2026-11-06T16:40:00+00:00",
            finished_at=None if case == "lineage-missing-window" else "2026-11-06T17:40:00+00:00",
        )
        write_json(attempts / "ancestor.json", ancestor)
        for step in (0, 1):
            path = source / f"exports/metrics/step-{step:06d}.json"
            row = json.loads(path.read_text())
            row["attempt_id"] = "ancestor"
            write_json(path, row)
    if case == "external-resume":
        receipt["result"]["provenance"]["command_line"].extend(["--resume-from", "source-run@2026.11.01"])
        write_json(source / "terminal.json", receipt)
        initial = source / "exports/metrics/step-000000.json"
        first = source / "exports/metrics/step-000001.json"
        if case == "external-resume":
            baseline = json.loads(initial.read_text())
            baseline["step"] = 1
            write_json(first, baseline)
            (source / "exports/dumped_evals/global_step_0_evals").rename(
                source / "exports/dumped_evals/global_step_1_evals"
            )
        initial.unlink()
    config = native_config(source, launch)

    def native(edge_config):
        if case == "unknown-time":
            receipt = json.loads((source / "terminal.json").read_text())
            receipt["result"]["started_at"] = None
            write_json(source / "terminal.json", receipt)
        if case == "missing-tracking":
            (source / "exports/attempts/native-attempt/tracking.json").unlink()
        if case == "historical":
            receipt = json.loads((source / "terminal.json").read_text())
            write_json(attempts / "native-attempt.json", receipt)
            (source / "terminal.json").unlink()
            current = json.loads(json.dumps(launch))
            current["run"]["attempt_id"] = "newer-attempt"
            (source / "resolved-launch.yaml").write_text(yaml.safe_dump({"config": current}))
        if case == "builder-error":
            descriptor = launch["inputs"]["validation_data"][0]
            pq.write_table(
                pa.Table.from_pylist([{"task_spec": "invalid-json"}]),
                Path(descriptor["uri"]) / descriptor["relative_path"],
            )
        if case == "wrong-cohort":
            descriptor = launch["inputs"]["validation_data"][0]
            path = Path(descriptor["uri"]) / descriptor["relative_path"]
            rows = pq.read_table(path).to_pylist()
            task = json.loads(rows[0]["task_spec"])
            task["id"] = "different-task"
            rows[0]["task_spec"] = json.dumps(task)
            pq.write_table(pa.Table.from_pylist(rows), path)
        if case == "native-failed":
            shutil.rmtree(source / "exports")
            receipt = json.loads((source / "terminal.json").read_text())
            receipt["result"]["state"] = "failed"
            write_json(source / "terminal.json", receipt)
            write_json(attempts / "native-attempt.json", receipt)
            raise RuntimeError("native failed")
        if case == "invariant-failed":
            step = 1
            path = source / f"exports/metrics/step-{step:06d}.json"
            row = json.loads(path.read_text())
            row["policy/raw_grad_norm"] = float("nan")
            write_json(path, row)
        return SkyRLRun.raw_load(str(source))

    if case == "native-failed":
        with pytest.raises(RuntimeError, match="native failed"):
            training.run(config)
    else:
        result = training.run(config)
        assert result.path == str(source)
    assert list(source.rglob(publication.RESULT_FILENAME)) == []
    if case == "builder-error":
        with pytest.raises(json.JSONDecodeError):
            publication.retain_record(str(source), preset)
        return
    if not publishing:
        return
    row = publication.retain_record(str(source), preset, attempt_id="native-attempt")
    path = Path(row["artifact_uri"])
    assert (row["run_id"], row["attempt_id"]) == (launch["run"]["id"], "native-attempt")
    assert row["label"] == preset.name + ("" if case == "control" else " lr=2e-06")
    assert row["issue"] is None and row["verdict"] is None
    assert row["evaluation"]["task_count"] == (100 if case in {"learning-long", "train-long"} else 1500)
    assert row["elapsed_seconds"] == (
        None if case in {"unknown-time", "lineage-missing-window"} else 7200 if case == "lineage" else 3600
    )
    assert row["wandb_url"] == (None if case in {"missing-tracking", "native-failed"} else tracking["wandb_url"])
    assert row["engine"] == (None if case in {"missing-tracking", "native-failed"} else tracking["engine"])
    if case in {"native-failed", "invariant-failed"}:
        assert row["status"] == "failed"
        assert ("native_attempt" if case == "native-failed" else "finite_losses_and_gradients") in row["failed_facts"]
    else:
        assert row["status"] == "succeeded" and row["failed_facts"] == []
    if case in {"native-failed", "wrong-cohort", "external-resume"}:
        assert row["baseline"] is None and row["final"] is None
    else:
        assert row["baseline"]["reward"] == pytest.approx(0.3)
        assert row["final"]["reward"] == pytest.approx(0.5)
        assert row["optimizer_steps"] == 2
    if case.startswith("external-"):
        assert row["resume_from"] == "source-run@2026.11.01" and row["baseline_step"] is None
        assert row["planned_updates_valid"] is None
        assert row["optimizer_steps"] == 1
    if case == "lineage":
        old = publication.build_record(str(source), preset, attempt_id="ancestor")
        assert old["final"] is None and old["baseline"]["reward"] == pytest.approx(0.3)
        assert old["optimizer_steps"] == 1 and old["elapsed_seconds"] == 3600
    if case == "partial":
        assert row["final"]["attempted"] == 1500 and row["final"]["scored"] == 1485
        saved = path.read_bytes()
        receipt = json.loads((source / "terminal.json").read_text())
        receipt["result"]["finished_at"] = (
            datetime.fromisoformat(receipt["result"]["started_at"]) + timedelta(hours=2)
        ).isoformat()
        write_json(source / "terminal.json", receipt)

        def forbid_live(*args, **kwargs):
            raise AssertionError("Host rebuilding requested live execution or state")

        monkeypatch.setattr("marin.rl.skyrl.run_skyrl", forbid_live)
        monkeypatch.setattr("marin.rl.skyrl.read_skyrl_launch_receipt", forbid_live)
        monkeypatch.setattr("rigging.provenance.Provenance.capture", forbid_live)
        monkeypatch.setattr("marin.rl.cli.iris_client", forbid_live)
        rebuilt = publication.build_record(str(source), preset)
        assert rebuilt["elapsed_seconds"] == 7200 and rebuilt["final"] == row["final"]
        assert rebuilt["label"] == row["label"]
        assert rebuilt["source_revision"] == row["source_revision"] and path.read_bytes() == saved
    assert "vllm_version" not in row["evaluation"]["protocol"]
    if case == "native-composed":
        assert row["evaluation"]["protocol"]["task_env"]["RECORD_BUILDER_ENV"] == "native-composition"
        assert row["evaluation"]["protocol"]["context_budget"] == {"max_new_tokens_per_turn": 8192, "max_turns": 1}
    if case == "native-failed":
        assert row["optimizer_steps"] == 0


@pytest.mark.integration
def test_snowball_publication_retries_keep_the_first_gate(rl_progress_client, recorded_runs, monkeypatch, tmp_path):
    client, request = rl_progress_client
    preset = replace(presets.PRESETS["learning-medium"], updates=2, measured=range(1, 3))
    source = recorded_runs(
        preset, "tests/snowball-publication", initial_score=0.3, score=0.5, unscored=15, attempt_id="native-one"
    )
    launch = read_skyrl_launch_config(str(source))
    tracking_path = source / "exports/attempts/native-one/tracking.json"
    tracking = json.loads(tracking_path.read_text())
    tracking.update(wandb_url="https://wandb.ai/test/publication/runs/native-one", engine={"model_dtype": "bfloat16"})
    write_json(tracking_path, tracking)
    raw = publication.retain_record(str(source), preset)
    raw_path = Path(raw["artifact_uri"])

    def forbid_native(config):
        raise AssertionError("Publication launched native execution")

    training = ArtifactStep(
        name="tests/snowball-publication",
        version=presets.PRESET_REVISION,
        artifact_type=SkyRLRun,
        build_config=lambda context: {},
        run=forbid_native,
        override_path=str(source),
    )

    def saved_gate(name, verdict):
        path = tmp_path / name
        report = gate.GateReport(
            path=str(path),
            preset=preset.name,
            verdict=verdict,
            label=None,
            reason=None,
            run=gate.RunMetadata(settings=(), tasks=None, init=None, replicates=1, requirements=()),
            control=gate.ControlMetadata(main_commit="source", revision=presets.PRESET_REVISION, runs=4),
            invariants=(),
            guards=(),
            reported=(),
            gaps=(),
        )
        record = ArtifactRecord(
            name=name,
            version=presets.PRESET_REVISION,
            fingerprint=name,
            result_type=result_type_name(gate.GateReport),
            result=report.result_payload(),
        )
        write_json(path / RECORD_FILENAME, record.model_dump(mode="json"))
        write_json(path / gate.GATE_REPORT_FILENAME, report.model_dump(mode="json"))
        return str(path)

    first_gate = saved_gate("first-gate", "pass")
    second_gate = saved_gate("current-gate", "fail")
    interrupted = {"once": True}

    def unavailable_readback(origin, method, path, *, json_body=None, **kwargs):
        if method == "POST" and json_body["sql"] == publisher.READ and interrupted["once"]:
            interrupted["once"] = False
            raise RuntimeError("Readback unavailable after insert")
        return request(origin, method, path, json_body=json_body)

    monkeypatch.setattr("marina.client.marina_request", unavailable_readback)
    step = publication.publication_step(training, json.loads(raw_path.read_text()), gate_path=first_gate, issue=73)
    with pytest.raises(RuntimeError):
        resolve(step)
    frozen = Path(step.path()) / publication.RESULT_FILENAME
    first_bytes = frozen.read_bytes()
    raw = json.loads(raw_path.read_text())
    raw["elapsed_seconds"] = 7200
    write_json(raw_path, raw)
    monkeypatch.setattr("marina.client.marina_request", request)
    retried = publication.publication_step(training, json.loads(raw_path.read_text()), gate_path=second_gate, issue=74)
    resolve(retried)
    assert frozen.read_bytes() == first_bytes
    receipt = json.loads((Path(retried.path()) / publication.PUBLICATION_FILENAME).read_text())
    assert receipt["verified"] and receipt["inserted"] is False
    payload = client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()
    assert payload["skipped_records"] == [] and len(payload["runs"]) == 1
    served = payload["runs"][0]
    assert (served["run_id"], served["attempt_id"]) == (launch["run"]["id"], "native-one")
    assert served["gate"] == "first-gate" and served["verdict"] == "pass" and (served["issue"] == 73)
    assert served["elapsed_seconds"] == 3600 and served["evaluation"]["task_count"] == 1500
    assert served["final"]["scored"] == 1485 and served["pareto_eligible"] and served["trusted_gain_eligible"]
    assert served["wandb_url"] == tracking["wandb_url"] and served["engine"] == tracking["engine"]
    for change in ("engine", "request", "content", "membership"):
        variant = tmp_path / f"stored-{change}"
        shutil.copytree(source, variant)
        receipt = json.loads((variant / "terminal.json").read_text())
        recorded = receipt["config"]
        recorded["run"]["id"] = f"tests/{change}"
        variant_tracking = dict(tracking, run_id=recorded["run"]["id"])
        if change == "engine":
            recorded["skyrl"]["generator"]["model_dtype"] = "float16"
            variant_tracking["engine"] = {"model_dtype": "float16"}
        elif change == "request":
            recorded["skyrl"]["generator"]["eval_sampling_params"]["temperature"] = 0.1
        else:
            descriptor = recorded["inputs"]["validation_data"][0]
            selection = tmp_path / f"selection-{change}"
            shutil.copytree(descriptor["uri"], selection)
            descriptor["uri"] = str(selection)
            task_path = selection / descriptor["relative_path"]
            tasks = pq.read_table(task_path).to_pylist()
            if change == "content":
                tasks[0]["task_spec"] += " "
            else:
                task = json.loads(tasks[0]["task_spec"])
                previous = task["id"]
                task["id"] = "new-member"
                tasks[0]["task_spec"] = json.dumps(task)
                manifest_path = selection / presets.TASK_MANIFEST_FILENAME
                manifest = pq.read_table(manifest_path).to_pylist()
                for row in manifest:
                    if row["task_id"] == previous:
                        row["task_id"] = "new-member"
                pq.write_table(pa.Table.from_pylist(manifest), manifest_path)
                for answer_path in (variant / "exports/dumped_evals").glob("*/panel.jsonl"):
                    answers = [json.loads(line) for line in answer_path.read_text().splitlines()]
                    for row in answers:
                        if row["task_id"] == previous:
                            row["task_id"] = "new-member"
                    answer_path.write_text("".join(json.dumps(row) + "\n" for row in answers))
            pq.write_table(pa.Table.from_pylist(tasks), task_path)
        write_json(variant / "terminal.json", receipt)
        write_json(variant / "resolved-launch.yaml", {"config": recorded})
        write_json(variant / "exports/attempts/native-one/tracking.json", variant_tracking)
        produced = publication.build_record(str(variant), preset)
        publisher.publish_result(produced, "http://localhost")
    variants = {row["run_id"]: row for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]}
    assert variants["tests/engine"]["comparison_key"] == served["comparison_key"]
    assert variants["tests/engine"]["engine"]["model_dtype"] == "float16"
    for change in ("request", "content", "membership"):
        assert variants[f"tests/{change}"]["comparison_key"] != served["comparison_key"]
    assert variants["tests/content"]["evaluation"]["task_set_sha256"] != served["evaluation"]["task_set_sha256"]
    assert variants["tests/membership"]["evaluation"]["membership_sha256"] != served["evaluation"]["membership_sha256"]

    monkeypatch.setattr(presets, "PRESETS", dict(presets.PRESETS) | {preset.name: preset})
    pin = gate.ControlPin("learning-medium", "pinned-source-abbrev", presets.PRESET_REVISION, 2)
    monkeypatch.setattr(gate, "CONTROLS", {"learning-medium": (pin,)})
    controls = [
        recorded_runs(
            preset,
            presets.control_name("learning-medium", "pinned-source-abbrev", index),
            initial_score=0.3,
            score=0.5,
            attempt_id=f"pinned-{index}",
        )
        for index in range(2)
    ]
    command = CliRunner().invoke(publisher.main, ["--preset", "data-medium", "--controls", "--url", "http://localhost"])
    assert command.exit_code == 0, (command.output, command.exception)
    published = json.loads(command.output)
    assert published["unserved_records"] == []
    assert sorted(published["served_records"]) == sorted(
        str(root / f"exports/attempts/pinned-{index}/{publication.RESULT_FILENAME}")
        for index, root in enumerate(controls)
    )
    pairs = [
        (row["run_id"], row["attempt_id"])
        for row in client.get(f"/a/{publisher.APPLET_ID}/api/runs").json()["runs"]
        if row["attempt_id"].startswith("pinned-")
    ]
    assert sorted(pairs) == sorted(
        (read_skyrl_launch_config(str(root))["run"]["id"], f"pinned-{index}") for index, root in enumerate(controls)
    )
