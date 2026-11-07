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
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, StepContext, resolve
from marin.rl.rl_progress import RunSummary
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
        "native-failed",
        "invariant-failed",
        "stale-initial",
        "control",
        "train-long",
        "unknown-time",
        "missing-start",
        "missing-finish",
        "missing-tracking",
        "historical",
        "wrong-cohort",
        "wrong-final-cohort",
        "native-composed",
        "speed-medium",
        "data-short",
    ],
)
@pytest.mark.timeout(180)
def test_native_attempt_records_keep_identity_measurements_and_failures(recorded_runs, monkeypatch, native_json, case):
    publishing = case not in {"speed-medium", "data-short"}
    original = presets.PRESETS[case if case in {"train-long", "speed-medium", "data-short"} else "learning-medium"]
    preset = replace(original, updates=2, measured=range(1, 3))
    monkeypatch.setattr(presets, "PRESETS", dict(presets.PRESETS) | {preset.name: preset})
    monkeypatch.setattr("marin.rl.skyrl.run_skyrl", lambda config: native(config))
    built = snowball.build_run(
        preset,
        settings=(
            ()
            if case == "control" or not publishing
            else (presets.parse_setting("trainer.policy.optimizer_config.lr=2.0e-6"),)
        ),
        control=case == "control" or not publishing,
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
    config = native_config(source, launch)

    def native(edge_config):
        if case in {"unknown-time", "missing-start", "missing-finish"}:
            receipt = json.loads((source / "terminal.json").read_text())
            for key in ("started_at", "finished_at"):
                if case == "unknown-time" or key == ("started_at" if case == "missing-start" else "finished_at"):
                    receipt["result"][key] = None
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
        if case in {"wrong-cohort", "wrong-final-cohort"}:
            descriptor = launch["inputs"]["validation_data"][0]
            selection = Path(descriptor["uri"])
            if case == "wrong-cohort":
                path = selection / descriptor["relative_path"]
                rows = pq.read_table(path).to_pylist()
                task = json.loads(rows[0]["task_spec"])
                task["id"] = "different-task"
                rows[0]["task_spec"] = json.dumps(task)
                pq.write_table(pa.Table.from_pylist(rows), path)
            else:
                manifest_path = selection / presets.TASK_MANIFEST_FILENAME
                manifest = pq.read_table(manifest_path).to_pylist()
                manifest.append(dict(manifest[0], task_id="different-task"))
                pq.write_table(pa.Table.from_pylist(manifest), manifest_path)
                path = source / "exports/dumped_evals/global_step_2_evals/panel.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows[0]["task_id"] = "different-task"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if case == "native-failed":
            shutil.rmtree(source / "exports")
            receipt = json.loads((source / "terminal.json").read_text())
            receipt["result"]["state"] = "failed"
            write_json(source / "terminal.json", receipt)
            write_json(attempts / "native-attempt.json", receipt)
            raise RuntimeError("native failed")
        if case in {"invariant-failed", "stale-initial"}:
            step = 1 if case == "invariant-failed" else 0
            path = source / f"exports/metrics/step-{step:06d}.json"
            row = json.loads(path.read_text())
            row.update({"policy/raw_grad_norm": float("nan")} if step else {"attempt_id": "old-attempt"})
            write_json(path, row)
        return SkyRLRun.raw_load(str(source))

    monkeypatch.setattr(publication, "run_skyrl", native)
    if case == "native-failed":
        with pytest.raises(RuntimeError, match="native failed"):
            training.run(config)
    else:
        training.run(config)
    if not publishing:
        assert list(source.rglob("loop-result.json")) == []
        return
    path = source / "exports/attempts/native-attempt/loop-result.json"
    row = json.loads(path.read_text())
    RunSummary.model_validate(row)
    assert (row["run_id"], row["attempt_id"]) == (launch["run"]["id"], "native-attempt")
    assert row["label"] == preset.name + ("" if case == "control" else " lr=2e-06")
    assert row["issue"] is None and row["verdict"] is None
    assert row["min_scored_fraction"] == 0.99
    assert row["evaluation"]["task_count"] == (100 if case == "train-long" else 1500)
    assert row["elapsed_seconds"] == (None if case in {"unknown-time", "missing-start", "missing-finish"} else 3600)
    assert row["timing_scope"] == "native_job_start_to_finish"
    assert row["wandb_url"] == (None if case in {"missing-tracking", "native-failed"} else tracking["wandb_url"])
    assert row["engine"] == (None if case in {"missing-tracking", "native-failed"} else tracking["engine"])
    if case in {"native-failed", "invariant-failed"}:
        assert row["status"] == "failed"
        assert ("native_attempt" if case == "native-failed" else "finite_losses_and_gradients") in row["failed_facts"]
    else:
        assert row["status"] == "succeeded" and row["failed_facts"] == []
    if case in {"native-failed", "stale-initial", "wrong-cohort"}:
        assert row["baseline"] is None and row["final"] is None
    elif case == "wrong-final-cohort":
        assert row["baseline"]["reward"] == pytest.approx(0.3) and row["final"] is None
    else:
        assert row["baseline"]["reward"] == pytest.approx(0.3)
        assert row["final"]["reward"] == pytest.approx(0.5)
        assert row["optimizer_steps"] == 2
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

        monkeypatch.setattr(publication, "run_skyrl", forbid_live)
        monkeypatch.setattr("marin.rl.skyrl.read_skyrl_launch_receipt", forbid_live)
        monkeypatch.setattr("rigging.provenance.Provenance.capture", forbid_live)
        monkeypatch.setattr("marin.rl.cli.iris_client", forbid_live)
        rebuilt = publication.build_record(str(source), preset)
        assert rebuilt["elapsed_seconds"] == 7200 and rebuilt["final"] == row["final"]
        assert rebuilt["source_revision"] == row["source_revision"] and path.read_bytes() == saved
    assert "vllm_version" not in row["evaluation"]["protocol"]
    if case == "native-composed":
        assert row["evaluation"]["protocol"]["task_env"]["RECORD_BUILDER_ENV"] == "native-composition"
        assert row["evaluation"]["protocol"]["context_budget"] == {"max_new_tokens_per_turn": 8192, "max_turns": 1}
    if case == "native-failed":
        assert row["optimizer_steps"] == 0


@pytest.mark.integration
def test_snowball_publication_retries_keep_the_first_gate_without_native_execution(
    rl_progress_client, recorded_runs, monkeypatch, tmp_path
):
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
    monkeypatch.setattr(publication, "run_skyrl", lambda config: SkyRLRun.raw_load(str(source)))
    publication.run_loop(native_config(source, launch), preset, preset.name)
    raw_path = source / "exports/attempts/native-one/loop-result.json"

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
    (source / ".executor_status").write_text("FAILED")

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

    def unavailable_readback(origin, method, path, *, json_body=None):
        if method == "POST" and json_body["sql"] == publisher.READ and interrupted["once"]:
            interrupted["once"] = False
            raise RuntimeError("Readback unavailable after insert")
        return request(origin, method, path, json_body=json_body)

    monkeypatch.setattr(publisher, "marina_request", unavailable_readback)
    step = publication.publication_step(training, str(raw_path), gate_path=first_gate, issue=73)
    with pytest.raises(RuntimeError, match="step\\(s\\) failed"):
        resolve(step)
    frozen = Path(step.path()) / publication.RESULT_FILENAME
    first_bytes = frozen.read_bytes()
    raw = json.loads(raw_path.read_text())
    raw["elapsed_seconds"] = 7200
    write_json(raw_path, raw)
    monkeypatch.setattr(publisher, "marina_request", request)
    retried = publication.publication_step(training, str(raw_path), gate_path=second_gate, issue=74)
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
    assert json.loads((Path(second_gate) / gate.GATE_REPORT_FILENAME).read_text())["verdict"] == "fail"
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
