# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.execution.artifact import RECORD_FILENAME, ArtifactRecord, result_type_name
from marin.execution.lazy import ArtifactStep, resolve
from marin.rl.skyrl import SkyRLRun, read_skyrl_launch_config
from pydantic import ValidationError

from experiments.post_training import publish_rl_progress as publication
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_publication as snowball_publication
from experiments.post_training.snowball_rl_test_fixtures import native_config, write_json


def test_progress_publication_retains_attempts_and_serves_valid_comparisons(rl_progress_client):
    client, request = rl_progress_client
    query_path = f"/a/{publication.APPLET_ID}/query"
    membership = [f"task-{index}" for index in range(100)]
    record = {
        "run_id": "test/control",
        "attempt_id": "native-first",
        "label": "learning-medium lr=2e-06",
        "campaign": "experiment",
        "series": "learning-medium",
        "issue": 123,
        "gate": "test/gate",
        "verdict": None,
        "min_scored_fraction": 0.99,
        "run_date": "2026-11-02T00:00:00Z",
        "date_basis": "submission_acknowledged",
        "started_at": "2026-11-02T00:00:00Z",
        "completed_at": "2026-11-02T01:00:00Z",
        "elapsed_seconds": 3600,
        "timing_scope": "native_job_start_to_finish",
        "status": "succeeded",
        "optimizer_steps": 25,
        "source_revision": "source-1",
        "data_identity": "data-1",
        "initializer_identity": "model-1",
        "evaluation": {
            "name": "test-panel",
            "version": "1",
            "task_count": 100,
            "task_set_sha256": "a" * 64,
            "membership_sha256": (
                hashlib.sha256(json.dumps(sorted(membership), separators=(",", ":")).encode()).hexdigest()
            ),
            "protocol": {"temperature": 0},
            "protocol_sha256": hashlib.sha256(b'{"temperature":0}').hexdigest(),
        },
        "baseline": {"reward": 0.2, "attempted": 100, "scored": 99},
        "final": {"reward": 0.4, "attempted": 100, "scored": 99},
        "artifact_uri": "s3://example/test",
        "evidence": [],
    }
    first = publication.publish_result(record, "http://localhost")
    assert first["inserted"] and first["verified"]
    assert publication.publish_result(copy.deepcopy(record), "http://localhost")["inserted"] is False
    changed = copy.deepcopy(record)
    changed["final"]["reward"] = 0.9
    with pytest.raises(ValueError, match="different record"):
        publication.publish_result(changed, "http://localhost")
    failed = record | {
        "attempt_id": "native-second",
        "started_at": "2026-11-03T00:00:00Z",
        "completed_at": "2026-11-03T01:00:00Z",
        "status": "failed",
        "baseline": None,
        "final": None,
        "optimizer_steps": 0,
    }
    assert publication.publish_result(failed, "http://localhost")["verified"]
    below = copy.deepcopy(record)
    below.update(run_id="test/under-coverage", attempt_id="native-third")
    below["final"]["scored"] = 98
    publication.publish_result(below, "http://localhost")
    other_membership = copy.deepcopy(record)
    other_membership.update(run_id="test/different-membership", attempt_id="native-fourth")
    other_membership["evaluation"]["membership_sha256"] = "b" * 64
    publication.publish_result(other_membership, "http://localhost")
    partial_baseline = copy.deepcopy(record)
    partial_baseline.update(run_id="test/partial-baseline", attempt_id="native-fifth")
    partial_baseline["baseline"]["scored"] = 98
    publication.publish_result(partial_baseline, "http://localhost")
    for field, value in (
        ("label", "arbitrary label"),
        ("issue", 987),
        ("task_set_sha256", "c" * 64),
        ("protocol_sha256", None),
        ("timing_scope", "launcher_start_to_finish"),
    ):
        distinct = copy.deepcopy(record)
        distinct.update(run_id="test/" + field, attempt_id=field)
        if field in {"task_set_sha256", "protocol_sha256"}:
            distinct["evaluation"][field] = value
            if field == "protocol_sha256":
                distinct["evaluation"]["protocol"] = {"temperature": 1}
                distinct["evaluation"][field] = hashlib.sha256(b'{"temperature":1}').hexdigest()
        else:
            distinct[field] = value
        publication.publish_result(distinct, "http://localhost")
    unknown = copy.deepcopy(record)
    unknown.update(run_id="test/unknown-protocol", attempt_id="unknown")
    unknown["evaluation"].update(protocol=None, protocol_sha256=None)
    publication.publish_result(unknown, "http://localhost")
    unknown_time = record | {
        "run_id": "test/unknown-time",
        "attempt_id": "unknown-time",
        "elapsed_seconds": None,
        "run_date": None,
        "started_at": None,
        "completed_at": None,
    }
    publication.publish_result(unknown_time, "http://localhost")
    malformed = copy.deepcopy(record)
    malformed.update(run_id="test/malformed", attempt_id="bad")
    malformed["evaluation"]["task_ids"] = [f"replacement-{index}" for index in range(100)]
    with pytest.raises(ValidationError, match="membership"):
        publication.publish_result(malformed, "http://localhost")
    underflow = copy.deepcopy(record)
    underflow.update(run_id="test/underflow", attempt_id="bad", elapsed_seconds=1e-323)
    overflow = copy.deepcopy(record)
    overflow.update(run_id="test/overflow", attempt_id="bad")
    overflow["baseline"]["reward"], overflow["final"]["reward"] = -1e308, 1e308
    duplicate = copy.deepcopy(malformed)
    duplicate.update(run_id="test/duplicate-membership")
    duplicate["evaluation"]["task_ids"] = ["duplicate"] * 100
    duplicate["evaluation"]["membership_sha256"] = hashlib.sha256(
        json.dumps(duplicate["evaluation"]["task_ids"], separators=(",", ":")).encode()
    ).hexdigest()
    for run_id, invalid in (
        ("test/underflow", underflow),
        ("test/overflow", overflow),
        ("test/duplicate-membership", duplicate),
        (malformed["run_id"], malformed),
        ("test/wrong-sql-key", record | {"attempt_id": "bad"}),
        ("test/missing-baseline", record | {"run_id": "test/missing-baseline", "attempt_id": "bad", "baseline": None}),
        ("test/bad-shape", ["unexpected"]),
        (
            "test/bad-timestamp",
            record | {"run_id": "test/bad-timestamp", "attempt_id": "bad", "started_at": "yesterday"},
        ),
    ):
        request(
            "http://localhost",
            "POST",
            query_path,
            json_body={
                "sql": publication.INSERT,
                "parameters": {
                    "run_id": run_id,
                    "attempt_id": "bad",
                    "record": json.dumps(invalid),
                },
            },
        )
    huge_extra = json.dumps(
        record | {"run_id": "test/huge-extra", "attempt_id": "bad", "extra": "large-number"}
    ).replace('"large-number"', "1" + "0" * 400 + ".5")
    request(
        "http://localhost",
        "POST",
        query_path,
        json_body={
            "sql": publication.INSERT,
            "parameters": {"run_id": "test/huge-extra", "attempt_id": "bad", "record": huge_extra},
        },
    )
    api = client.get(f"/a/{publication.APPLET_ID}/api/runs")
    assert api.status_code == 200
    payload = api.json()
    rows = {row["attempt_id"]: row for row in payload["runs"]}
    assert len(rows) == 12
    good = rows["native-first"]
    assert good["final"]["reward"] == 0.4
    assert good["partial_coverage"] and good["pareto_eligible"] and good["trusted_gain_eligible"]
    assert not good["provisional"] and good["gain_per_hour"] == pytest.approx(0.2)
    assert rows["native-second"]["status"] == "failed"
    assert rows["native-second"]["gain_per_hour"] is None
    assert not rows["native-second"]["pareto_eligible"]
    assert rows["native-third"]["provisional"] and not rows["native-third"]["pareto_eligible"]
    assert rows["native-fourth"]["comparison_key"] != good["comparison_key"]
    assert rows["native-fifth"]["pareto_eligible"] and not rows["native-fifth"]["trusted_gain_eligible"]
    assert rows["label"]["comparison_key"] == rows["issue"]["comparison_key"] == good["comparison_key"]
    assert (
        len(
            {
                rows[key]["comparison_key"]
                for key in ("native-first", "task_set_sha256", "protocol_sha256", "timing_scope")
            }
        )
        == 4
    )
    assert rows["unknown-time"]["final"]["reward"] == 0.4
    assert rows["unknown-time"]["elapsed_seconds"] is None and rows["unknown-time"]["gain_per_hour"] is None
    assert rows["unknown-time"]["provisional"] and not rows["unknown-time"]["pareto_eligible"]
    assert rows["unknown"]["comparison_key"] is None
    assert rows["unknown"]["provisional"] and not rows["unknown"]["pareto_eligible"]
    assert not rows["unknown"]["trusted_gain_eligible"]
    skipped = {row["run_id"]: row["error"] for row in payload["skipped_records"]}
    assert set(skipped) == {
        "test/malformed",
        "test/bad-shape",
        "test/bad-timestamp",
        "test/underflow",
        "test/overflow",
        "test/duplicate-membership",
        "test/huge-extra",
        "test/wrong-sql-key",
        "test/missing-baseline",
    }
    assert "evaluation" in skipped["test/malformed"]


def test_snowball_publication_retries_keep_the_first_gate_without_native_execution(
    rl_progress_client,
    recorded_runs,
    monkeypatch,
    tmp_path,
):
    client, request = rl_progress_client
    preset = replace(presets.PRESETS["learning-medium"], updates=2, measured=range(1, 3))
    source = recorded_runs(
        preset,
        "tests/snowball-publication",
        initial_score=0.3,
        score=0.5,
        unscored=15,
        attempt_id="native-one",
    )
    launch = read_skyrl_launch_config(str(source))
    tracking_path = source / "exports/attempts/native-one/tracking.json"
    tracking = json.loads(tracking_path.read_text())
    tracking.update(wandb_url="https://wandb.ai/test/publication/runs/native-one", engine={"model_dtype": "bfloat16"})
    write_json(tracking_path, tracking)
    monkeypatch.setattr(snowball_publication, "run_skyrl", lambda config: SkyRLRun.raw_load(str(source)))
    snowball_publication.run_loop(snowball_publication.RecordedRun(native_config(source, launch), preset, preset.name))
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
        if method == "POST" and json_body["sql"] == publication.READ and interrupted["once"]:
            interrupted["once"] = False
            raise RuntimeError("Readback unavailable after insert")
        return request(origin, method, path, json_body=json_body)

    monkeypatch.setattr(publication, "marina_request", unavailable_readback)
    step = snowball_publication.publication_step(training, str(raw_path), gate_path=first_gate, issue=73)
    with pytest.raises(RuntimeError, match=r"step\(s\) failed"):
        resolve(step)
    frozen = Path(step.path()) / snowball_publication.RESULT_FILENAME
    first_bytes = frozen.read_bytes()
    raw = json.loads(raw_path.read_text())
    raw["elapsed_seconds"] = 7200
    write_json(raw_path, raw)
    monkeypatch.setattr(publication, "marina_request", request)
    retried = snowball_publication.publication_step(training, str(raw_path), gate_path=second_gate, issue=74)
    resolve(retried)
    assert frozen.read_bytes() == first_bytes
    receipt = json.loads((Path(retried.path()) / snowball_publication.PUBLICATION_FILENAME).read_text())
    assert receipt["verified"] and receipt["inserted"] is False
    payload = client.get(f"/a/{publication.APPLET_ID}/api/runs").json()
    assert payload["skipped_records"] == [] and len(payload["runs"]) == 1
    served = payload["runs"][0]
    assert (served["run_id"], served["attempt_id"]) == (launch["run"]["id"], "native-one")
    assert served["gate"] == "first-gate" and served["verdict"] == "pass" and served["issue"] == 73
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
        produced = snowball_publication.build_record(str(variant), preset)
        publication.publish_result(produced, "http://localhost")
    variants = {row["run_id"]: row for row in client.get(f"/a/{publication.APPLET_ID}/api/runs").json()["runs"]}
    assert variants["tests/engine"]["comparison_key"] == served["comparison_key"]
    assert variants["tests/engine"]["engine"]["model_dtype"] == "float16"
    for change in ("request", "content", "membership"):
        assert variants[f"tests/{change}"]["comparison_key"] != served["comparison_key"]
    assert variants["tests/content"]["evaluation"]["task_set_sha256"] != served["evaluation"]["task_set_sha256"]
    assert variants["tests/membership"]["evaluation"]["membership_sha256"] != served["evaluation"]["membership_sha256"]
