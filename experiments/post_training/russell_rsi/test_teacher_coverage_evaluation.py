# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import StepContext, artifact_identity
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL
from marin.rl.skyrl import SkyRLRun

from experiments.post_training.russell_rsi import launch_interrupted_calibration_sft as foreground
from experiments.post_training.russell_rsi import test_teacher_coverage_study as fixtures
from experiments.post_training.russell_rsi.calibration_recovery import LAUNCH_PROTOCOL
from experiments.post_training.russell_rsi.launch import adopted
from experiments.post_training.russell_rsi.launch_post_teacher_sft import post_sft_evaluation_stages
from experiments.post_training.russell_rsi.launch_teacher_coverage import main
from experiments.post_training.russell_rsi.teacher_coverage_study import coverage_evaluation, coverage_sft_workflow
from experiments.post_training.russell_rsi.test_teacher_four_pass import four_update_qualification

coverage_condition_inputs = fixtures.coverage_condition_inputs
coverage_training_inputs = fixtures.coverage_training_inputs
incumbent_inputs = fixtures.incumbent_inputs
continuation_inputs = fixtures.continuation_inputs


@pytest.fixture
def completed_coverage_evaluation(coverage_training_inputs, tmp_path):
    dose, _ = coverage_training_inputs
    source_head = "f" * 40
    source_files = {"coverage.py": "a" * 64}
    review = fixtures.pin(
        tmp_path / "evaluation-source-review.json",
        {
            "status": "approved",
            "source_head": source_head,
            "source_files": source_files,
            "runtime_commit": MARIN_SKYRL.commit,
        },
    )
    dose["sft_source_review"] = review
    dose_pin = fixtures.pin(tmp_path / "completed-dose.json", dose)
    stages = coverage_sft_workflow(dose)
    prefix = str(tmp_path / "completed-artifacts")
    post = {
        "version": "2026.10.06.24",
        "runtime_commit": MARIN_SKYRL.commit,
        "protocol": fixtures.PROTOCOL,
        "sft_config": dose_pin,
        "sft_source_review": review,
    }
    records = {}
    reload_pin = None
    for role, handle in (("sft", stages["train"]), ("reload", stages["reload"])):
        root = Path(handle.path(prefix))
        ctx = StepContext.for_run(str(root), prefix, deps=handle.deps, runtime_args=handle.runtime_args)
        bound = json.loads(canonical_json(handle.build_config(ctx)))
        producer = fixtures.producer_pins(handle, root)
        record = json.loads(Path(producer["producer"]["uri"]).read_bytes())
        record.update(config=bound, provenance={"base_commit": source_head, "dirty": False})
        request = fixtures.pin(
            tmp_path / f"{role}-request.json",
            {
                "stage": role,
                "version": handle.version,
                "source_head": source_head,
                "runtime_commit": MARIN_SKYRL.commit,
                "config_uri": dose_pin["uri"],
                "config_sha256": dose_pin["sha256"],
            },
        )
        preflight = fixtures.pin(
            tmp_path / f"{role}-preflight.json",
            {"exit_code": 0, "identity": {"source_head": source_head, "request_sha256": request["sha256"]}},
        )
        launch = {
            "protocol": LAUNCH_PROTOCOL,
            "source_head": source_head,
            "runtime_commit": MARIN_SKYRL.commit,
            "source_files": source_files,
            "config": dose_pin,
            "source_review": review,
            "producer_identity": artifact_identity(handle),
            "output_path": str(root),
            "request": request,
            "preflight": preflight,
            "bound_config": bound,
            "rl_authorized": False,
            "signal_gate_passed": None,
        }
        if role == "reload":
            reload_pin = fixtures.pin(
                root / "reload-smoke" / "record.json",
                {
                    "run_id": "reload-smoke",
                    "status": "succeeded",
                    "error": None,
                    "metrics": {"mmlu_abstract_algebra_0shot": {"sample_len": 1.0, "acc,none": 0.0}},
                    "model": {
                        "location": bound["model"]["location"],
                        "config": {"identity": bound["model"]["identity"]},
                    },
                    "eval": {"name": "mmlu-smoke", "evalchemy": {"max_eval_instances": 1}},
                },
            )
            record["result"] = {
                "records_prefix": str(root),
                "run_ids": ["reload-smoke"],
                "results_paths": [str(root / "reload-smoke" / "results")],
            }
        post[f"{role}_producer"] = fixtures.pin(root / ".artifact.json", record)
        post[f"{role}_launch_proof"] = fixtures.pin(tmp_path / f"{role}-launch.json", launch)
        records[role] = record
    assert reload_pin is not None
    loader_root = Path(stages["loader"].path(prefix))
    post["loader"] = fixtures.producer_pins(stages["loader"], loader_root)
    post["loader_proof"] = fixtures.pin(
        loader_root / "loader-proof.json",
        {
            "status": "passed",
            "input_pins": {"study_config_sha256": dose["study_config"]["sha256"]},
            "jsonl_sha256": dose["collection"]["train"]["sha256"],
            "example_exposures": 32,
        },
    )
    qualification = four_update_qualification(artifact_identity(stages["train"]), records["sft"]["output_path"])
    qualification["source_config_sha256"] = dose_pin["sha256"]
    qualification["serving_reload"].update(evidence_uri=reload_pin["uri"], evidence_sha256=reload_pin["sha256"])
    trainer = records["sft"]["config"]["train_config"]["trainer"]
    destination = Path(trainer["tracker"][0]["metric_destination"])
    dtype = trainer["mp"]["param_dtype"]["__dtype__"]
    event_pins = []
    for step in qualification["optimizer_steps"]:
        event = {
            "tracker": "json_logger",
            "event": "log",
            "run_id": trainer["id"],
            "step": step["step"],
            "metrics": {
                "train/loss": step["loss"],
                "optim/learning_rate": float(np.asarray(step["learning_rate"], dtype=dtype)),
                "grad/norm/total": step["gradient_norm"],
                "updates/norm/total": step["update_norm"],
            },
        }
        raw = json.dumps(event, sort_keys=True).encode()
        sha256 = hashlib.sha256(raw).hexdigest()
        event_pins.append(fixtures.pin(destination / f"step-{step['step']}-{sha256}.json", event))
    qualification["optimizer_telemetry"] = {
        "files": event_pins,
        "skip_bad_steps": False,
        "crash_on_nan": True,
        "crash_on_inf": True,
        "learning_rate_dtype": dtype,
    }
    post["qualification"] = fixtures.pin(tmp_path / "coverage-qualification.json", qualification)
    return post, stages, qualification


def test_coverage_evaluation_binds_actual_student_and_one_candidate(completed_coverage_evaluation, tmp_path):
    post, stages, qualification = completed_coverage_evaluation
    outputs = coverage_evaluation(post)
    assert set(outputs) == {"coding-sft", "retention-sft", "selection", "terminal"}
    handles = list(graph_handles([outputs["terminal"]]))
    assert not any(
        "calibration" in handle.name or "sft-rl" in handle.name or "skyrl" in handle.name for handle in handles
    )
    coding = next(handle for handle in handles if handle.name.startswith("evals/") and "humaneval" in handle.name)
    prefix = str(tmp_path / "evaluation-artifacts")
    bound = coding.build_config(
        StepContext.for_run(coding.path(prefix), prefix, deps=coding.deps, runtime_args=coding.runtime_args)
    )
    assert bound.model.location == qualification["hf_export_uri"]
    assert bound.model.identity == artifact_identity(stages["train"])
    retention = outputs["retention-sft"]
    retained = retention.build_config(
        StepContext.for_run(retention.path(prefix), prefix, deps=retention.deps, runtime_args=retention.runtime_args)
    )
    assert retained.model_uri == qualification["hf_export_uri"]
    assert retained.model_identity == artifact_identity(stages["train"])


@pytest.mark.parametrize("failure", ["wrong_producer", "missing_step", "wrong_metric"])
def test_coverage_evaluation_rejects_unbound_optimizer_evidence(completed_coverage_evaluation, tmp_path, failure):
    post, _, qualification = completed_coverage_evaluation
    if failure == "wrong_producer":
        qualification["sft_identity"] = "wrong-student"
    elif failure == "missing_step":
        qualification["optimizer_telemetry"]["files"] = qualification["optimizer_telemetry"]["files"][:-1]
    else:
        original = qualification["optimizer_telemetry"]["files"][0]
        event = json.loads(Path(original["uri"]).read_bytes())
        event["metrics"]["grad/norm/total"] = 99.0
        raw = json.dumps(event, sort_keys=True).encode()
        sha256 = hashlib.sha256(raw).hexdigest()
        path = Path(original["uri"]).parent / f"step-0-{sha256}.json"
        qualification["optimizer_telemetry"]["files"][0] = fixtures.pin(path, event)
    post["qualification"] = fixtures.pin(tmp_path / f"{failure}-qualification.json", qualification)
    errors = {
        "wrong_producer": "pinned 4-update",
        "missing_step": "cover all qualified steps",
        "wrong_metric": "metrics differ",
    }
    with pytest.raises(ValueError, match=errors[failure]):
        coverage_evaluation(post)


@pytest.mark.parametrize("stage", ["collect", "sft", "reload", "evaluate"])
def test_public_coverage_cli_preflight_uses_explicit_runtime(
    stage, completed_coverage_evaluation, tmp_path, monkeypatch
):
    post, _, _ = completed_coverage_evaluation
    dose = json.loads(Path(post["sft_config"]["uri"]).read_bytes())
    study = json.loads(Path(dose["study_config"]["uri"]).read_bytes())
    config = study if stage == "collect" else post if stage == "evaluate" else dose
    root = Path(foreground.__file__).resolve().parents[3]
    check_output = foreground.subprocess.check_output
    head = check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    review = fixtures.pin(
        tmp_path / "cli-source-review.json",
        {
            "status": "approved",
            "source_path": str(root),
            "source_head": head,
        },
    )

    # The test tree has pending changes; keep all source and installed-runtime checks active.
    def git_output(command, **kwargs):
        if command == ["git", "status", "--porcelain"]:
            return ""
        return check_output(command, **kwargs)

    monkeypatch.setattr(foreground.subprocess, "check_output", git_output)
    pin = fixtures.pin(tmp_path / "cli-config.json", config)
    result = CliRunner().invoke(
        main,
        [
            "--config-uri",
            pin["uri"],
            "--config-sha256",
            pin["sha256"],
            "--source-review-uri",
            review["uri"],
            "--source-review-sha256",
            review["sha256"],
            "--stage",
            stage,
            "--version",
            config["version"],
        ],
    )
    assert result.exit_code == 0, result.output


def test_checkpoint_locations_bind_both_evaluation_fingerprints(completed_coverage_evaluation):
    post, stages, _ = completed_coverage_evaluation
    dose = json.loads(Path(post["sft_config"]["uri"]).read_bytes())
    study = json.loads(Path(dose["study_config"]["uri"]).read_bytes())
    v21 = json.loads(Path(study["condition"]["v21_config"]["uri"]).read_bytes())
    retention = adopted(v21["retention"])
    model = stages["train"]

    def graph(location, checkpoint=model, locations=None):
        return post_sft_evaluation_stages(
            v21,
            retention=retention,
            export_uri="<unused qualified export>",
            checkpoints=[("sft", checkpoint)],
            barriers=(),
            outputs={},
            study=None,
            checkpoint_locations={"sft": location} if locations is None else locations,
        )

    first, second = graph("/qualified/export-one"), graph("/qualified/export-two")
    for role in ("coding-sft", "retention-sft"):
        assert artifact_identity(first[role]) != artifact_identity(second[role])
    with pytest.raises(ValueError, match="SkyRL checkpoint locations"):
        graph("/invalid/override", replace(model, artifact_type=SkyRLRun))
    with pytest.raises(ValueError, match="missing for sft"):
        graph("/qualified/export-one", locations={})
