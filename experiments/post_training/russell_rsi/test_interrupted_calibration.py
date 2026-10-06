# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use synthetic study metadata; these checks run no provider or checkpoint inference."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from fray.client import Client
from fray.current_client import current_client, set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import ResourceConfig, create_environment
from iris.client.client import IrisClient, IrisContext, iris_ctx, iris_ctx_scope
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.rpc import controller_pb2, job_pb2
from marin.evaluation.evalchemy.client import CONFIG_ENV_KEY
from marin.evaluation.evalchemy.runner import EvalchemyRunConfig, _run_evalchemy_child
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.runner import EndpointRoute
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity, run
from marin.experiment.cli import graph_handles
from marin.inference.config import IrisConfig, RemoteInferenceConfig, ServedModelConfig, VllmEngineConfig
from marin.inference.iris import remote_inference
from marin.inference.types import OpenAIEndpoint, RunningModel

from experiments.post_training.russell_rsi import interrupted_calibration, test_teacher_four_pass
from experiments.post_training.russell_rsi.interrupted_calibration import (
    OUTPUT_PROTOCOL,
    PROTOCOL,
    WORKER_TIMEOUT_HOURS,
    BoundedIrisClient,
    coding_attempt,
    require_interruption,
    retention_request,
    run_foreground_coding,
)
from experiments.post_training.russell_rsi.launch import CLUSTER
from experiments.post_training.russell_rsi.teacher_four_pass import four_pass_post_workflow, four_pass_teacher_workflow

continuation_inputs = test_teacher_four_pass.continuation_inputs
study_inputs = test_teacher_four_pass.study_inputs


@pytest.fixture
def interrupted_study(study_inputs, tmp_path):
    study, pin = study_inputs
    producer = four_pass_teacher_workflow(study)["train"]
    original = {**study, "sft_uri": producer.path(str(tmp_path / "artifacts"))}
    pin(original, "sft_config", study)
    qualification = test_teacher_four_pass.four_update_qualification(artifact_identity(producer), original["sft_uri"])
    qualification["source_config_sha256"] = original["sft_config_sha256"]
    pin(original, "qualification", qualification)
    outputs = four_pass_post_workflow(original, "calibrate")
    model = outputs["calibration"].deps[1]
    config = {**original, "version": "2026.10.06.7"}
    carrier = {}
    pin(carrier, "original", original)
    pin(
        carrier,
        "journal",
        {
            "config": {"model_identity": artifact_identity(model)},
            "attempts": {"task": {str(index): "hash" for index in range(256)}},
        },
    )
    evidence = {}
    for label in (
        "terminal_root",
        "terminal_tree",
        "terminal_inventory",
        "issuance_census",
        "completion_issuance",
        "slot_dispositions",
        "grade_intake",
        "terminal_inventory_data",
    ):
        pin(carrier, label, {"fixture": label})
        evidence[label] = {"uri": carrier[f"{label}_uri"], "sha256": carrier[f"{label}_sha256"]}
    amendment = {
        "protocol": PROTOCOL,
        "calibration_status": "incomplete_infrastructure",
        "signal_gate_passed": None,
        "rl_authorized": False,
        "whole_cohort_replacements_remaining": 0,
        "repeated_issued_samples": 0,
        "source": {
            "config": {"uri": carrier["original_uri"], "sha256": carrier["original_sha256"]},
            "journal_binding": {"uri": carrier["journal_uri"], "sha256": carrier["journal_sha256"]},
            "qualification_sha256": original["qualification_sha256"],
            "model_identity": artifact_identity(model),
            "calibration_identity": artifact_identity(outputs["calibration"]),
        },
        "evidence": evidence,
        "execution": {
            "child_detachment": False,
            "cluster_changes": False,
            "coordinator": "foreground local artifact main",
            "priority": "batch",
            "remote_worker_cluster": CLUSTER,
        },
        "evaluation": {"version": config["version"], "conditions": ["sft"], "coding_limit": 32, "retention_limit": 3},
    }
    pin(config, "calibration_interruption", amendment)
    return config, amendment, model, pin


@pytest.mark.parametrize("gain", [False, True])
def test_interrupted_study_selects_sft_with_original_gates_without_calibration_or_rl(interrupted_study, tmp_path, gain):
    config, amendment, model, _ = interrupted_study
    outputs = four_pass_post_workflow(config, "evaluate-interrupted")
    graph = graph_handles([outputs["terminal"]])
    identities = {artifact_identity(step) for step in graph}
    assert artifact_identity(model) in identities
    assert amendment["source"]["calibration_identity"] not in identities
    assert not any(
        "calibration-development" in step.name or "replay" in step.name or "sft-rl" in step.name for step in graph
    )
    terminal = outputs["terminal"]
    bound = terminal.build_config(
        StepContext.for_run(
            str(tmp_path / "selection"),
            str(tmp_path / "outputs"),
            deps=terminal.deps,
        )
    )
    selected = bound.selection.record
    identity = selected.model_identities[0]
    coding = {
        "model_identity": identity,
        "panel_sha256": selected.panel_sha256,
        "scores": {"humanevalplus": (26 if gain else 25) / 32, "mbppplus": 27 / 32},
    }
    retained = {
        "model_identity": identity,
        "tasks_identity": selected.retention_identity,
        "count": 3,
        "task_rewards": {key: [int(index == 0)] for index, key in enumerate(selected.retention_task_ids)},
    }
    for directory, filename, value in (
        (selected.coding_paths[0], "coding-evidence.json", coding),
        (selected.retention_paths[0], "failure_summary.json", retained),
    ):
        Path(directory).mkdir(parents=True)
        (Path(directory) / filename).write_text(json.dumps(value))
    terminal.run(bound)
    record = json.loads((tmp_path / "selection/post-sft-selection.json").read_text())
    context = json.loads((tmp_path / "selection/calibration-interruption.json").read_text())
    assert record["protocol"] == OUTPUT_PROTOCOL
    assert record["sft_rl"] is None
    assert record["promoted"]["checkpoint_identity"] == (
        identity if gain else record["incumbent"]["checkpoint_identity"]
    )
    assert record["incumbent"]["development"] == [25 / 32, 27 / 32]
    assert context["signal_gate_passed"] is None and context["rl_authorized"] is False


@pytest.mark.parametrize("defect", ["replacement", "gate", "parent", "terminal_evidence"])
def test_interruption_rejects_replay_or_changed_science_before_build(interrupted_study, defect):
    config, amendment, _, pin = interrupted_study
    if defect == "replacement":
        amendment["whole_cohort_replacements_remaining"] = 1
    elif defect == "gate":
        amendment["signal_gate_passed"] = False
    elif defect == "parent":
        config = {**config, "parent": {**config["parent"], "uri": "/changed-model"}}
    else:
        Path(amendment["evidence"]["terminal_tree"]["uri"]).write_text("changed terminal tree")
    pin(config, "calibration_interruption", amendment)
    with pytest.raises(ValueError):
        require_interruption(config)


def test_coding_completed_replay_and_incomplete_refusal_precede_any_client(interrupted_study, tmp_path):
    config, _, _, _ = interrupted_study
    outputs = four_pass_post_workflow(config, "evaluate-interrupted")
    coding = outputs["coding-sft"].deps[0]
    bound = coding.build_config(
        StepContext.for_run(
            str(tmp_path / "coding"),
            str(tmp_path / "outputs"),
            deps=coding.deps,
            runtime_args=coding.runtime_args,
        )
    )
    saved = {
        "path": bound.artifact_path,
        "group_id": "saved",
        "records_prefix": "/saved-records",
        "run_ids": ["he", "mbpp"],
        "results_paths": ["/saved-he", "/saved-mbpp"],
    }

    async def finish():
        return saved

    asyncio.run(coding_attempt(bound).run(finish))
    result = run_foreground_coding(bound)
    assert result.results_paths == ("/saved-he", "/saved-mbpp")
    with pytest.raises(ValueError, match="Immutable record differs"):
        run_foreground_coding(replace(bound, model=replace(bound.model, identity="different-checkpoint")))
    (Path(bound.artifact_path) / "journal/coding/result.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        run_foreground_coding(bound)


@pytest.mark.parametrize(
    "transport_binding,route", [(None, EndpointRoute.DIRECT), ({"replacement": "v9"}, EndpointRoute.CAPABILITY)]
)
def test_foreground_coding_preserves_capability_route(
    interrupted_study, tmp_path, monkeypatch, transport_binding, route
):
    config, _, _, _ = interrupted_study
    coding = four_pass_post_workflow(config, "evaluate-interrupted")["coding-sft"].deps[0]
    bound = coding.build_config(
        StepContext.for_run(str(tmp_path / "coding"), str(tmp_path), deps=coding.deps, runtime_args=coding.runtime_args)
    )
    captured = []

    def capture(batch, **kwargs):
        captured.append(batch)
        raise CapturedSubmission

    monkeypatch.setattr(interrupted_calibration, "run_evaluation_batch", capture)
    client = BoundedIrisClient(cast(IrisClient, object()))
    with iris_ctx_scope(IrisContext(job_id=None, client=cast(IrisClient, client))):
        with pytest.raises(CapturedSubmission):
            run_foreground_coding(bound, transport_binding=transport_binding)
    (batch,) = captured
    assert [item.endpoint_route for item in batch.evaluations] == [route] * 2
    assert [item.executor.config.extra_model_args["max_retries"] for item in batch.evaluations] == [1, 1]


def save_foreground_context(config: dict) -> None:
    path = config["path"]
    context = iris_ctx()
    Path(path).mkdir(parents=True, exist_ok=True)
    (Path(path) / "context.json").write_text(
        json.dumps(
            {
                "job_id": context.job_id,
                "iris_client_is_fray_client": context.client is current_client(),
                "bounded_client": isinstance(context.client, BoundedIrisClient),
            }
        )
    )


def test_parallel_artifact_workers_keep_foreground_context(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    client = BoundedIrisClient(cast(IrisClient, object()))
    steps = [
        ArtifactStep(
            name=f"documents/foreground-{label}",
            version="2026.10.06.7",
            artifact_type=Artifact,
            build_config=lambda ctx: {"path": ctx.output_path},
            run=save_foreground_context,
        )
        for label in ("coding", "retention")
    ]
    with iris_ctx_scope(IrisContext(job_id=None, client=cast(IrisClient, client))):
        with set_current_client(cast(Client, client)):
            run(*steps, max_concurrent=2, force_run_failed=False)
    for step in steps:
        context = json.loads((Path(step.path()) / "context.json").read_text())
        assert context == {"job_id": None, "iris_client_is_fray_client": True, "bounded_client": True}


class CapturedSubmission(Exception):
    pass


@pytest.mark.parametrize("stage", ["serving", "retention", "evalchemy"])
def test_direct_cluster_route_in_serialized_worker_requests(interrupted_study, tmp_path, monkeypatch, stage):
    captured = []

    def capture(request, **kwargs):
        captured.append(controller_pb2.Controller.LaunchJobRequest.FromString(request.SerializeToString()))
        raise CapturedSubmission

    cluster = RemoteClusterClient("http://controller.invalid", bundle_id="reviewed-source")
    monkeypatch.setattr(cluster._client, "launch_job", capture)
    client = BoundedIrisClient(IrisClient(cluster))
    fray = FrayIrisClient.from_iris_client(cast(IrisClient, client))
    config, _, _, _ = interrupted_study
    outputs = four_pass_post_workflow(config, "evaluate-interrupted")
    retention = outputs["retention-sft"]
    bound = retention.build_config(StepContext.for_run(str(tmp_path / "retention"), str(tmp_path), deps=retention.deps))
    resources = ResourceConfig.with_gpu("H100", 8, regions=["us-east"])
    try:
        with iris_ctx_scope(IrisContext(job_id=None, client=cast(IrisClient, client))):
            with set_current_client(fray), pytest.raises(CapturedSubmission):
                if stage == "retention":
                    request = retention_request(bound)
                    fray.submit(request)
                elif stage == "serving":
                    with remote_inference(
                        RemoteInferenceConfig(
                            model=ServedModelConfig(weights="no-weight-read", tensor_parallel_size=8),
                            engine=VllmEngineConfig(),
                            iris=IrisConfig(worker_resources=resources, worker_environment=create_environment()),
                        )
                    ):
                        pytest.fail("No worker may start")
                else:
                    _run_evalchemy_child(
                        RunningModel(
                            OpenAIEndpoint(
                                "https://iris.example/proxy/t/cluster=cw-us-east-02a/token/endpoint/v1",
                                "frozen-model",
                            ),
                            tokenizer="frozen-tokenizer",
                        ),
                        EvalchemyRunConfig(name="coding", tasks=(EvalTaskConfig("humanevalplus", None),)),
                        str(tmp_path),
                        {},
                    )
    finally:
        cluster.shutdown()
    (wire,) = captured
    assert not any(constraint.key == CLUSTER_CONSTRAINT_KEY for constraint in wire.constraints)
    if stage == "evalchemy":
        child = json.loads(wire.environment.env_vars[CONFIG_ENV_KEY])
        assert child["base_url"] == "https://iris.example/proxy/t/cluster=cw-us-east-02a/token/endpoint/v1"
        assert child["model_id"] == "frozen-model"
    if stage != "evalchemy":
        assert wire.resources.device.gpu.count == 8
    if stage == "serving":
        assert any(constraint.key == "region" for constraint in wire.constraints)
    if stage == "retention":
        assert wire.max_retries_failure == wire.max_retries_preemption == wire.max_task_failures == 0
        assert wire.priority_band == job_pb2.PRIORITY_BAND_BATCH
        assert wire.timeout.milliseconds == WORKER_TIMEOUT_HOURS * 3600 * 1000


@pytest.mark.parametrize("operator,cluster", [(ConstraintOp.EQ, "cw-us-west-04a"), (ConstraintOp.NE, CLUSTER)])
def test_direct_cluster_route_rejects_other_cluster_before_rpc(operator, cluster):
    client = BoundedIrisClient(cast(IrisClient, object()))
    constraint = Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=operator, value=cluster)
    with pytest.raises(ValueError, match="Unsupported cluster"):
        client.submit(constraints=[constraint])
