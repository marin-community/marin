# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the endpoint-oriented evaluation loop and durable records."""

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint
from marin.evaluation.records import EvalRef, HarborRef, Provenance, RunStatus, read_record
from marin.evaluation.runner import (
    Evaluation,
    EvaluationBatch,
    EvaluationError,
    EvaluationIdentity,
    EvaluationOutcome,
    evaluate_batch,
    submit_evaluation_batch,
)
from marin.inference.iris import RemoteInferenceSession
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem import StoragePath

from experiments.evaluation.evals import EVALS
from experiments.evaluation.launch import LaunchSpec, build_evaluation_batch


def _successful_evaluation(
    model: RunningModel,
    output_dir: str,
    _env_vars: Mapping[str, str],
) -> EvaluationOutcome:
    output = StoragePath(output_dir)
    output.mkdirs()
    (output / "endpoint.txt").write_text(model.endpoint.base_url)
    return EvaluationOutcome(metrics={"task": {"accuracy": 0.75}}, jobs={"eval": "/eval/success"})


def _failed_evaluation(
    _model: RunningModel,
    _output_dir: str,
    _env_vars: Mapping[str, str],
) -> EvaluationOutcome:
    raise EvaluationError(
        "evaluation failed",
        status=RunStatus.FAILED,
        jobs={"eval": "/eval/failure"},
        log_tails={"eval": ("failure detail",)},
        eval_ref=EvalRef(
            name="failure",
            mechanism="harbor",
            harbor=HarborRef(
                dataset="benchmark",
                version="commit",
                agent="agent",
                env="sandbox",
                mirror_uri="s3://regional/artifact",
            ),
        ),
    )


def _evaluation(root: Path, name: str, executor) -> Evaluation:
    return Evaluation(
        identity=EvaluationIdentity(
            run_id=f"run-{name}",
            created_at="2026-07-24T00:00:00+00:00",
            output_dir=str(root / name),
            eval_ref=EvalRef(name=name, mechanism="test"),
        ),
        executor=executor,
    )


def test_evaluate_batch_persists_failures_and_continues_on_the_same_endpoint(tmp_path):
    records = tmp_path / "records"
    endpoint = "https://iris.example/proxy/t/token/inference/v1"
    session = RemoteInferenceSession(
        model=RunningModel(
            endpoint=OpenAIEndpoint(base_url=endpoint, model="model"),
            tokenizer="tokenizer",
        ),
        jobs=(),
        streaming=True,
        tensor_parallel_size=1,
        backend_name="vllm",
    )
    batch = EvaluationBatch(
        group_id="group",
        user="tester",
        version="v1",
        description=None,
        records_prefix=str(records),
        model=ModelConfig(
            name="model",
            location="org/model",
            tokenizer="tokenizer",
            resource_hint=ResourceHint(hbm_gb=3),
        ),
        accelerator=AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-4", region="us-central1"),
        capability_origin="https://iris.example",
        api_model="model",
        evaluations=(
            _evaluation(tmp_path, "failure", _failed_evaluation),
            _evaluation(tmp_path, "success", _successful_evaluation),
        ),
        provenance=Provenance(git_sha="abc", eval_image="image", launch_host="host"),
    )

    with pytest.raises(RuntimeError, match="1 of 2 evals failed"):
        evaluate_batch(batch, session, orchestrator_job_id="/orchestrator", env_vars={})

    failed = read_record(str(records / "run-failure" / "record.json"))
    succeeded = read_record(str(records / "run-success" / "record.json"))
    assert failed.status is RunStatus.FAILED
    assert failed.jobs == {"orchestrator": "/orchestrator", "eval": "/eval/failure"}
    assert failed.log_tails == {"eval": ("failure detail",)}
    assert failed.evaluation.harbor is not None
    assert failed.evaluation.harbor.mirror_uri == "s3://regional/artifact"
    assert succeeded.status is RunStatus.SUCCEEDED
    assert succeeded.metrics == {"task": {"accuracy": 0.75}}
    assert (tmp_path / "success" / "endpoint.txt").read_text() == endpoint


def test_submit_evaluation_batch_resolves_declared_secrets_outside_the_pickled_batch(tmp_path, monkeypatch):
    captured: dict = {}
    resolved_value = "resolved-evaluation-secret"

    class Client:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(job_id="/eval/job")

    monkeypatch.setenv("MARIN_TEST_EVAL_SECRET", resolved_value)
    monkeypatch.setenv("MARIN_PREFIX", "gs://launcher-region")
    evaluation = Evaluation(
        identity=EvaluationIdentity(
            run_id="run-secret",
            created_at="2026-07-24T00:00:00+00:00",
            output_dir=str(tmp_path / "secret"),
            eval_ref=EvalRef(name="secret", mechanism="test"),
        ),
        executor=_successful_evaluation,
    )
    batch = EvaluationBatch(
        group_id="group",
        user="tester",
        version=None,
        description=None,
        records_prefix=str(tmp_path / "records"),
        model=ModelConfig(
            name="model",
            location="org/model",
            tokenizer="tokenizer",
            resource_hint=ResourceHint(hbm_gb=3),
        ),
        accelerator=AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-4", region="us-central1"),
        capability_origin="https://iris.example",
        api_model="model",
        evaluations=(evaluation,),
        provenance=Provenance(git_sha="abc", eval_image="image", launch_host="host"),
        secret_env={"DAYTONA_API_KEY": ("env:MARIN_TEST_EVAL_SECRET",)},
    )

    submit_evaluation_batch(batch, Client())

    assert captured["environment"].env_vars["DAYTONA_API_KEY"] == resolved_value
    assert "MARIN_PREFIX" not in captured["environment"].env_vars
    assert resolved_value.encode() not in captured["entrypoint"].workdir_files["_callable.pkl"]


def test_build_evaluation_batch_merges_the_shared_daytona_spec(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("aime-harbor", "tb2"),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        cluster="marin",
    )

    batch = build_evaluation_batch(
        spec,
        Provenance(git_sha="abc", eval_image="image", launch_host="host"),
        "tester",
    )

    assert batch.secret_env == {
        "DAYTONA_API_KEY": (
            "env:DAYTONA_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest",
        )
    }


def test_build_evaluation_batch_rejects_conflicting_secret_specs(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    conflicting = replace(
        EVALS["aime-harbor"],
        name="conflicting-daytona",
        secret_env={"DAYTONA_API_KEY": ("env:OTHER_DAYTONA_API_KEY",)},
    )
    monkeypatch.setitem(EVALS, "conflicting-daytona", conflicting)
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("aime-harbor", "conflicting-daytona"),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        cluster="marin",
    )

    with pytest.raises(ValueError, match="conflicting secret specifications for DAYTONA_API_KEY"):
        build_evaluation_batch(
            spec,
            Provenance(git_sha="abc", eval_image="image", launch_host="host"),
            "tester",
        )


def test_agentic_evaluation_uses_a_revision_pinned_dataset_artifact(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("tb2-lite",),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=None,
        records_prefix="s3://marin-us-east-02a/marin/eval-metadata/runs",
        cluster="marin",
    )

    batch = build_evaluation_batch(
        spec,
        Provenance(git_sha="abc", eval_image="image", launch_host="host"),
        "tester",
    )

    evaluation = batch.evaluations[0]
    harbor = evaluation.identity.eval_ref.harbor
    assert harbor is not None
    assert harbor.repository == evaluation.executor.config.dataset
    assert harbor.commit == evaluation.executor.config.revision
    assert harbor.mirror_uri is None
    artifact = evaluation.executor.dataset_artifact
    assert artifact is not None
    assert artifact.override_path is None
    assert artifact.path("gs://marin-us-west4") == (
        "gs://marin-us-west4/evaluation/harbor-datasets/DCAgent2--terminal_bench_2/2026.07.27"
    )
    assert artifact.path("s3://marin-us-east-02a/marin") == (
        "s3://marin-us-east-02a/marin/evaluation/harbor-datasets/DCAgent2--terminal_bench_2/2026.07.27"
    )
