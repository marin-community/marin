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
from marin.evaluation.records import EvalRef, Provenance, RunStatus, read_record
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

from experiments.evaluation.evals import EVALS, HarborDefinition
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


@pytest.mark.parametrize(
    ("platform", "records_prefix", "mirror_prefix"),
    [
        (
            Platform.TPU,
            "gs://marin-eval-metadata/runs",
            "gs://marin-us-west4/tmp/ttl=7d/evaluation/harbor-datasets",
        ),
        (
            Platform.GPU,
            "s3://marin-us-east-02a/marin/eval-metadata/runs",
            "s3://marin-us-east-02a/tmp/ttl=7d/evaluation/harbor-datasets",
        ),
    ],
)
def test_agentic_evaluation_uses_a_revision_pinned_regional_dataset_artifact(
    platform,
    records_prefix,
    mirror_prefix,
    monkeypatch,
):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("tb2-lite",),
        platform=platform,
        accelerator=None,
        limit=None,
        records_prefix=records_prefix,
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
    assert harbor.repository == "DCAgent2/terminal_bench_2"
    assert harbor.commit == "693231ec029249e7c91ed2e414bcc9c45d7cd879"
    assert harbor.mirror_uri.startswith(mirror_prefix)
    assert harbor.mirror_uri.endswith(f"DCAgent2--terminal_bench_2/{harbor.commit}")
    assert evaluation.executor.dataset_artifact.path() == harbor.mirror_uri


@pytest.mark.parametrize(
    "eval_name",
    [
        "tb2",
        "swebench",
        "swebench-full",
        "gaia",
        "bfcl",
        "aider",
        "medagentbench",
        "financeagent",
        "grug-opencode-id",
    ],
)
def test_hugging_face_harbor_presets_use_full_immutable_commits(eval_name):
    definition = EVALS[eval_name]

    assert isinstance(definition, HarborDefinition)
    assert definition.dataset_artifact is not None
    assert definition.config.dataset == definition.dataset_artifact.repository
    assert definition.config.revision == definition.dataset_artifact.commit
    assert len(definition.config.revision) == 40
    assert all(character in "0123456789abcdef" for character in definition.config.revision)
