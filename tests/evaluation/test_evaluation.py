# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the endpoint-oriented evaluation loop and durable records."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from marin.evaluation.harbor.driver_config import HARBOR_RUNTIME, HarborDatasetKind, ValidatedHarborConfig
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint
from marin.evaluation.records import EvalRef, RunStatus, read_record
from marin.evaluation.runner import (
    Evaluation,
    EvaluationBatch,
    EvaluationError,
    EvaluationIdentity,
    EvaluationOutcome,
    LaunchProvenance,
    evaluate_batch,
    submit_evaluation_batch,
)
from marin.external_dependencies import EVALCHEMY
from marin.inference.iris import RemoteInferenceSession
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem import StoragePath

from experiments.evaluation.cli import cli
from experiments.evaluation.evals import EVALS
from experiments.evaluation.launch import (
    HarborConfigSelection,
    LaunchSpec,
    build_evaluation_batch,
)


def _install_fake_harbor_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    def preflight(requests):
        configs = []
        for path, _model_agent_kwargs in requests:
            policy = json.dumps({"source": path.name}, separators=(",", ":"))
            configs.append(
                ValidatedHarborConfig(
                    stable_policy_json=policy,
                    digest=f"sha256:{hashlib.sha256(policy.encode()).hexdigest()}",
                    dataset_kind=HarborDatasetKind.HARBOR_REGISTRY,
                    dataset_selector="aime",
                    dataset_revision="1.0",
                    config_dir=path.parent,
                    agent="opencode",
                    environment="daytona",
                )
            )
        return tuple(configs)

    monkeypatch.setattr("experiments.evaluation.launch.preflight_harbor_configs", preflight)


def _write_harbor_config(path: Path, *, agents: str | None = None) -> Path:
    path.write_text(
        f"""
job_name: external-aime
jobs_dir: ignored-by-marin
n_attempts: 2
n_concurrent_trials: 3
environment:
  type: daytona
  force_build: true
agents:
{agents or '''  - name: opencode
    model_name: ignored-by-marin
    kwargs:
      trajectory_config:
        raw_content: false
      opencode_config:
        compaction:
          auto: false'''}
datasets:
  - name: aime
    version: "1.0"
"""
    )
    return path


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
            eval_runtime="test-runtime",
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
        provenance=LaunchProvenance(git_sha="abc", launch_host="host"),
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
    assert succeeded.provenance.eval_runtime == "test-runtime"
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
            eval_runtime="test-runtime",
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
        provenance=LaunchProvenance(git_sha="abc", launch_host="host"),
        secret_env={"DAYTONA_API_KEY": ("env:MARIN_TEST_EVAL_SECRET",)},
    )

    submit_evaluation_batch(batch, Client())

    assert captured["environment"].env_vars["DAYTONA_API_KEY"] == resolved_value
    assert resolved_value.encode() not in captured["entrypoint"].workdir_files["_callable.pkl"]


def test_build_evaluation_batch_merges_the_shared_daytona_spec(monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("aime-harbor", "tb2"),
        harbor_configs=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        cluster="marin",
    )

    batch = build_evaluation_batch(
        spec,
        LaunchProvenance(git_sha="abc", launch_host="host"),
        "tester",
    )

    assert batch.secret_env == {
        "DAYTONA_API_KEY": (
            "env:DAYTONA_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest",
        )
    }
    assert {evaluation.identity.eval_runtime for evaluation in batch.evaluations} == {HARBOR_RUNTIME}
    assert all(evaluation.identity.eval_ref.harbor.config_digest for evaluation in batch.evaluations)
    assert all(evaluation.identity.eval_ref.harbor.task_limit == 1 for evaluation in batch.evaluations)


def test_build_evaluation_batch_records_evalchemy_benchmark_extras(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("math500",),
        harbor_configs=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        cluster="marin",
    )

    batch = build_evaluation_batch(
        spec,
        LaunchProvenance(git_sha="abc", launch_host="host"),
        "tester",
    )

    assert batch.evaluations[0].identity.eval_runtime == EVALCHEMY.requirement(("math500",))


def test_build_evaluation_batch_rejects_conflicting_secret_specs(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    first = replace(
        EVALS["math500"],
        secret_env={"EVAL_TOKEN": ("env:FIRST_EVAL_TOKEN",)},
    )
    second = replace(
        EVALS["math500"],
        secret_env={"EVAL_TOKEN": ("env:SECOND_EVAL_TOKEN",)},
    )
    monkeypatch.setitem(EVALS, "secret-first", first)
    monkeypatch.setitem(EVALS, "secret-second", second)
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("secret-first", "secret-second"),
        harbor_configs=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        cluster="marin",
    )

    with pytest.raises(ValueError, match="conflicting secret specifications for EVAL_TOKEN"):
        build_evaluation_batch(
            spec,
            LaunchProvenance(git_sha="abc", launch_host="host"),
            "tester",
        )


def test_build_evaluation_batch_combines_registry_and_file_harbor_configs(tmp_path, monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    config_path = _write_harbor_config(tmp_path / "aime-policy.yaml")
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("mmlu-smoke",),
        harbor_configs=(HarborConfigSelection(name="aime-policy", path=config_path),),
        platform=Platform.TPU,
        accelerator=None,
        limit=2,
        records_prefix="memory://records",
        cluster="marin",
    )

    batch = build_evaluation_batch(
        spec,
        LaunchProvenance(git_sha="abc", launch_host="host"),
        "tester",
    )

    assert [evaluation.identity.eval_ref.name for evaluation in batch.evaluations] == [
        "mmlu-smoke",
        "aime-policy",
    ]
    evaluation = batch.evaluations[1]
    assert evaluation.identity.eval_ref.model_dump(mode="json", exclude_none=True) == {
        "name": "aime-policy",
        "mechanism": "harbor",
        "tasks": [],
        "harbor": {
            "dataset": "aime",
            "version": "1.0",
            "agent": "opencode",
            "env": "daytona",
            "task_limit": 2,
            "config_digest": evaluation.identity.eval_ref.harbor.config_digest,
        },
    }
    assert batch.secret_env == {
        "DAYTONA_API_KEY": (
            "env:DAYTONA_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest",
        )
    }

    captured: dict = {}

    def run_driver(config, overlay, driver_env) -> None:
        assert driver_env["DAYTONA_API_KEY"] == "daytona-key"
        captured["config"] = config
        captured["overlay"] = overlay
        trial_dir = Path(overlay.jobs_dir) / overlay.job_name / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text('{"task_name":"trial-one","verifier_result":{"rewards":{"reward":1}}}')

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    outcome = evaluation.executor(
        RunningModel(
            endpoint=OpenAIEndpoint(
                base_url="https://iris.example/capability/v1",
                model="served-qwen3-8b",
            )
        ),
        str(output_dir),
        {"DAYTONA_API_KEY": "daytona-key"},
    )

    assert outcome.metrics["aime"]["accuracy"] == 1.0
    assert captured["config"].stable_policy_json
    assert captured["overlay"].task_limit == 2
    assert captured["overlay"].served_model == "served-qwen3-8b"
    assert captured["overlay"].endpoint_url == "https://iris.example/capability/v1"
    assert captured["overlay"].model_agent_kwargs["extra_body"] == ('{"chat_template_kwargs":{"enable_thinking":true}}')


def test_launch_rejects_incompatible_harbor_config_before_iris_submission(tmp_path, monkeypatch):
    config_path = _write_harbor_config(
        tmp_path / "incompatible.yaml",
        agents="""  - name: terminus-2
  - name: opencode""",
    )
    iris_opened = False

    def reject_preflight(_requests):
        raise ValueError("Harbor config must declare exactly one agent")

    def open_iris_client(**_kwargs):
        nonlocal iris_opened
        iris_opened = True
        raise AssertionError("Iris must not be opened for an incompatible Harbor config")

    monkeypatch.setattr("experiments.evaluation.launch.preflight_harbor_configs", reject_preflight)
    monkeypatch.setattr("experiments.evaluation.cli.open_iris_client", open_iris_client)

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model",
            "qwen3-8b",
            "--harbor-config",
            str(config_path),
            "--no-wait",
        ],
    )

    assert result.exit_code == 2
    assert not iris_opened


def test_launch_accepts_registry_evals_and_repeated_harbor_configs(tmp_path, monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    first = _write_harbor_config(tmp_path / "first-policy.yaml")
    second = _write_harbor_config(tmp_path / "second-policy.yaml")
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model",
            "qwen3-8b",
            "--evals",
            "mmlu-smoke",
            "--harbor-config",
            str(first),
            "--harbor-config",
            str(second),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "eval=mmlu-smoke" in result.output
    assert "eval=first-policy" in result.output
    assert "eval=second-policy" in result.output
