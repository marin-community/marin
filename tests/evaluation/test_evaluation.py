# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the endpoint-oriented evaluation loop and durable records."""

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from iris.rpc import job_pb2
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
    run_evaluation_batch,
    submit_evaluation_batch,
)
from marin.evaluation.serving_config import BATCH_ENDPOINT_READY_TIMEOUT_SECONDS, ENDPOINT_READY_TIMEOUT_SECONDS
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
                    workspace_dataset_path=None,
                    agent="opencode",
                    environment="daytona",
                )
            )
        return tuple(configs)

    monkeypatch.setattr("experiments.evaluation.launch.preflight_harbor_configs", preflight)


def _write_harbor_config(path: Path) -> Path:
    path.write_text("{}")
    return path


def _successful_evaluation(
    session: RemoteInferenceSession,
    output_dir: str,
    _env_vars: Mapping[str, str],
) -> EvaluationOutcome:
    output = StoragePath(output_dir)
    output.mkdirs()
    (output / "endpoint.txt").write_text(session.model.endpoint.base_url)
    return EvaluationOutcome(metrics={"task": {"accuracy": 0.75}}, jobs={"eval": "/eval/success"})


def _failed_evaluation(
    _session: RemoteInferenceSession,
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


def _batch(
    root: Path,
    evaluations: tuple[Evaluation, ...],
    *,
    version: str | None = None,
    secret_env=None,
    priority_band: int = job_pb2.PRIORITY_BAND_INHERIT,
) -> EvaluationBatch:
    return EvaluationBatch(
        group_id="group",
        user="tester",
        version=version,
        description=None,
        records_prefix=str(root / "records"),
        model=ModelConfig(
            name="model",
            location="org/model",
            tokenizer="tokenizer",
            resource_hint=ResourceHint(hbm_gb=3),
        ),
        accelerator=AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-4", region="us-central1"),
        capability_origin="https://iris.example",
        api_model="model",
        evaluations=evaluations,
        provenance=LaunchProvenance(git_sha="abc", launch_host="host"),
        secret_env=secret_env or {},
        priority_band=priority_band,
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
        endpoint_name="/serve/test",
        endpoint_health_timeout_seconds=1800.0,
        streaming=True,
        tensor_parallel_size=1,
        backend_name="vllm",
    )
    batch = _batch(
        tmp_path,
        (
            _evaluation(tmp_path, "failure", _failed_evaluation),
            _evaluation(tmp_path, "success", _successful_evaluation),
        ),
        version="v1",
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
    batch = _batch(
        tmp_path,
        (evaluation,),
        secret_env={"DAYTONA_API_KEY": ("env:MARIN_TEST_EVAL_SECRET",)},
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
    )

    submit_evaluation_batch(batch, Client())

    assert captured["environment"].env_vars["DAYTONA_API_KEY"] == resolved_value
    assert captured["priority_band"] == job_pb2.PRIORITY_BAND_BATCH
    assert resolved_value.encode() not in captured["entrypoint"].workdir_files["_callable.pkl"]


@pytest.mark.parametrize(
    ("priority_band", "expected_timeout"),
    [
        (job_pb2.PRIORITY_BAND_BATCH, BATCH_ENDPOINT_READY_TIMEOUT_SECONDS),
        (job_pb2.PRIORITY_BAND_INTERACTIVE, ENDPOINT_READY_TIMEOUT_SECONDS),
    ],
)
def test_evaluation_endpoint_wait_distinguishes_batch_admission_from_model_startup(
    tmp_path, monkeypatch, priority_band, expected_timeout
):
    captured: dict[str, float] = {}

    def inference_config(*_args, endpoint_ready_timeout_seconds, **_kwargs):
        captured["endpoint_ready_timeout_seconds"] = endpoint_ready_timeout_seconds
        return object()

    @contextmanager
    def inference_session(_config):
        yield object()

    batch = _batch(
        tmp_path,
        (_evaluation(tmp_path, "success", _successful_evaluation),),
        priority_band=priority_band,
    )
    monkeypatch.setattr("marin.evaluation.runner.configure_coreweave_s3", lambda: None)
    monkeypatch.setattr("marin.evaluation.runner.iris_ctx", lambda: SimpleNamespace(job_id="/orchestrator"))
    monkeypatch.setattr("marin.evaluation.runner.env_vars_from_keys", lambda _keys: {})
    monkeypatch.setattr("marin.evaluation.runner.inference_config_for_model", inference_config)
    monkeypatch.setattr("marin.evaluation.runner.remote_inference", inference_session)
    monkeypatch.setattr("marin.evaluation.runner.evaluate_batch", lambda *_args, **_kwargs: [])

    assert run_evaluation_batch(batch) == []
    assert captured == {"endpoint_ready_timeout_seconds": expected_timeout}


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

    def run_driver(config, overlay, driver_env, _backend_state) -> None:
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
        RemoteInferenceSession(
            model=RunningModel(
                endpoint=OpenAIEndpoint(
                    base_url="https://iris.example/capability/v1",
                    model="served-qwen3-8b",
                )
            ),
            jobs=(),
            endpoint_name="/serve/test",
            endpoint_health_timeout_seconds=1800.0,
            streaming=True,
            tensor_parallel_size=1,
            backend_name="vllm",
        ),
        str(output_dir),
        {"DAYTONA_API_KEY": "daytona-key"},
    )

    assert outcome.metrics["aime"]["accuracy"] == 1.0
    assert captured["overlay"].task_limit == 2
    assert captured["overlay"].served_model == "served-qwen3-8b"
    assert captured["overlay"].endpoint_url == "https://iris.example/capability/v1"
    assert captured["overlay"].model_agent_kwargs["extra_body"] == ('{"chat_template_kwargs":{"enable_thinking":true}}')


def test_launch_rejects_incompatible_harbor_config_before_iris_submission(tmp_path, monkeypatch):
    config_path = _write_harbor_config(tmp_path / "incompatible.yaml")
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
            "--priority",
            "batch",
            "--target-cluster",
            "cw-rno2a",
            "--platform",
            "gpu",
            "--accelerator",
            "H100x1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "eval=mmlu-smoke" in result.output
    assert "eval=first-policy" in result.output
    assert "eval=second-policy" in result.output
    assert "priority: batch" in result.output
    assert "region_or_cluster=cw-rno2a" in result.output


def test_build_evaluation_batch_defaults_results_to_eval_root(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("mmlu-smoke",),
        harbor_configs=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix=None,
        cluster="marin",
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    assert batch.records_prefix == "gs://marin-eval-metadata/evals"
    evaluation = batch.evaluations[0]
    assert evaluation.identity.output_dir == f"{batch.records_prefix}/{evaluation.identity.run_id}/results"


def test_build_evaluation_batch_reuses_harbor_results_path(monkeypatch, tmp_path):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    monkeypatch.setattr("experiments.evaluation.launch.validate_harbor_resume_root", lambda *_args, **_kwargs: None)
    policy = _write_harbor_config(tmp_path / "policy.yaml")
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=(),
        harbor_configs=(HarborConfigSelection(name="policy", path=policy),),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=None,
        records_prefix="s3://eval-bucket/records",
        cluster="marin",
        resume_results_path="s3://eval-bucket/existing/results",
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    assert batch.evaluations[0].identity.output_dir == "s3://eval-bucket/existing/results"


def test_build_evaluation_batch_validates_resume_root_before_reuse(monkeypatch, tmp_path):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    policy = _write_harbor_config(tmp_path / "policy.yaml")
    captured: dict = {}

    def fake_validate(path, config):
        captured["path"] = path
        captured["dataset"] = config.record_dataset

    monkeypatch.setattr("experiments.evaluation.launch.validate_harbor_resume_root", fake_validate)
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=(),
        harbor_configs=(HarborConfigSelection(name="policy", path=policy),),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=None,
        records_prefix="s3://eval-bucket/records",
        cluster="marin",
        resume_results_path="s3://eval-bucket/existing/results",
    )

    build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    assert captured == {"path": "s3://eval-bucket/existing/results", "dataset": "aime"}


def test_build_evaluation_batch_propagates_resume_root_mismatch(monkeypatch, tmp_path):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    policy = _write_harbor_config(tmp_path / "policy.yaml")

    def reject(_path, _config):
        raise ValueError("dataset mismatch: found 'other', expected 'aime'")

    monkeypatch.setattr("experiments.evaluation.launch.validate_harbor_resume_root", reject)
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=(),
        harbor_configs=(HarborConfigSelection(name="policy", path=policy),),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=None,
        records_prefix="s3://eval-bucket/records",
        cluster="marin",
        resume_results_path="s3://eval-bucket/existing/results",
    )

    with pytest.raises(ValueError, match="dataset mismatch"):
        build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")
