# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the endpoint-oriented evaluation loop and durable records."""

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.rpc import job_pb2
from marin.evaluation.evalchemy.runner import EvalchemyExecutor, EvalchemyOutcome, EvalchemyRunConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.harbor.driver_config import HARBOR_RUNTIME, HarborDatasetKind, ValidatedHarborConfig
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint
from marin.evaluation.records import EvalRef, RunStatus, TaskCoverage, read_record
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
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.cli import cli, resolve_model_config
from experiments.evaluation.evals import EVALS, EvalchemyDefinition, HarborDefinition, resolve_eval_keys
from experiments.evaluation.launch import (
    LaunchSpec,
    build_evaluation_batch,
)
from experiments.evaluation.models import models


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


def _write_model_config(path: Path) -> Path:
    path.write_text(
        """\
name: fresh-rl-checkpoint
location: s3://marin-us-east-02a/marin/exports/rl/fresh-checkpoint/
tokenizer: Qwen/Qwen3-8B
resource_hint:
  gpu:
    H100: 8
serve:
  tensor_parallel_size: 1
  data_parallel_size: 8
  auto_overrides: false
"""
    )
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
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
        capability_origin="https://iris.example",
        api_model="model",
        evaluations=(
            _evaluation(tmp_path, "failure", _failed_evaluation),
            _evaluation(tmp_path, "success", _successful_evaluation),
        ),
        provenance=LaunchProvenance(git_sha="abc", launch_host="host"),
        submission_cluster="marin",
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
    assert succeeded.model.config is not None
    assert succeeded.model.config.model_dump(mode="json") == json.loads(json.dumps(asdict(batch.model)))
    assert (tmp_path / "success" / "endpoint.txt").read_text() == endpoint


def test_evalchemy_executor_classifies_archive_export_failure(tmp_path, monkeypatch):
    output_dir = str(StoragePath("memory://evalchemy-export-failure") / tmp_path.name)
    model_dir = StoragePath(output_dir) / "gsm8k_5shot" / "model"
    model_dir.mkdirs()
    (model_dir / "results_20260807.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match,flexible-extract": 0.75}}})
    )
    (model_dir / "samples_gsm8k_20260807.jsonl").write_text('{"unterminated": "sample\n')
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars: "/eval/completed",
    )
    session = RemoteInferenceSession(
        model=RunningModel(
            endpoint=OpenAIEndpoint(base_url="https://inference.example/v1", model="model"),
            tokenizer="tokenizer",
        ),
        jobs=(),
        endpoint_name="/serve/test",
        endpoint_health_timeout_seconds=1800.0,
        streaming=True,
        tensor_parallel_size=1,
        backend_name="vllm",
    )
    executor = EvalchemyExecutor(EvalchemyRunConfig(name="gsm8k", tasks=(EvalTaskConfig(name="gsm8k", num_fewshot=5),)))

    with pytest.raises(EvaluationError) as exc_info:
        executor(session, output_dir, {})

    assert exc_info.value.status is RunStatus.ARTIFACT_FAILED
    assert exc_info.value.jobs == {"eval": "/eval/completed"}


def test_evalchemy_executor_rejects_a_task_with_no_successful_responses(tmp_path, monkeypatch):
    coverage = {
        "humaneval_0shot": TaskCoverage(
            n_attempted=2,
            n_scored=0,
            n_unanswered=2,
            errors={"EVALCHEMY_INFRASTRUCTURE_ERROR": 2},
        )
    }
    outcome = EvalchemyOutcome(
        jobs={"eval": "/eval/completed"},
        result=SimpleNamespace(task_metrics=lambda: {"humaneval_0shot": {"pass@1,create_test": 0.0}}),
        coverage=coverage,
        recovered_metrics={"humaneval_0shot": {}},
    )
    monkeypatch.setattr("marin.evaluation.evalchemy.runner.run_evalchemy", lambda *_args, **_kwargs: outcome)
    session = RemoteInferenceSession(
        model=RunningModel(
            endpoint=OpenAIEndpoint(base_url="https://inference.example/v1", model="model"),
            tokenizer="tokenizer",
        ),
        jobs=(),
        endpoint_name="/serve/test",
        endpoint_health_timeout_seconds=1800.0,
        streaming=True,
        tensor_parallel_size=1,
        backend_name="vllm",
    )
    executor = EvalchemyExecutor(
        EvalchemyRunConfig(name="humaneval", tasks=(EvalTaskConfig(name="humaneval", num_fewshot=0),))
    )

    with pytest.raises(EvaluationError) as exc_info:
        executor(session, str(tmp_path), {})

    assert exc_info.value.status is RunStatus.INFRA_FAILED
    assert exc_info.value.coverage == coverage


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
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
        capability_origin="https://iris.example",
        api_model="model",
        evaluations=(evaluation,),
        provenance=LaunchProvenance(git_sha="abc", launch_host="host"),
        submission_cluster="marin",
        secret_env={"DAYTONA_API_KEY": ("env:MARIN_TEST_EVAL_SECRET",)},
    )

    submit_evaluation_batch(batch, Client())

    assert captured["environment"].env_vars["DAYTONA_API_KEY"] == resolved_value
    assert resolved_value.encode() not in captured["entrypoint"].workdir_files["_callable.pkl"]


@pytest.mark.parametrize(
    ("submission_cluster", "expects_federation"),
    (("cw-us-east-08a", False), ("marin", True)),
)
def test_submit_evaluation_batch_only_federates_to_a_different_cluster(tmp_path, submission_cluster, expects_federation):
    captured: dict = {}

    class Client:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(job_id="/eval/job")

    batch = EvaluationBatch(
        group_id="group",
        user="tester",
        version=None,
        description=None,
        records_prefix=str(tmp_path / "records"),
        model=ModelConfig(name="model", location="org/model", tokenizer="tokenizer"),
        accelerator=AcceleratorChoice(
            platform=Platform.GPU,
            gpu_type="GB200",
            gpu_count=1,
            target_cluster="cw-us-east-08a",
        ),
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
        capability_origin="https://iris.example",
        api_model="model",
        evaluations=(_evaluation(tmp_path, "eval", _successful_evaluation),),
        provenance=LaunchProvenance(git_sha="abc", launch_host="host"),
        submission_cluster=submission_cluster,
    )

    submit_evaluation_batch(batch, Client())

    constraints = captured["constraints"]
    if expects_federation:
        assert constraints[0].key == "cluster"
    else:
        assert constraints is None


def test_submit_evaluation_batch_uses_resolved_federated_cluster_and_priority(monkeypatch):
    captured: dict = {}

    class Client:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(job_id="/eval/job")

    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=models()["qwen3-32b"],
        evals=("mmlu-smoke",),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.GPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")
    submit_evaluation_batch(batch, Client())

    assert captured["constraints"] == [
        Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value="cw-rno2a")
    ]
    assert captured["priority_band"] == job_pb2.PRIORITY_BAND_INTERACTIVE


def test_build_evaluation_batch_uses_submission_cluster_for_direct_endpoint(monkeypatch):
    monkeypatch.setattr(
        "experiments.evaluation.launch._capability_origin",
        lambda cluster: f"https://{cluster}.example",
    )
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=("mmlu-smoke",),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        submission_cluster="custom-controller",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    assert batch.capability_origin == "https://custom-controller.example"


def test_build_evaluation_batch_merges_the_shared_daytona_spec(monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=("aime-harbor", "tb2"),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
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


def test_resolve_eval_keys_validates_programmatic_selections() -> None:
    assert resolve_eval_keys("gsm8k-smoke,aime-smoke") == ("gsm8k-smoke", "aime-smoke")
    with pytest.raises(ValueError):
        resolve_eval_keys("gsm8k-smoke,missing")


def test_build_evaluation_batch_records_evalchemy_benchmark_extras(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=("math500",),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    batch = build_evaluation_batch(
        spec,
        LaunchProvenance(git_sha="abc", launch_host="host"),
        "tester",
    )

    assert batch.evaluations[0].identity.eval_runtime == EVALCHEMY.requirement(("math500",))


def test_file_evalchemy_chat_template_overrides_model_default(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    definition = EvalchemyDefinition(
        name="ifeval",
        config_path=Path("experiments/evaluation/configs/evalchemy/ifeval.yaml"),
    )
    spec = LaunchSpec(
        model=models()["llama-3.1-8b-base"],
        evals=(),
        evalchemy_definitions=(definition,),
        harbor_definitions=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    batch = build_evaluation_batch(
        spec,
        LaunchProvenance(git_sha="abc", launch_host="host"),
        "tester",
    )

    evalchemy = batch.evaluations[0].identity.eval_ref.evalchemy
    assert evalchemy is not None
    assert evalchemy.apply_chat_template is True


@pytest.mark.parametrize(
    ("benchmark_limit", "model_limit", "expected_limit", "expected_warnings"),
    [
        (128, 8192, 128, 1),
        (8192, 2048, 2048, 0),
        (None, 8192, 8192, 0),
    ],
)
def test_evalchemy_generation_budget_preserves_benchmark_protocol(
    tmp_path,
    monkeypatch,
    caplog,
    benchmark_limit,
    model_limit,
    expected_limit,
    expected_warnings,
):
    config_path = tmp_path / "generation.yaml"
    max_tokens = "" if benchmark_limit is None else f"max_tokens: {benchmark_limit}\n"
    config_path.write_text(f"tasks: [triviaqa]\n{max_tokens}")
    model = replace(models()["qwen3-8b"], generation=GenerationConfig(max_gen_toks=model_limit))
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    caplog.set_level(logging.WARNING, logger="experiments.evaluation.evals")
    spec = LaunchSpec(
        model=model,
        evals=(),
        evalchemy_definitions=(EvalchemyDefinition(name="generation", config_path=config_path),),
        harbor_definitions=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    evalchemy = batch.evaluations[0].identity.eval_ref.evalchemy
    assert evalchemy is not None
    assert evalchemy.max_gen_toks == expected_limit
    warnings = [record for record in caplog.records if record.name == "experiments.evaluation.evals"]
    assert len(warnings) == expected_warnings


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
        model=models()["qwen3-8b"],
        evals=("secret-first", "secret-second"),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    with pytest.raises(ValueError, match="conflicting secret specifications for EVAL_TOKEN"):
        build_evaluation_batch(
            spec,
            LaunchProvenance(git_sha="abc", launch_host="host"),
            "tester",
        )


def test_build_evaluation_batch_combines_registry_evalchemy_and_harbor_configs(tmp_path, monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    evalchemy_config_path = tmp_path / "ifeval.yaml"
    evalchemy_config_path.write_text(Path("experiments/evaluation/configs/evalchemy/ifeval.yaml").read_text())
    config_path = _write_harbor_config(tmp_path / "aime-policy.yaml")
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=("mmlu-smoke",),
        evalchemy_definitions=(EvalchemyDefinition(name="ifeval", config_path=evalchemy_config_path),),
        harbor_definitions=(HarborDefinition(name="aime-policy", config_path=config_path),),
        platform=Platform.TPU,
        accelerator=None,
        limit=2,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    batch = build_evaluation_batch(
        spec,
        LaunchProvenance(git_sha="abc", launch_host="host"),
        "tester",
    )

    assert [evaluation.identity.eval_ref.name for evaluation in batch.evaluations] == [
        "mmlu-smoke",
        "ifeval",
        "aime-policy",
    ]
    ifeval = batch.evaluations[1].identity.eval_ref
    assert batch.evaluations[1].identity.eval_runtime == EVALCHEMY.requirement(("ifeval",))
    assert ifeval.model_dump(mode="json", exclude_none=True) == {
        "name": "ifeval",
        "mechanism": "evalchemy",
        "tasks": [
            {
                "name": "ifeval",
                "num_fewshot": 0,
                "task_alias": "ifeval_0shot",
                "generation": True,
                "unsafe_code": False,
                "completion_only": False,
            }
        ],
        "evalchemy": {
            "apply_chat_template": True,
            "max_gen_toks": 2048,
            "max_eval_instances": 2,
            "num_concurrent": 16,
            "batch_size": "1",
            "seed": 1234,
            "extra_gen_kwargs": {},
            "extra_model_args": {},
        },
    }

    evaluation = batch.evaluations[2]
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


@pytest.mark.parametrize(
    ("overrides", "target_cluster", "priority"),
    [
        ((), "cw-us-east-02a", "inherit"),
        (("--federated_cluster", "cw-rno2a", "--priority", "interactive"), "cw-rno2a", "interactive"),
    ],
)
def test_launch_dry_run_prints_resolved_federated_cluster_and_priority(
    overrides,
    target_cluster,
    priority,
    monkeypatch,
):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model",
            "qwen3-32b",
            "--evals",
            "mmlu-smoke",
            *overrides,
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "controller_cluster=marin" in result.output
    assert f"target_cluster={target_cluster}" in result.output
    assert f"priority={priority}" in result.output


def test_launch_dry_run_accepts_file_backed_model_config(tmp_path, monkeypatch):
    config_path = _write_model_config(tmp_path / "fresh-checkpoint.yaml")
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")

    model = resolve_model_config(None, config_path)
    assert model.name == "fresh-rl-checkpoint"
    assert model.location == "s3://marin-us-east-02a/marin/exports/rl/fresh-checkpoint/"
    assert model.resource_hint.gpu == {"H100": 8}
    assert model.serve.data_parallel_size == 8

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model-config",
            str(config_path),
            "--evals",
            "mmlu-smoke",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output


def test_resolve_model_config_rejects_registry_and_file_selectors_together(tmp_path):
    config_path = _write_model_config(tmp_path / "fresh-checkpoint.yaml")

    with pytest.raises(click.BadParameter):
        resolve_model_config("qwen3-8b", config_path)


def test_resolve_model_config_requires_one_selector():
    with pytest.raises(click.BadParameter):
        resolve_model_config(None, None)


def test_launch_rejects_invalid_harbor_config_before_iris_submission(tmp_path, monkeypatch):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("{}")
    iris_opened = False
    error = "Harbor config must declare exactly one agent"

    def reject_preflight(_requests):
        raise ValueError(error)

    def open_iris_client(**_kwargs):
        nonlocal iris_opened
        iris_opened = True
        raise AssertionError("Iris must not be opened for an invalid evaluator config")

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


def test_launch_rejects_malformed_evalchemy_yaml_before_iris_submission(tmp_path, monkeypatch):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("tasks: ifeval\n")
    iris_opened = False

    def open_iris_client(**_kwargs):
        nonlocal iris_opened
        iris_opened = True
        raise AssertionError("Iris must not be opened for a malformed evaluator config")

    monkeypatch.setattr("experiments.evaluation.cli.open_iris_client", open_iris_client)

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model",
            "qwen3-8b",
            "--evalchemy-config",
            str(config_path),
            "--no-wait",
        ],
    )

    assert result.exit_code == 2
    assert not iris_opened


def test_launch_defers_evalchemy_task_validation_to_external_cli(tmp_path, monkeypatch):
    config_path = tmp_path / "external-task.yaml"
    config_path.write_text("tasks: [task_added_after_marin_release]\n")
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model",
            "qwen3-8b",
            "--evalchemy-config",
            str(config_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "eval=external-task" in result.output


def test_launch_accepts_registry_ifeval_and_repeated_harbor_configs(tmp_path, monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    ifeval = Path("experiments/evaluation/configs/evalchemy/ifeval.yaml")
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
            "--evalchemy-config",
            str(ifeval),
            "--harbor-config",
            str(first),
            "--harbor-config",
            str(second),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "eval=mmlu-smoke" in result.output
    assert "eval=ifeval" in result.output
    assert "eval=first-policy" in result.output
    assert "eval=second-policy" in result.output


def test_build_evaluation_batch_defaults_results_to_eval_root(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=("mmlu-smoke",),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix=None,
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    assert batch.records_prefix == "gs://marin-eval-metadata/evals"
    evaluation = batch.evaluations[0]
    assert evaluation.identity.output_dir == f"{batch.records_prefix}/{evaluation.identity.run_id}/results"
