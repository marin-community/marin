# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the endpoint-oriented evaluation loop and durable records."""

import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner
from finestore.eval import (
    ARCHIVE_ROLLOUTS_TABLE,
    ARCHIVE_SAMPLES_TABLE,
    EvaluationStore,
    SampleKind,
    sample_from_archive_row,
)
from finestore.reader import ReadView
from finestore.store import DataStore
from fray.types import JobStatus
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.resources.state import JobState, TaskState
from iris.rpc import job_pb2
from marin.evaluation.eval_policy import SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION, source_config_digest
from marin.evaluation.evalchemy.config import EvalchemyJudgeConfig
from marin.evaluation.evalchemy.result import FineStoreEvalchemyResult
from marin.evaluation.evalchemy.runner import (
    EvalchemyExecutor,
    EvalchemyRunConfig,
    _coverage_with_aggregate_counts,
    run_evalchemy,
)
from marin.evaluation.evalchemy.runtime import evalchemy_runtime_descriptor, evalchemy_runtime_source_hash
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.harbor.driver_config import (
    HARBOR_SOURCE,
    HarborDatasetKind,
    HarborErrorTaxonomy,
    ValidatedHarborConfig,
    harbor_runtime_descriptor,
)
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.lm_eval_samples import samples_from_lm_eval
from marin.evaluation.model_config import GenerationConfig, ModelConfig, ResourceHint, ServeConfig
from marin.evaluation.records import (
    EVALCHEMY_INFRASTRUCTURE_ERROR,
    BenchmarkMetadataRef,
    BenchmarkMetricRef,
    EvalRef,
    MetricKind,
    RunStatus,
    TaskCoverage,
    read_record,
    record_path,
)
from marin.evaluation.runner import (
    EndpointRoute,
    Evaluation,
    EvaluationBatch,
    EvaluationError,
    EvaluationIdentity,
    EvaluationOutcome,
    HostedJudge,
    InferenceBackendFinished,
    InferenceUnavailable,
    LaunchProvenance,
    run_with_inference_recovery,
)
from marin.evaluation.serving_config import inference_config_for_model
from marin.execution.lazy import ArtifactStep, resolve
from marin.execution.remote import remote
from marin.execution.step_status import STATUS_FAILED, STATUS_SUCCESS, StatusFile
from marin.experiment.evaluation import EvalchemyEvalConfig, run_served_evalchemy
from marin.inference.config import (
    EffectiveServing,
    ResolvedModelLocator,
    SpeculativeMethod,
    SpeculativeServingConfig,
)
from marin.inference.iris import RemoteInferenceSession, RemoteInferenceStartupError
from marin.inference.types import InferenceBackendState, OpenAIEndpoint, RunningModel
from prometheus_client.parser import text_string_to_metric_families
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.cli import cli, resolve_model_config
from experiments.evaluation.evals import (
    EVALS,
    EvalchemyDefinition,
    HarborDefinition,
    resolve_eval_keys,
)
from experiments.evaluation.launch import (
    LaunchSpec,
    build_evaluation_batch,
    launch_group,
    run_evaluation_group,
)
from experiments.evaluation.models import models
from experiments.evaluation.pipeline import (
    EvalStepConfig,
    run_eval_pipeline_step,
)

# Stand-in agent limits the fake driver reports back. They match neither Harbor's defaults nor any
# model in these tests, so an assertion on them can only be satisfied by the preflight result.
_PREFLIGHT_MAX_INPUT_TOKENS = 262144
_PREFLIGHT_MAX_OUTPUT_TOKENS = 65536


def _install_fake_harbor_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    verifier_env_keys: tuple[str, ...] = (),
) -> list[Mapping[str, object]]:
    received: list[Mapping[str, object]] = []

    def preflight(requests):
        configs = []
        for path, model_agent_kwargs in requests:
            received.append(model_agent_kwargs)
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
                    verifier_env_keys=verifier_env_keys,
                    error_taxonomy=HarborErrorTaxonomy(
                        infrastructure=frozenset({"InfrastructureError"}),
                        agent=frozenset({"AgentError"}),
                        passthrough=frozenset({"PassthroughError"}),
                        undecided=frozenset({"VerifierTimeoutError"}),
                        commit=HARBOR_SOURCE.commit,
                    ),
                    max_input_tokens=_PREFLIGHT_MAX_INPUT_TOKENS,
                    max_output_tokens=_PREFLIGHT_MAX_OUTPUT_TOKENS,
                    benchmark=BenchmarkMetadataRef(
                        schema_version=1,
                        task="aime",
                        primary_metric="reward",
                        metric_kind=MetricKind.CONTINUOUS,
                        metrics=(
                            BenchmarkMetricRef(
                                name="reward",
                                source_name="reward",
                                kind=MetricKind.CONTINUOUS,
                                higher_is_better=True,
                            ),
                        ),
                        n_benchmark=1,
                        n_attempted=1,
                    ),
                    trials_per_task=1,
                )
            )
        return tuple(configs)

    monkeypatch.setattr("experiments.evaluation.launch.preflight_harbor_configs", preflight)
    return received


@pytest.mark.parametrize("version", [SEPTEMBER_16_VERSION, SEPTEMBER_24_VERSION])
def test_verified_harbor_launch_rejects_vendored_runtime(monkeypatch, version):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=ModelConfig(name="test-model", location="org/test-model"),
        evals=(),
        evalchemy_definitions=(),
        harbor_definitions=(
            HarborDefinition(
                name="ot-tblite-recovery",
                config_path=Path("experiments/evaluation/configs/harbor/ot-tblite-recovery.yaml"),
            ),
        ),
        platform=Platform.GPU,
        accelerator="H100x8",
        limit=None,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
        version=version,
    )

    with pytest.raises(ValueError, match="evaluator runtime must use commit"):
        build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="test"), "test")


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
    *,
    backend_state: Callable[[], InferenceBackendState],
    judge: RemoteInferenceSession | None = None,
) -> EvaluationOutcome:
    output = StoragePath(output_dir)
    output.mkdirs()
    (output / "endpoint.txt").write_text(session.model.endpoint.base_url)
    return EvaluationOutcome(metrics={"task": {"accuracy": 0.75}}, jobs={"eval": "/eval/success"})


def _failed_evaluation(
    _session: RemoteInferenceSession,
    _output_dir: str,
    _env_vars: Mapping[str, str],
    *,
    backend_state: Callable[[], InferenceBackendState],
    judge: RemoteInferenceSession | None = None,
) -> EvaluationOutcome:
    raise EvaluationError(
        "evaluation failed",
        status=RunStatus.FAILED,
        jobs={"eval": "/eval/failure"},
        log_tails={"eval": ("failure detail",)},
    )


def _evaluation(root: Path, name: str, executor, endpoint_route: EndpointRoute = EndpointRoute.CAPABILITY) -> Evaluation:
    return Evaluation(
        identity=EvaluationIdentity(
            run_id=f"run-{name}",
            created_at="2026-07-24T00:00:00+00:00",
            output_dir=str(root / name),
            eval_ref=EvalRef(name=name, mechanism="test"),
            eval_runtime="test-runtime",
        ),
        executor=executor,
        endpoint_route=endpoint_route,
    )


def _remote_session(endpoint: str = "https://inference.example/v1") -> RemoteInferenceSession:
    return RemoteInferenceSession(
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


def _patch_inference_runtime(monkeypatch: pytest.MonkeyPatch, remote) -> None:
    """Point the batch runner at a fake inference runtime with ``remote`` as its session factory."""
    monkeypatch.setattr("marin.evaluation.runner.configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(
        "marin.evaluation.runner.iris_ctx",
        lambda: SimpleNamespace(
            job_id="/orchestrator",
            client=SimpleNamespace(resolve_endpoint=lambda _name: "http://10.0.0.1:8000"),
        ),
    )
    monkeypatch.setattr("marin.evaluation.runner.remote_inference", remote)
    monkeypatch.setattr(
        "marin.evaluation.runner.inference_config_for_model",
        lambda model, *_args, **_kwargs: SimpleNamespace(model=SimpleNamespace(model_id=model.name)),
    )


def _hosted_judge_batch(tmp_path, evaluations: tuple[Evaluation, ...]) -> EvaluationBatch:
    """One GPU batch with a co-hosted judge session beside the candidate model."""
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=1, target_cluster="cw-rno2a")
    return EvaluationBatch(
        group_id="group",
        user="tester",
        version=None,
        description=None,
        records_prefix=str(tmp_path / "records"),
        model=ModelConfig(name="candidate", location="org/candidate", resource_hint=ResourceHint(hbm_gb=3)),
        accelerator=accelerator,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
        capability_origin="https://iris.example",
        api_model="candidate",
        evaluations=evaluations,
        provenance=LaunchProvenance(git_sha="abc", launch_host="host"),
        submission_cluster="marin",
        max_concurrent=8,
        judge=HostedJudge(
            model=ModelConfig(name="judge", location="org/judge", resource_hint=ResourceHint(hbm_gb=3)),
            accelerator=accelerator,
            api_model="judge",
        ),
    )


def _run_test_batch(batch: EvaluationBatch) -> list[str]:
    run_evaluation_group(batch)
    return [record_path(batch.records_prefix, evaluation.identity.run_id) for evaluation in batch.evaluations]


def _step_status(batch: EvaluationBatch, index: int) -> str | None:
    return StatusFile(
        str(Path(batch.records_prefix) / batch.evaluations[index].identity.run_id), worker_id="test"
    ).status


def _evaluate_with_session(
    batch: EvaluationBatch, session: RemoteInferenceSession, orchestrator_job_id: str
) -> list[str]:
    context = SimpleNamespace(
        job_id=orchestrator_job_id,
        client=SimpleNamespace(resolve_endpoint=lambda _name: "http://10.0.0.1:8000"),
    )
    with (
        patch("marin.evaluation.runner.configure_coreweave_s3", lambda: None),
        patch("marin.evaluation.runner.iris_ctx", lambda: context),
        patch("marin.evaluation.runner.remote_inference", lambda _config: nullcontext(session)),
        patch(
            "marin.evaluation.runner.inference_config_for_model",
            lambda model, *_args, **_kwargs: SimpleNamespace(model=SimpleNamespace(model_id=model.name)),
        ),
    ):
        return _run_test_batch(batch)


def test_evaluation_steps_share_candidate_and_judge(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "stepspec-test-secret-do-not-persist")
    opened_models: list[str] = []
    observed_candidates: list[RemoteInferenceSession] = []
    observed_judges: list[RemoteInferenceSession | None] = []

    class InferenceContext:
        def __init__(self, session: RemoteInferenceSession):
            self.session = session

        def __enter__(self) -> RemoteInferenceSession:
            return self.session

        def __exit__(self, *_args) -> None:
            return None

    def remote(config):
        model = config.model.model_id
        opened_models.append(model)
        return InferenceContext(_remote_session(f"https://{model}.example/v1"))

    def executor(
        session: RemoteInferenceSession,
        _output_dir: str,
        _env_vars: Mapping[str, str],
        *,
        backend_state: Callable[[], InferenceBackendState],
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome:
        observed_candidates.append(session)
        observed_judges.append(judge)
        return EvaluationOutcome(metrics={"task": {"accuracy": 1.0}})

    _patch_inference_runtime(monkeypatch, remote)
    batch = _hosted_judge_batch(
        tmp_path,
        (_evaluation(tmp_path, "one", executor), _evaluation(tmp_path, "two", executor)),
    )

    _run_test_batch(batch)

    assert opened_models == ["candidate", "judge"]
    assert all(candidate.model.endpoint.base_url == "https://candidate.example/v1" for candidate in observed_candidates)
    assert observed_candidates[0] is observed_candidates[1]
    assert len(observed_judges) == 2
    assert observed_judges[0] is observed_judges[1]
    assert observed_judges[0] is not None
    assert observed_judges[0].model.endpoint.base_url == "https://judge.example/v1"
    record = read_record(str(tmp_path / "records" / "run-one" / "record.json"))
    assert record.judge is not None
    assert record.judge.model.name == "judge"
    assert record.judge.hardware.accelerator == "H100x1"
    for evaluation in batch.evaluations:
        metadata = json.loads((tmp_path / "records" / evaluation.identity.run_id / ".executor_info").read_text())
        assert "https://candidate.example/v1" not in metadata["fn_name"]
        assert "https://judge.example/v1" not in metadata["fn_name"]
        assert "stepspec-test-secret-do-not-persist" not in metadata["fn_name"]


def test_evaluation_restart_skips_completed_step_without_serving(tmp_path, monkeypatch):
    evaluation = _evaluation(tmp_path, "finished", _successful_evaluation)
    batch = replace(_hosted_judge_batch(tmp_path, (evaluation,)), judge=None)
    monkeypatch.setattr("marin.evaluation.runner.record_rollout_run", lambda _record: None)
    path = _evaluate_with_session(batch, _remote_session(), orchestrator_job_id="/first")[0]
    original = Path(path).read_bytes()

    _patch_inference_runtime(monkeypatch, lambda _config: pytest.fail("completed eval started serving again"))

    assert _run_test_batch(batch) == [path]
    assert Path(path).read_bytes() == original
    assert _step_status(batch, 0) == STATUS_SUCCESS


def test_evaluation_restart_only_runs_unfinished_evals(tmp_path, monkeypatch):
    executed: list[str] = []

    def executor(_session, output_dir, _env_vars, *, backend_state, judge=None):
        executed.append(output_dir)
        return EvaluationOutcome(metrics={"task": {"accuracy": 0.5}})

    finished = _evaluation(tmp_path, "finished", executor)
    pending = _evaluation(tmp_path, "pending", executor)
    batch = replace(_hosted_judge_batch(tmp_path, (finished, pending)), judge=None)
    monkeypatch.setattr("marin.evaluation.runner.record_rollout_run", lambda _record: None)
    _evaluate_with_session(replace(batch, evaluations=(finished,)), _remote_session(), orchestrator_job_id="/first")
    executed.clear()
    _patch_inference_runtime(monkeypatch, lambda _config: nullcontext(_remote_session()))

    paths = _run_test_batch(batch)

    assert executed == [pending.identity.output_dir]
    assert paths == [
        str(tmp_path / "records" / "run-finished" / "record.json"),
        str(tmp_path / "records" / "run-pending" / "record.json"),
    ]
    assert read_record(paths[0]).jobs["orchestrator"] == "/first"
    assert read_record(paths[1]).jobs["orchestrator"] == "/orchestrator"
    assert _step_status(batch, 0) == STATUS_SUCCESS
    assert _step_status(batch, 1) == STATUS_SUCCESS


def test_evaluation_steps_refresh_direct_endpoint_between_evaluations(tmp_path, monkeypatch):
    observed_urls: list[str] = []
    addresses = iter(("http://10.0.0.1:8000", "http://10.0.0.2:8000"))

    def executor(
        session: RemoteInferenceSession,
        _output_dir: str,
        _env_vars: Mapping[str, str],
        *,
        backend_state: Callable[[], InferenceBackendState],
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome:
        observed_urls.append(session.model.endpoint.base_url)
        return EvaluationOutcome(metrics={"task": {"accuracy": 1.0}})

    _patch_inference_runtime(
        monkeypatch,
        lambda _config: nullcontext(_remote_session()),
    )
    monkeypatch.setattr(
        "marin.evaluation.runner.iris_ctx",
        lambda: SimpleNamespace(
            job_id="/orchestrator",
            client=SimpleNamespace(resolve_endpoint=lambda _name: next(addresses)),
        ),
    )
    batch = replace(
        _hosted_judge_batch(
            tmp_path,
            (
                _evaluation(tmp_path, "one", executor, EndpointRoute.DIRECT),
                _evaluation(tmp_path, "two", executor, EndpointRoute.DIRECT),
            ),
        ),
        judge=None,
    )

    _run_test_batch(batch)

    assert set(observed_urls) == {"http://10.0.0.1:8000/v1", "http://10.0.0.2:8000/v1"}


def test_evaluation_steps_record_each_hosted_judge_startup_failure(tmp_path, monkeypatch):
    class InferenceContext:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def __enter__(self) -> RemoteInferenceSession:
            if self.model_name == "judge":
                raise RemoteInferenceStartupError("judge did not become ready", jobs=())
            return _remote_session("https://candidate.example/v1")

        def __exit__(self, *_args) -> None:
            return None

    evaluations = (
        _evaluation(tmp_path, "one", _successful_evaluation),
        _evaluation(tmp_path, "two", _successful_evaluation),
    )
    batch = _hosted_judge_batch(tmp_path, evaluations)
    _patch_inference_runtime(monkeypatch, lambda config: InferenceContext(config.model.model_id))

    with pytest.raises(RuntimeError, match=r"1 step\(s\) failed"):
        _run_test_batch(batch)

    for index, evaluation in enumerate(evaluations):
        record = read_record(str(tmp_path / "records" / evaluation.identity.run_id / "record.json"))
        assert record.status is RunStatus.INFRA_FAILED
        assert _step_status(batch, index) == STATUS_FAILED
        assert record.jobs == {"orchestrator": "/orchestrator"}
        assert "judge did not become ready" in (record.error or "")

    _patch_inference_runtime(monkeypatch, lambda _config: nullcontext(_remote_session()))
    monkeypatch.setattr("marin.evaluation.runner.record_rollout_run", lambda _record: None)
    _run_test_batch(batch)

    for index, evaluation in enumerate(evaluations):
        record = read_record(str(tmp_path / "records" / evaluation.identity.run_id / "record.json"))
        assert record.status is RunStatus.SUCCEEDED
        assert _step_status(batch, index) == STATUS_SUCCESS


def _lm_eval_generation(doc_id: int, metric: str, score: float, response: str) -> dict:
    return {
        "doc_id": doc_id,
        "doc": {"question": "2+2?"},
        "target": "4",
        "arguments": [["Question: 2+2?"]],
        "resps": [[response]],
        "filtered_resps": [response],
        "filter": "none",
        "metrics": [metric],
        metric: score,
        "schema_version": 1,
    }


def _write_evalchemy_output(
    output_dir: str,
    task_dir: str,
    results: dict[str, dict[str, float]],
    samples: dict[str, list[dict]],
    *,
    scored_counts: Mapping[str, int] | None = None,
    failure_category: str | None = None,
) -> None:
    sample_counts = {task: max((int(row["doc_id"]) for row in rows), default=-1) + 1 for task, rows in samples.items()}
    benchmark_metadata = {}
    canonical_results = {}
    primary_sources = {}
    for task, count in sample_counts.items():
        source_name = next(name.split(",", 1)[0] for name in results[task] if "stderr" not in name)
        canonical_name = "accuracy" if source_name in {"acc", "exact_match", "accuracy_avg"} else source_name
        primary_sources[task] = source_name
        benchmark_metadata[task] = {
            "schema_version": 1,
            "task": task,
            "primary_metric": canonical_name,
            "metric_kind": "binary",
            "metrics": [
                {
                    "name": canonical_name,
                    "source_name": source_name,
                    "kind": "binary",
                    "higher_is_better": True,
                }
            ],
            "n_benchmark": count,
            "n_attempted": count,
        }
        canonical_results[task] = {canonical_name: next(iter(results[task].values()))}

    data_store = DataStore.open(output_dir, writer_id="evalchemy-test")
    store = EvaluationStore(data_store)
    try:
        data_store.table("evalchemy_results", primary_key=("source_prefix",)).append(
            {
                "source_prefix": task_dir,
                "document": json.dumps(
                    {
                        "results": results,
                        "benchmark_metadata": benchmark_metadata,
                        "canonical_results": canonical_results,
                        "task_outcomes": {
                            task: {
                                "status": "failed" if failure_category else "succeeded",
                                "scored_count": (scored_counts or {}).get(task, count),
                                "failure": {"category": failure_category},
                            }
                            for task, count in sample_counts.items()
                        },
                    }
                ),
                "scored": failure_category is None,
            }
        )
        for task, rows in samples.items():
            normalized_task = task_dir if len(samples) == 1 else f"{task_dir}/{task}"
            for row in rows:
                for sample in samples_from_lm_eval(normalized_task, row, primary_sources[task]):
                    store.add_sample(sample)
        store.seal()
    finally:
        store.close()


def test_evaluation_steps_record_failures_without_blocking_other_steps(tmp_path, monkeypatch):
    records = tmp_path / "records"
    endpoint = "https://iris.example/proxy/t/token/inference/v1"
    session = _remote_session(endpoint)
    session = replace(session, effective_serving=EffectiveServing(2, 1, 1, 1, 4096))
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
        max_concurrent=8,
    )
    catalog_rows = []
    monkeypatch.setattr("marin.evaluation.runner.record_rollout_run", catalog_rows.append)

    with pytest.raises(RuntimeError, match="1 step"):
        _evaluate_with_session(batch, session, orchestrator_job_id="/orchestrator")

    failed = read_record(str(records / "run-failure" / "record.json"))
    succeeded = read_record(str(records / "run-success" / "record.json"))
    assert failed.status is RunStatus.FAILED
    assert _step_status(batch, 0) == STATUS_FAILED
    assert failed.jobs == {"orchestrator": "/orchestrator", "eval": "/eval/failure"}
    assert failed.log_tails == {"eval": ("failure detail",)}
    assert succeeded.status is RunStatus.SUCCEEDED
    assert _step_status(batch, 1) == STATUS_SUCCESS
    assert succeeded.metrics == {"task": {"accuracy": 0.75}}
    assert succeeded.serving is not None
    assert succeeded.serving.effective
    assert succeeded.serving.tensor_parallel_size == 2
    assert succeeded.serving.max_model_len == 4096
    assert failed.serving == succeeded.serving
    assert succeeded.provenance.eval_runtime == "test-runtime"
    assert succeeded.model.config is not None
    assert succeeded.model.config.model_dump(mode="json") == json.loads(json.dumps(asdict(batch.model)))
    assert (tmp_path / "success" / "endpoint.txt").read_text() == endpoint
    assert {(row.run_id, row.status) for row in catalog_rows} == {
        ("run-failure", "failed"),
        ("run-success", "succeeded"),
    }
    assert all(row.storage_format == "finestore" for row in catalog_rows)


@pytest.mark.parametrize("n_scored, expected_status", [(9, RunStatus.SUCCEEDED), (8, RunStatus.INFRA_FAILED)])
def test_evaluation_step_gates_transport_failure_coverage(tmp_path, monkeypatch, n_scored, expected_status):
    def executor(
        _session: RemoteInferenceSession,
        _output_dir: str,
        _env_vars: Mapping[str, str],
        *,
        backend_state: Callable[[], InferenceBackendState],
        judge: RemoteInferenceSession | None = None,
    ) -> EvaluationOutcome:
        return EvaluationOutcome(
            metrics={"math500": {"accuracy,none": 1.0}},
            coverage={
                "math500": TaskCoverage(
                    n_attempted=10,
                    n_scored=n_scored,
                    errors={EVALCHEMY_INFRASTRUCTURE_ERROR: 10 - n_scored},
                )
            },
        )

    batch = replace(_hosted_judge_batch(tmp_path, (_evaluation(tmp_path, "math500", executor),)), judge=None)
    monkeypatch.setattr("marin.evaluation.runner.record_rollout_run", lambda _record: None)

    if expected_status is RunStatus.INFRA_FAILED:
        with pytest.raises(RuntimeError, match="1 step"):
            _evaluate_with_session(batch, _remote_session(), orchestrator_job_id="/orchestrator")
    else:
        _evaluate_with_session(batch, _remote_session(), orchestrator_job_id="/orchestrator")

    record = read_record(str(tmp_path / "records" / "run-math500" / "record.json"))
    assert record.status is expected_status
    assert record.metrics["math500"]["accuracy,none"] == 1.0
    assert record.coverage["math500"].n_scored == n_scored


@pytest.mark.parametrize("recovered_before_result", [False, True])
@pytest.mark.parametrize("failure_mode", ["none", "reported", "existing"])
@pytest.mark.parametrize(
    ("entrypoint", "endpoint_route"),
    [("batch", EndpointRoute.DIRECT), ("batch", EndpointRoute.CAPABILITY), ("served", EndpointRoute.DIRECT)],
)
def test_evaluation_resumes_one_archive_and_retries_transport_failures(
    tmp_path, monkeypatch, recovered_before_result, failure_mode, entrypoint, endpoint_route
):
    controller = SimpleNamespace(ready=False, preemptions=0)

    def job_status(_job_id):
        return SimpleNamespace(
            state=JobState.RUNNING,
            task_count=1,
            task_state_counts={TaskState.RUNNING: int(controller.ready)},
            preemption_count=controller.preemptions,
        )

    client = SimpleNamespace(
        job_status=job_status,
        list_endpoint_instances=lambda _name: [object()] if controller.ready else [],
        resolve_endpoint=lambda _name: f"http://10.0.0.{controller.preemptions + 1}:8000",
    )
    monkeypatch.setattr("marin.inference.iris.iris_ctx", lambda: SimpleNamespace(client=client))
    monkeypatch.setattr(
        "marin.evaluation.runner.iris_ctx", lambda: SimpleNamespace(client=client, job_id="/orchestrator")
    )
    monkeypatch.setattr(
        "marin.inference.iris.time", SimpleNamespace(sleep=lambda _seconds: setattr(controller, "ready", True))
    )
    monkeypatch.setattr("marin.inference.iris.requests.get", lambda *_args, **_kwargs: SimpleNamespace(status_code=200))
    session = replace(
        _remote_session(),
        jobs=(SimpleNamespace(job_id="/test/inference", status=lambda: JobStatus.RUNNING),),
    )
    abandoned_paths = []
    partial_result = failure_mode != "none"

    def run_child(_model, _config, output_dir, _env_vars, **_kwargs):
        assert controller.ready
        if entrypoint == "served":
            assert _env_vars["JUDGE_API_KEY"] == "test-judge-key"
        expected_url = (
            f"http://10.0.0.{controller.preemptions + 1}:8000/v1"
            if endpoint_route is EndpointRoute.DIRECT
            else session.model.endpoint.base_url
        )
        assert _model.endpoint.base_url == expected_url
        interrupted = controller.preemptions < 3
        rows = [_lm_eval_generation(0, "accuracy", 1.0, "4")]
        if partial_result and interrupted:
            failed = _lm_eval_generation(1, "accuracy", 0.0, "")
            failed["failure_category"] = "model_transport"
            rows.append(failed)
        elif not interrupted:
            rows.append(_lm_eval_generation(1, "accuracy", 1.0, "5"))
        _write_evalchemy_output(output_dir, "task", {"task": {"accuracy": 0.5 if interrupted else 1.0}}, {"task": rows})
        if interrupted:
            abandoned_paths.append(output_dir)
            controller.preemptions += 1
            controller.ready = recovered_before_result
            if failure_mode != "reported":
                raise InferenceUnavailable("endpoint unavailable")
        return "/eval/child"

    monkeypatch.setattr("marin.evaluation.evalchemy.runner._run_evalchemy_child", run_child)
    monkeypatch.setenv("TEST_EVALUATION_JUDGE_KEY", "test-judge-key")
    judge_config = (
        EvalchemyJudgeConfig(
            base_url="https://judge.example/v1", model="judge", api_key=("env:TEST_EVALUATION_JUDGE_KEY",)
        )
        if entrypoint == "served"
        else None
    )
    executor = EvalchemyExecutor(
        EvalchemyRunConfig(
            name="task",
            tasks=(EvalTaskConfig(name="task", num_fewshot=0, task_alias="task"),),
            judge=judge_config,
        )
    )
    evaluation = _evaluation(tmp_path, "interrupted", executor, endpoint_route)
    evaluation = replace(
        evaluation, identity=replace(evaluation.identity, output_dir=f"file://{tmp_path / 'interrupted'}")
    )
    if failure_mode == "existing":
        failed = _lm_eval_generation(1, "accuracy", 0.0, "")
        failed["failure_category"] = "model_transport"
        _write_evalchemy_output(
            evaluation.identity.output_dir,
            "task",
            {"task": {"accuracy": 0.5}},
            {"task": [_lm_eval_generation(0, "accuracy", 1.0, "4"), failed]},
        )
    batch = replace(
        _hosted_judge_batch(
            tmp_path,
            (
                _evaluation(tmp_path, "before", _successful_evaluation),
                evaluation,
                _evaluation(tmp_path, "after", _successful_evaluation),
            ),
        ),
        judge=None,
        max_concurrent=1,
    )
    catalog_rows = []
    monkeypatch.setattr("marin.evaluation.runner.record_rollout_run", catalog_rows.append)

    if entrypoint == "batch":
        _patch_inference_runtime(monkeypatch, lambda _config: nullcontext(session))
        monkeypatch.setattr(
            "marin.evaluation.runner.iris_ctx",
            lambda: SimpleNamespace(client=client, job_id="/orchestrator"),
        )
        _run_test_batch(batch)
        assert [(row.run_id, row.status) for row in catalog_rows] == [
            ("run-before", "succeeded"),
            ("run-interrupted", "succeeded"),
            ("run-after", "succeeded"),
        ]
        record = read_record(str(tmp_path / "records" / "run-interrupted" / "record.json"))
        assert record.coverage["task"].n_scored == 2
        final_path = record.results_path
    else:
        monkeypatch.setattr("marin.experiment.evaluation.remote_inference", lambda _config: nullcontext(session))
        monkeypatch.setattr("marin.experiment.evaluation.env_vars_from_keys", lambda _keys: {})
        config = EvalchemyEvalConfig(
            model=replace(
                batch.model,
                serve=ServeConfig(auto_overrides=False),
                resource_hint=ResourceHint(gpu={"H100": 1}, memory="16g"),
            ),
            accelerator=batch.accelerator,
            run=executor.config,
            out_path=evaluation.identity.output_dir,
        )
        monkeypatch.setenv("MARIN_PREFIX", f"file://{tmp_path / 'pipeline'}")
        result = resolve(
            ArtifactStep(
                name="evaluation/recovery",
                version="2026.10.15",
                artifact_type=FineStoreEvalchemyResult,
                run=remote(run_served_evalchemy),
                build_config=lambda ctx: replace(config, out_path=ctx.output_path),
            )
        )
        final_path = result.path
        assert result.jobs["eval"] == "/eval/child"
        restored = FineStoreEvalchemyResult.raw_load(result.path)
        assert restored.jobs["eval"] == "/eval/child"
        assert restored.record is not None
        assert restored.record.name == "evaluation/recovery"
        assert restored.record.version == "2026.10.15"
        assert restored.task_metrics()["task"]["accuracy"] == 1.0
    assert len(abandoned_paths) == 3
    assert set(abandoned_paths) == {final_path}
    assert {row["doc_id"] for row in ReadView(final_path).iter_rows(ARCHIVE_SAMPLES_TABLE)} == {"0", "1"}


def test_inference_recovery_stops_when_judge_fails_while_model_is_rebuilding(monkeypatch):
    judge_failed = False

    def job_status(job_id):
        is_judge = str(job_id) == "/test/judge"
        return SimpleNamespace(
            state=JobState.FAILED if is_judge and judge_failed else JobState.RUNNING,
            task_count=1,
            task_state_counts={TaskState.RUNNING: int(is_judge)},
        )

    def advance_recovery(_seconds):
        nonlocal judge_failed
        judge_failed = True

    client = SimpleNamespace(job_status=job_status, list_endpoint_instances=lambda _name: [object()])
    monkeypatch.setattr("marin.inference.iris.iris_ctx", lambda: SimpleNamespace(client=client))
    monkeypatch.setattr("marin.inference.iris.time", SimpleNamespace(sleep=advance_recovery))
    sessions = tuple(
        replace(_remote_session(), jobs=(SimpleNamespace(job_id=f"/test/{name}"),)) for name in ("model", "judge")
    )

    def attempt(_attempt_number):
        pytest.fail("evaluation must not start against an unavailable model")

    with pytest.raises(InferenceBackendFinished):
        run_with_inference_recovery(attempt, sessions)


@pytest.mark.parametrize("max_concurrent", [1, 2])
def test_evaluation_step_persists_run_scoped_speculative_metrics(tmp_path, monkeypatch, max_concurrent):
    def scrape(prompt: int, generated: int, drafts: int, draft_tokens: int, accepted: int):
        return tuple(
            text_string_to_metric_families(
                f"""
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total {prompt}
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total {generated}
# TYPE vllm:spec_decode_num_drafts_total counter
vllm:spec_decode_num_drafts_total {drafts}
# TYPE vllm:spec_decode_num_draft_tokens_total counter
vllm:spec_decode_num_draft_tokens_total {draft_tokens}
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total {accepted}
"""
            )
        )

    scrapes = [scrape(10, 20, 3, 9, 4), scrape(30, 120, 13, 39, 19)]
    monkeypatch.setattr(
        "marin.evaluation.inference_metrics.PrometheusScraper.scrape",
        lambda _self: scrapes.pop(0),
    )
    clock = iter((10.0, 12.0))
    monkeypatch.setattr("marin.evaluation.inference_metrics.time", SimpleNamespace(monotonic=lambda: next(clock)))
    monkeypatch.setattr("marin.evaluation.runner.record_rollout_run", lambda _row: None)
    monkeypatch.setattr(
        "marin.evaluation.runner.iris_ctx",
        lambda: SimpleNamespace(client=SimpleNamespace(resolve_endpoint=lambda _name: "http://10.0.0.1:8000")),
    )
    speculative = SpeculativeServingConfig(
        method=SpeculativeMethod.EAGLE3,
        model=ResolvedModelLocator(uri="s3://models/draft", identity="draft@2026.09.23:abc123"),
        num_speculative_tokens=3,
    )
    batch = EvaluationBatch(
        group_id="group",
        user="tester",
        version="v1",
        description=None,
        records_prefix=str(tmp_path / "records"),
        model=ModelConfig(
            name="model",
            location="s3://models/target",
            identity="target@2026.09.23:def456",
            tokenizer="org/tokenizer",
            tokenizer_revision="tokenizer-revision",
            resource_hint=ResourceHint(gpu={"H100": 1}),
            serve=ServeConfig(speculative=speculative),
        ),
        accelerator=AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=1),
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
        capability_origin="https://iris.example",
        api_model="model",
        evaluations=(_evaluation(tmp_path, "measured", _successful_evaluation),),
        provenance=LaunchProvenance(git_sha="abc", launch_host="host"),
        submission_cluster="marin",
        max_concurrent=max_concurrent,
    )
    session = replace(_remote_session(), metrics_url="https://inference.example/metrics")

    _evaluate_with_session(batch, session, orchestrator_job_id="/orchestrator")

    record = read_record(str(tmp_path / "records" / "run-measured" / "record.json"))
    assert record.model.config is not None
    assert record.model.config.identity == "target@2026.09.23:def456"
    if max_concurrent > 1:
        assert record.inference_metrics is None
        assert len(scrapes) == 2
        return
    assert record.inference_metrics is not None
    assert record.inference_metrics.prompt_tokens == 20
    assert record.inference_metrics.generation_tokens == 100
    assert record.inference_metrics.wall_time_seconds == 2.0
    assert record.inference_metrics.generation_tokens_per_second == 50.0
    assert record.inference_metrics.speculative_decoding is not None
    assert record.inference_metrics.speculative_decoding.model_dump() == {
        "drafts": 10,
        "draft_tokens": 30,
        "accepted_tokens": 15,
        "mean_acceptance_length": 2.5,
        "draft_acceptance_rate": 0.5,
    }
    assert record.model.config.serve.speculative is not None
    assert record.model.config.serve.speculative.model.identity == "draft@2026.09.23:abc123"


def test_speculative_pipeline_launch_selects_gpu(tmp_path, monkeypatch):
    model = ModelConfig(
        name="target",
        location="s3://models/target",
        tokenizer="org/tokenizer",
        resource_hint=ResourceHint(hbm_gb=40),
    )
    speculative = SpeculativeServingConfig(
        method=SpeculativeMethod.EAGLE3,
        model=ResolvedModelLocator(uri="s3://models/draft", identity="draft@2026.09.23:abc123"),
        num_speculative_tokens=3,
    )
    control_config = EvalStepConfig(
        model=model,
        evals="gsm8k-smoke",
        limit=1,
        max_concurrent=1,
        artifact_path=str(tmp_path / "control"),
        accelerator=None,
        submission_cluster="marin",
        federated_cluster=None,
        version="2026.09.23",
    )
    drafted_config = replace(
        control_config,
        model=replace(model, serve=replace(model.serve, speculative=speculative)),
        version="2026.09.23.1",
    )

    submitted_batches: list[EvaluationBatch] = []

    def launch(batch: EvaluationBatch, _client: object):
        submitted_batches.append(batch)
        return SimpleNamespace(
            group_id=batch.group_id,
            records_prefix=batch.records_prefix,
            evaluations=tuple(SimpleNamespace(run_id=evaluation.identity.run_id) for evaluation in batch.evaluations),
            job=SimpleNamespace(wait=lambda *, timeout: None),
        )

    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    monkeypatch.setattr("experiments.evaluation.pipeline.launch_group", launch)
    monkeypatch.setattr(
        "experiments.evaluation.pipeline.iris_ctx",
        lambda: SimpleNamespace(client=object()),
    )
    monkeypatch.setattr(
        "experiments.evaluation.pipeline.read_record",
        lambda path: SimpleNamespace(results_path=f"{path.removesuffix('/record.json')}/results"),
    )

    run_eval_pipeline_step(control_config)
    run_eval_pipeline_step(drafted_config)

    assert submitted_batches[0].accelerator.platform is Platform.TPU
    assert submitted_batches[1].accelerator.platform is Platform.GPU


@pytest.mark.parametrize(
    ("version", "retry_limit", "reuse_completed"),
    [("test", 1, True), ("dev", 1, False), ("test", 2, False)],
)
def test_pipeline_retry_preserves_completed_evaluations(tmp_path, monkeypatch, version, retry_limit, reuse_completed):
    completed: list[str] = []
    attempts: list[str] = []

    def successful(session, output_dir, env_vars, *, backend_state, judge=None):
        completed.append(output_dir)
        return _successful_evaluation(session, output_dir, env_vars, backend_state=backend_state, judge=judge)

    def interrupted(session, output_dir, env_vars, *, backend_state, judge=None):
        attempts.append(output_dir)
        if len(attempts) == 1:
            raise EvaluationError("interrupted evaluation", status=RunStatus.INFRA_FAILED)
        return _successful_evaluation(session, output_dir, env_vars, backend_state=backend_state, judge=judge)

    def launch(batch, _client):
        executions = (successful, interrupted)
        evaluations = tuple(
            replace(evaluation, executor=executor)
            for evaluation, executor in zip(batch.evaluations, executions, strict=True)
        )
        _run_test_batch(replace(batch, evaluations=evaluations))
        return SimpleNamespace(
            group_id=batch.group_id,
            records_prefix=batch.records_prefix,
            evaluations=tuple(SimpleNamespace(run_id=evaluation.identity.run_id) for evaluation in evaluations),
            job=SimpleNamespace(wait=lambda *, timeout: None),
        )

    monkeypatch.setattr("experiments.evaluation.launch.DEFAULT_RECORDS_PREFIX", str(tmp_path / "records"))
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    monkeypatch.setattr("experiments.evaluation.pipeline.iris_ctx", lambda: SimpleNamespace(client=object()))
    monkeypatch.setattr("experiments.evaluation.pipeline.launch_group", launch)
    monkeypatch.setattr("marin.evaluation.runner.record_rollout_run", lambda _record: None)
    _patch_inference_runtime(monkeypatch, lambda _config: nullcontext(_remote_session()))
    config = EvalStepConfig(
        model=models()["qwen3-8b"],
        evals="smoke",
        limit=1,
        max_concurrent=1,
        artifact_path=str(tmp_path / "pipeline"),
        accelerator=None,
        submission_cluster="marin",
        federated_cluster=None,
        version=version,
    )

    with pytest.raises(RuntimeError, match="1 step"):
        run_eval_pipeline_step(config)
    result = run_eval_pipeline_step(replace(config, limit=retry_limit))

    assert len(completed) == (1 if reuse_completed else 2)
    assert (attempts[0] == attempts[1]) == reuse_completed
    assert len(result.run_ids) == 2


def test_evalchemy_executor_classifies_missing_native_archive(tmp_path, monkeypatch):
    output_dir = str(StoragePath("memory://evalchemy-export-failure") / tmp_path.name)
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    session = _remote_session()
    executor = EvalchemyExecutor(
        EvalchemyRunConfig(
            name="gsm8k",
            tasks=(EvalTaskConfig(name="gsm8k", num_fewshot=5),),
        )
    )

    with pytest.raises(EvaluationError) as exc_info:
        executor(session, output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    assert exc_info.value.status is RunStatus.ARTIFACT_FAILED
    assert exc_info.value.jobs == {"eval": "/eval/completed"}


@pytest.mark.parametrize("missing_result", [True, False], ids=["missing", "unscored"])
def test_evalchemy_executor_rejects_incomplete_task_beside_scored_results(tmp_path, monkeypatch, missing_result):
    output_dir = f"file://{tmp_path / 'incomplete'}"
    _write_evalchemy_output(
        output_dir,
        "good_0shot",
        {"good": {"exact_match,none": 1.0}},
        {"good": [_lm_eval_generation(0, "exact_match", 1.0, "4")]},
    )
    if not missing_result:
        with DataStore.open(output_dir, writer_id="unscored-task") as store:
            store.table("evalchemy_results", primary_key=("source_prefix",)).append(
                {"source_prefix": "bad_0shot", "document": json.dumps({"results": {}}), "scored": False}
            )
            store.seal()
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    executor = EvalchemyExecutor(
        EvalchemyRunConfig(name="mixed", tasks=(EvalTaskConfig("good", 0), EvalTaskConfig("bad", 0)))
    )

    with pytest.raises(EvaluationError, match="bad_0shot") as exc_info:
        executor(_remote_session(), output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    assert exc_info.value.status is RunStatus.ARTIFACT_FAILED
    assert exc_info.value.jobs == {"eval": "/eval/completed"}


@pytest.mark.parametrize("failed_sample_has_grade", [True, False])
def test_evalchemy_executor_accepts_scored_partial_results_and_reports_drops(
    tmp_path, monkeypatch, failed_sample_has_grade
):
    output_dir = f"file://{tmp_path / 'partial'}"
    marker = f"[{EVALCHEMY_INFRASTRUCTURE_ERROR}] request timed out"
    samples = [_lm_eval_generation(i, "exact_match", 1.0, "4") for i in range(9)]
    failed_sample = _lm_eval_generation(9, "exact_match", 0.0, marker)
    if not failed_sample_has_grade:
        failed_sample.pop("exact_match")
    samples.append(failed_sample)
    _write_evalchemy_output(
        output_dir,
        "gsm8k_5shot",
        {"gsm8k": {"exact_match,none": 1.0}},
        {"gsm8k": samples},
        scored_counts={"gsm8k": 9},
        failure_category="AgentTimeoutError",
    )
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    executor = EvalchemyExecutor(EvalchemyRunConfig(name="gsm8k", tasks=(EvalTaskConfig("gsm8k", 5),)))

    outcome = executor(_remote_session(), output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    assert outcome.metrics == {"gsm8k_5shot": {"exact_match,none": 1.0}}
    assert outcome.coverage["gsm8k_5shot"] == TaskCoverage(
        n_benchmark=10,
        n_attempted=10,
        n_scored=9,
        n_correct=9,
        errors={EVALCHEMY_INFRASTRUCTURE_ERROR: 1},
    )


def test_aggregate_coverage_keeps_the_full_benchmark_extent():
    coverage = {
        "custom": TaskCoverage(n_benchmark=100, n_attempted=10, n_scored=0, errors={"ungraded": 10}),
    }

    reconciled = _coverage_with_aggregate_counts(coverage, {"custom": {"total_examples": 10.0}})

    assert reconciled["custom"] == TaskCoverage(n_benchmark=100, n_attempted=10, n_scored=10)


def test_run_evalchemy_rejects_partial_aggregate_with_inconsistent_scored_count(tmp_path, monkeypatch):
    output_dir = f"file://{tmp_path / 'partial'}"
    failed = {**_lm_eval_generation(1, "accuracy", 0.0, ""), "failure_category": "model_transport"}
    _write_evalchemy_output(
        output_dir,
        "mrcr",
        {"MRCR": {"mrcr_accuracy": 0.4}},
        {"MRCR": [_lm_eval_generation(0, "accuracy", 0.8, "4"), failed]},
        scored_counts={"MRCR": 2},
    )
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    config = EvalchemyRunConfig(
        name="mrcr",
        tasks=(EvalTaskConfig(name="MRCR", num_fewshot=0, task_alias="mrcr", generation=True),),
    )

    with pytest.raises(EvaluationError) as exc_info:
        run_evalchemy(
            _remote_session().model, config, output_dir, env_vars={}, backend_state=lambda: InferenceBackendState.READY
        )

    assert exc_info.value.status is RunStatus.INFRA_FAILED
    assert exc_info.value.jobs == {"eval": "/eval/completed"}
    assert exc_info.value.coverage["mrcr"] == TaskCoverage(
        n_benchmark=2,
        n_attempted=2,
        n_scored=1,
        errors={EVALCHEMY_INFRASTRUCTURE_ERROR: 1},
    )


def test_evalchemy_executor_rejects_aggregates_that_include_infrastructure_failures(tmp_path, monkeypatch):
    marker = f"[{EVALCHEMY_INFRASTRUCTURE_ERROR}] request failed"
    partial_output_dir = f"file://{tmp_path / 'partial'}"
    _write_evalchemy_output(
        partial_output_dir,
        "mmlu_5shot",
        {
            "mmlu": {"acc,none": 0.0},
            "mmlu_anatomy": {"acc,none": 0.5},
            "mmlu_astronomy": {"acc,none": 1.0},
        },
        {
            "mmlu_anatomy": [
                _lm_eval_generation(0, "acc", 1.0, "4"),
                _lm_eval_generation(1, "acc", 0.0, marker),
            ],
            "mmlu_astronomy": [_lm_eval_generation(0, "acc", 1.0, "4")],
        },
    )
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    executor = EvalchemyExecutor(
        EvalchemyRunConfig(
            name="mmlu",
            tasks=(EvalTaskConfig(name="mmlu", num_fewshot=5),),
        )
    )

    with pytest.raises(EvaluationError) as partial_error:
        executor(_remote_session(), partial_output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    assert partial_error.value.status is RunStatus.INFRA_FAILED
    assert partial_error.value.coverage == {
        "mmlu_5shot/mmlu_anatomy": TaskCoverage(
            n_benchmark=2,
            n_attempted=2,
            n_scored=1,
            n_correct=1,
            errors={EVALCHEMY_INFRASTRUCTURE_ERROR: 1},
        ),
        "mmlu_5shot/mmlu_astronomy": TaskCoverage(n_benchmark=1, n_attempted=1, n_scored=1, n_correct=1),
    }

    failed_output_dir = f"file://{tmp_path / 'failed'}"
    _write_evalchemy_output(
        failed_output_dir,
        "mmlu_5shot",
        {
            "mmlu": {"acc,none": 0.0},
            "mmlu_anatomy": {"acc,none": 0.0},
            "mmlu_astronomy": {"acc,none": 0.0},
        },
        {
            "mmlu_anatomy": [_lm_eval_generation(0, "acc", 0.0, marker)],
            "mmlu_astronomy": [_lm_eval_generation(0, "acc", 0.0, marker)],
        },
    )

    with pytest.raises(EvaluationError) as exc_info:
        executor(_remote_session(), failed_output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    assert exc_info.value.status is RunStatus.INFRA_FAILED
    assert exc_info.value.coverage == {
        "mmlu_5shot/mmlu_anatomy": TaskCoverage(
            n_benchmark=1,
            n_attempted=1,
            n_scored=0,
            errors={EVALCHEMY_INFRASTRUCTURE_ERROR: 1},
        ),
        "mmlu_5shot/mmlu_astronomy": TaskCoverage(
            n_benchmark=1,
            n_attempted=1,
            n_scored=0,
            errors={EVALCHEMY_INFRASTRUCTURE_ERROR: 1},
        ),
    }


def test_evalchemy_executor_rejects_multiple_choice_aggregate_with_inconsistent_scored_count(tmp_path, monkeypatch):
    output_dir = f"file://{tmp_path / 'multiple-choice-transport-failure'}"
    successful = {
        "doc_id": 0,
        "doc": {"question": "Which answer?", "choices": ["A", "B"]},
        "target": 0,
        "arguments": [["Which answer?", "A"], ["Which answer?", "B"]],
        "resps": [[-1.0, True], [-2.0, True]],
        "filtered_resps": [0],
        "filter": "none",
        "metrics": ["acc"],
        "acc": 1.0,
    }
    failed = {**successful, "doc_id": 1, "acc": 0.0, "failure_category": "model_transport"}
    _write_evalchemy_output(
        output_dir,
        "mmlu_0shot",
        {"mmlu": {"acc,none": 0.5}},
        {"mmlu": [successful, failed]},
    )
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    executor = EvalchemyExecutor(EvalchemyRunConfig(name="mmlu", tasks=(EvalTaskConfig(name="mmlu", num_fewshot=0),)))

    with pytest.raises(EvaluationError) as exc_info:
        executor(_remote_session(), output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    assert exc_info.value.status is RunStatus.INFRA_FAILED
    assert exc_info.value.coverage["mmlu_0shot"] == TaskCoverage(
        n_benchmark=2,
        n_attempted=2,
        n_scored=1,
        n_correct=1,
        errors={EVALCHEMY_INFRASTRUCTURE_ERROR: 1},
    )
    samples = [sample_from_archive_row(row) for row in ReadView(output_dir).scan("samples").to_pylist()]
    assert any(
        sample.kind is SampleKind.MULTIPLE_CHOICE and EVALCHEMY_INFRASTRUCTURE_ERROR in (sample.output or "")
        for sample in samples
    )


def test_evalchemy_executor_preserves_evaluator_partial_aggregates(tmp_path, monkeypatch):
    output_dir = f"file://{tmp_path / 'MRCR'}"
    failed = {**_lm_eval_generation(2, "accuracy", 0.0, ""), "failure_category": "model_transport"}
    metrics = {"mrcr_accuracy": 0.6, "bucket_low": 0.4, "bucket_high": 0.8, "num_total": 2.0}
    _write_evalchemy_output(
        output_dir,
        "partial",
        {"MRCR": metrics},
        {
            "MRCR": [
                _lm_eval_generation(0, "accuracy", 0.4, "4"),
                _lm_eval_generation(1, "accuracy", 0.8, "4"),
                failed,
            ]
        },
        scored_counts={"MRCR": 2},
    )
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    executor = EvalchemyExecutor(
        EvalchemyRunConfig(
            name="partial",
            tasks=(EvalTaskConfig(name="MRCR", num_fewshot=0, task_alias="partial", generation=True),),
        )
    )

    outcome = executor(_remote_session(), output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    assert outcome.metrics == {"partial": metrics}
    assert outcome.canonical_metrics == {"partial": {"mrcr_accuracy": 0.6}}
    assert outcome.coverage["partial"] == TaskCoverage(
        n_benchmark=3,
        n_attempted=3,
        n_scored=2,
        errors={EVALCHEMY_INFRASTRUCTURE_ERROR: 1},
    )


def test_evalchemy_executor_uses_aggregate_count_when_custom_task_omits_sample_scores(tmp_path, monkeypatch):
    output_dir = f"file://{tmp_path / 'custom-with-aggregate-grades'}"
    rows = []
    for doc_id in range(3):
        row = _lm_eval_generation(doc_id, "accuracy", float(doc_id > 0), "4")
        row.pop("metrics")
        row.pop("accuracy")
        row.update({"source_id": doc_id, "sample_ordinal": doc_id})
        rows.append(row)
    _write_evalchemy_output(
        output_dir,
        "mmlu-pro",
        {"MMLUPro": {"accuracy_avg": 2 / 3, "total_examples": 3}},
        {"MMLUPro": rows},
    )
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    executor = EvalchemyExecutor(
        EvalchemyRunConfig(
            name="mmlu-pro", tasks=(EvalTaskConfig(name="MMLUPro", num_fewshot=0, task_alias="mmlu-pro"),)
        )
    )

    outcome = executor(_remote_session(), output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    assert outcome.coverage == {
        "mmlu-pro": TaskCoverage(n_benchmark=3, n_attempted=3, n_scored=3, n_correct=None, n_unanswered=0)
    }
    [archived] = ReadView(output_dir).scan("samples").to_pylist(maps_as_pydicts="strict")[:1]
    assert sample_from_archive_row(archived).metrics == {}


@pytest.mark.parametrize(
    ("counts", "n_scored"),
    [
        ({"scored_count": 3}, 3),
        ({"scored_count": 2}, 0),
        ({"total_examples": 3, "scored_count": 2}, 0),
        ({"sample_len": 3, "scored_count": 2}, 0),
    ],
)
def test_evalchemy_executor_uses_code_task_scored_count_when_samples_omit_scores(
    tmp_path, monkeypatch, counts, n_scored
):
    output_dir = f"file://{tmp_path / 'code-with-aggregate-grades'}"
    rows = []
    for doc_id in range(3):
        row = _lm_eval_generation(doc_id, "python_pass@1", float(doc_id > 0), "def answer(): pass")
        row.pop("metrics")
        row.pop("python_pass@1")
        rows.append(row)
    _write_evalchemy_output(
        output_dir,
        "humanevalplus",
        {"HumanEvalPlus": {"python_pass@1": 2 / 3, **counts}},
        {"HumanEvalPlus": rows},
    )
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    executor = EvalchemyExecutor(
        EvalchemyRunConfig(
            name="humanevalplus",
            tasks=(EvalTaskConfig(name="HumanEvalPlus", num_fewshot=0, task_alias="humanevalplus"),),
        )
    )

    outcome = executor(_remote_session(), output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    assert outcome.coverage == {
        "humanevalplus": TaskCoverage(
            n_benchmark=3,
            n_attempted=3,
            n_scored=n_scored,
            n_correct=None,
            n_unanswered=0,
            errors={} if n_scored == 3 else {"ungraded": 3},
        )
    }


def test_evalchemy_executor_normalizes_native_prompts_into_rollouts(tmp_path, monkeypatch):
    output_dir = f"file://{tmp_path / 'native-prompt'}"
    prompt = json.dumps([{"role": "user", "content": "Question: 2+2?"}])
    raw = _lm_eval_generation(0, "exact_match", 1.0, "4")
    raw["arguments"] = [[[prompt], {"temperature": 1.0}]]
    _write_evalchemy_output(
        output_dir,
        "gsm8k_5shot",
        {"gsm8k": {"exact_match,none": 1.0}},
        {"gsm8k": [raw]},
    )
    monkeypatch.setattr(
        "marin.evaluation.evalchemy.runner._run_evalchemy_child",
        lambda _model, _config, _output_dir, _env_vars, **_kwargs: "/eval/completed",
    )
    executor = EvalchemyExecutor(
        EvalchemyRunConfig(
            name="gsm8k",
            tasks=(EvalTaskConfig(name="gsm8k", num_fewshot=5, generation=True),),
            apply_chat_template=True,
        )
    )

    executor(_remote_session(), output_dir, {}, backend_state=lambda: InferenceBackendState.READY)

    rows = list(ReadView(output_dir).iter_rows(ARCHIVE_ROLLOUTS_TABLE))
    assert [(row["participant_type"], row["content"]) for row in rows] == [
        ("user", "Question: 2+2?"),
        ("assistant", "4"),
    ]


def test_launch_group_resolves_declared_secrets_outside_the_pickled_batch(tmp_path, monkeypatch):
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
        endpoint_route=EndpointRoute.CAPABILITY,
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
        max_concurrent=8,
        secret_env={"DAYTONA_API_KEY": ("env:MARIN_TEST_EVAL_SECRET",)},
    )

    launch_group(batch, Client())

    assert captured["environment"].env_vars["DAYTONA_API_KEY"] == resolved_value
    assert resolved_value.encode() not in captured["entrypoint"].workdir_files["_callable.pkl"]


@pytest.mark.parametrize(
    ("submission_cluster", "expects_federation"),
    (("cw-us-east-08a", False), ("marin", True)),
)
def test_launch_group_only_federates_to_a_different_cluster(tmp_path, submission_cluster, expects_federation):
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
        max_concurrent=8,
    )

    launch_group(batch, Client())

    constraints = captured["constraints"]
    if expects_federation:
        assert constraints[0].key == "cluster"
    else:
        assert constraints is None


def test_launch_group_uses_resolved_federated_cluster_and_priority(monkeypatch):
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
    launch_group(batch, Client())

    assert captured["constraints"] == [
        Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value="cw-rno2a")
    ]
    assert captured["priority_band"] == job_pb2.PRIORITY_BAND_INTERACTIVE


def test_build_evaluation_batch_places_hosted_judge_with_candidate(monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=("aime-harbor",),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=1,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        judge_model=models()["qwen3.5-122b-a10b-fp8"],
        judge_accelerator="H100x8",
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    assert batch.accelerator.target_cluster == "cw-rno2a"
    assert batch.judge is not None
    assert batch.judge.accelerator.label == "H100x8"
    assert batch.judge.accelerator.target_cluster == "cw-rno2a"
    assert batch.judge.api_model == "qwen3.5-122b-a10b-fp8"


def test_build_evaluation_batch_rejects_hosted_judge_for_evalchemy_evaluations(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=("mmlu-smoke",),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.GPU,
        accelerator="H100x1",
        limit=1,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster="cw-rno2a",
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        judge_model=models()["qwen3.5-122b-a10b-fp8"],
        judge_accelerator="H100x8",
    )

    with pytest.raises(ValueError, match="serves Harbor verifiers only"):
        build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")


def test_build_evaluation_batch_uses_submission_cluster_for_capability_origin(monkeypatch):
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
    assert {evaluation.identity.eval_runtime for evaluation in batch.evaluations} == {harbor_runtime_descriptor()}
    assert all(evaluation.endpoint_route is EndpointRoute.CAPABILITY for evaluation in batch.evaluations)
    assert all(evaluation.identity.eval_ref.harbor.config_digest for evaluation in batch.evaluations)
    assert all(evaluation.identity.eval_ref.harbor.task_limit == 1 for evaluation in batch.evaluations)


def test_build_evaluation_batch_routes_declared_verifier_host_secrets(monkeypatch, tmp_path):
    _install_fake_harbor_preflight(monkeypatch, verifier_env_keys=("TOGETHER_API_KEY",))
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    config_path = _write_harbor_config(tmp_path / "simpleqa.yaml")
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=(),
        evalchemy_definitions=(),
        harbor_definitions=(HarborDefinition("simpleqa", config_path),),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    assert batch.secret_env == {
        "DAYTONA_API_KEY": (
            "env:DAYTONA_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/DAYTONA_EVAL_API_KEY/versions/latest",
        ),
        "TOGETHER_API_KEY": (
            "env:TOGETHER_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/TOGETHER_API_KEY/versions/latest",
        ),
    }
    assert batch.evaluations[0].secret_env_keys == ("DAYTONA_API_KEY", "TOGETHER_API_KEY")


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

    assert batch.evaluations[0].identity.eval_runtime == evalchemy_runtime_descriptor(
        ("serve-eval", "math500"), evalchemy_runtime_source_hash()
    )


def test_build_evaluation_batch_routes_and_records_financebench_judge(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    definition = EvalchemyDefinition(
        name="financebench",
        config_path=Path("experiments/evaluation/configs/evalchemy/financebench.yaml"),
    )
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
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

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    evaluation = batch.evaluations[0]
    assert batch.secret_env == {
        "JUDGE_API_KEY": (
            "env:TOGETHER_API_KEY",
            "gcp-secret://projects/hai-gcp-models/secrets/TOGETHER_API_KEY/versions/latest",
        )
    }
    assert evaluation.secret_env_keys == ("JUDGE_API_KEY",)
    assert evaluation.identity.eval_ref.evalchemy.judge.model_dump() == {
        "base_url": "https://api.together.xyz/v1",
        "model": "openai/gpt-oss-120b",
    }
    assert "TOGETHER_API_KEY" not in evaluation.identity.eval_ref.model_dump_json()


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
    ("generation", "expected"),
    (
        (
            GenerationConfig(chat_template_kwargs={"enable_thinking": True, "strict_format": False}),
            {"enable_thinking": False, "strict_format": False},
        ),
        (
            GenerationConfig(
                chat_template_kwargs={"reasoning_effort": "high"},
                thinking_off_template_kwargs={"reasoning_effort": "low"},
            ),
            {"reasoning_effort": "low"},
        ),
    ),
)
def test_file_evalchemy_chat_template_kwargs_override_model_per_key(tmp_path, monkeypatch, generation, expected):
    config_path = tmp_path / "thinking.yaml"
    config_path.write_text("tasks: [triviaqa]\nchat_template_kwargs:\n  enable_thinking: false\n")
    model = replace(
        models()["qwen3-8b"],
        generation=generation,
    )
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=model,
        evals=(),
        evalchemy_definitions=(EvalchemyDefinition(name="thinking", config_path=config_path),),
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
    assert evalchemy.chat_template_kwargs == expected


def test_model_evalchemy_concurrency_limits_the_effective_launch(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    model = replace(models()["qwen3-8b"], evalchemy_num_concurrent=4)
    definition = EvalchemyDefinition(
        name="gpqa-diamond",
        config_path=Path("experiments/evaluation/configs/evalchemy/gpqa-diamond.yaml"),
    )
    spec = LaunchSpec(
        model=model,
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

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    evalchemy = batch.evaluations[0].identity.eval_ref.evalchemy
    assert evalchemy is not None
    assert evalchemy.num_concurrent == 4


@pytest.mark.parametrize(
    ("config_name", "enable_thinking"),
    (
        ("aime24", True),
        ("math500", True),
        ("olympiadbench", True),
        ("mmlu-pro", True),
        ("gpqa-diamond", True),
        ("humanevalplus", False),
        ("mbppplus", False),
        ("gsm8k-0shot", False),
        ("triviaqa", False),
        ("cruxeval", False),
        ("financebench", False),
        ("ifbench", False),
        ("mrcr", False),
    ),
)
def test_policy_evalchemy_configs_override_model_thinking_mode(config_name, enable_thinking, monkeypatch):
    config_path = Path("experiments/evaluation/configs/evalchemy") / f"{config_name}.yaml"
    model = replace(
        models()["qwen3-8b"],
        generation=GenerationConfig(chat_template_kwargs={"enable_thinking": not enable_thinking}),
    )
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=model,
        evals=(),
        evalchemy_definitions=(EvalchemyDefinition(name=config_name, config_path=config_path),),
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
    assert evalchemy.chat_template_kwargs["enable_thinking"] is enable_thinking


def test_seed_override_replaces_the_evalchemy_config_seed_in_records(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    definition = EvalchemyDefinition(
        name="ifeval",
        config_path=Path("experiments/evaluation/configs/evalchemy/ifeval.yaml"),
    )
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=(),
        evalchemy_definitions=(definition,),
        harbor_definitions=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        seed=51,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    evalchemy = batch.evaluations[0].identity.eval_ref.evalchemy
    assert evalchemy is not None
    assert evalchemy.seed == 51


def test_registry_family_travels_into_the_record_the_launcher_writes(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=(),
        evalchemy_definitions=(
            EvalchemyDefinition(
                name="gsm8k-0shot",
                config_path=Path("experiments/evaluation/configs/evalchemy/gsm8k-0shot.yaml"),
                family="gsm8k",
            ),
            EvalchemyDefinition(
                name="ifeval",
                config_path=Path("experiments/evaluation/configs/evalchemy/ifeval.yaml"),
            ),
        ),
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

    families = {
        evaluation.identity.eval_ref.name: evaluation.identity.eval_ref.family for evaluation in batch.evaluations
    }
    assert families == {"gsm8k-0shot": "gsm8k", "ifeval": None}


@pytest.mark.parametrize(
    ("benchmark_limit", "model_limit", "expected_limit", "expected_warnings"),
    [
        (128, 8192, 128, 1),
        (8192, 2048, 2048, 0),
        (None, 8192, 8192, 0),
        (None, None, None, 0),
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
    assert batch.evaluations[1].identity.eval_runtime == evalchemy_runtime_descriptor(
        ("serve-eval", "ifeval"), evalchemy_runtime_source_hash()
    )
    assert ifeval.model_dump(mode="json", exclude_none=True) == {
        "name": "ifeval",
        "mechanism": "evalchemy",
        "source_digest": source_config_digest(evalchemy_config_path),
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
            "chat_tokenizer_backend": "none",
            "debug": False,
            "max_eval_instances": 2,
            "num_concurrent": 64,
            "batch_size": "1",
            "seed": 1234,
            "extra_gen_kwargs": {},
            "extra_model_args": {},
            "chat_template_kwargs": {},
        },
    }

    evaluation = batch.evaluations[2]
    assert evaluation.identity.eval_ref.model_dump(mode="json", exclude_none=True) == {
        "name": "aime-policy",
        "mechanism": "harbor",
        "source_digest": source_config_digest(config_path),
        "tasks": [
            {
                "name": "aime",
                "generation": False,
                "unsafe_code": False,
                "completion_only": False,
                "benchmark": {
                    "schema_version": 1,
                    "task": "aime",
                    "primary_metric": "reward",
                    "metric_kind": "continuous",
                    "metrics": [
                        {
                            "name": "reward",
                            "source_name": "reward",
                            "kind": "continuous",
                            "higher_is_better": True,
                        }
                    ],
                    "n_benchmark": 1,
                    "n_attempted": 1,
                    "reducers": [],
                    "requires_judge": False,
                },
            }
        ],
        "harbor": {
            "dataset": "aime",
            "version": "1.0",
            "agent": "opencode",
            "env": "daytona",
            "task_limit": 2,
            "config_digest": evaluation.identity.eval_ref.harbor.config_digest,
            "harbor_config_commit": HARBOR_SOURCE.commit,
            "max_input_tokens": _PREFLIGHT_MAX_INPUT_TOKENS,
            "max_output_tokens": _PREFLIGHT_MAX_OUTPUT_TOKENS,
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
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "result.json").write_text(
            json.dumps(
                {
                    "n_total_trials": 1,
                    "benchmark_metadata": [config.benchmark.model_dump(mode="json")],
                }
            )
        )
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
        backend_state=lambda: InferenceBackendState.READY,
    )

    assert outcome.canonical_metrics["aime"]["reward"] == 1.0
    assert captured["overlay"].task_limit == 2
    assert captured["overlay"].served_model == "served-qwen3-8b"
    assert captured["overlay"].endpoint_url == "https://iris.example/capability/v1"
    assert captured["overlay"].model_agent_kwargs["extra_body"] == ('{"chat_template_kwargs":{"enable_thinking":true}}')


def test_build_evaluation_batch_gives_harbor_the_served_context_limits(tmp_path, monkeypatch):
    preflight_requests = _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    config_path = _write_harbor_config(tmp_path / "aime-policy.yaml")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_config = checkpoint / "config.json"
    checkpoint_config.write_text(json.dumps({"max_position_embeddings": 131072}))
    spec = LaunchSpec(
        model=replace(
            models()["qwen3-8b"],
            location=str(checkpoint),
            resource_hint=ResourceHint(hbm_gb=21, memory="32g"),
            serve=ServeConfig(max_model_len=1048576),
            generation=GenerationConfig(max_gen_toks=8192),
        ),
        evals=(),
        evalchemy_definitions=(),
        harbor_definitions=(HarborDefinition(name="aime-policy", config_path=config_path),),
        platform=Platform.TPU,
        accelerator=None,
        limit=None,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")

    served_limits = {"max_input_tokens": 131072, "max_output_tokens": 8192}
    (evaluation,) = batch.evaluations
    assert [request["model_info"] for request in preflight_requests] == [served_limits]
    assert evaluation.executor.model_agent_kwargs["model_info"] == served_limits
    harbor = evaluation.identity.eval_ref.harbor
    assert (harbor.max_input_tokens, harbor.max_output_tokens) == (
        _PREFLIGHT_MAX_INPUT_TOKENS,
        _PREFLIGHT_MAX_OUTPUT_TOKENS,
    )
    checkpoint_config.unlink()
    inference = inference_config_for_model(batch.model, batch.accelerator, env_vars={}, priority=batch.priority_band)
    assert inference.model.max_model_len == served_limits["max_input_tokens"]


def test_launch_dry_run_prints_the_resolved_harbor_agent_context(tmp_path, monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    config_path = _write_harbor_config(tmp_path / "aime-policy.yaml")

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model",
            "qwen3-8b",
            "--harbor-config",
            str(config_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"max_input_tokens={_PREFLIGHT_MAX_INPUT_TOKENS}" in result.output
    assert f"max_output_tokens={_PREFLIGHT_MAX_OUTPUT_TOKENS}" in result.output


@pytest.mark.parametrize(
    ("overrides", "target_cluster", "priority"),
    [
        ((), "cw-us-east-02a", "inherit"),
        (
            ("--federated_cluster", "cw-rno2a", "--priority", "interactive"),
            "cw-rno2a",
            "interactive",
        ),
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


def test_launch_dry_run_accepts_hosted_judge(monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model",
            "qwen3-0.6b",
            "--judge-model",
            "qwen3.5-122b-a10b-fp8",
            "--judge-accelerator",
            "H100x8",
            "--harbor-config",
            "experiments/evaluation/configs/harbor/simpleqa-hosted-judge.yaml",
            "--platform",
            "gpu",
            "--accelerator",
            "H100x1",
            "--federated_cluster",
            "cw-rno2a",
            "--limit",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "judge: qwen3.5-122b-a10b-fp8" in result.output
    assert "accel=H100x8" in result.output
    assert "region_or_cluster=cw-rno2a" in result.output


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


def test_launch_reuses_compatible_harbor_results_path(tmp_path, monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    policy = _write_harbor_config(tmp_path / "aime-policy.yaml")
    output_dir = f"memory://{tmp_path.name}/existing-harbor-results"
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=(),
        evalchemy_definitions=(),
        harbor_definitions=(HarborDefinition(name="aime-policy", config_path=policy),),
        platform=Platform.TPU,
        accelerator=None,
        limit=None,
        records_prefix="memory://records",
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")

    batch = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="test"), "tester")
    (StoragePath(output_dir) / "harbor_resume_identity.json").write_text(
        batch.evaluations[0].executor.resume_identity.model_dump_json()
    )
    existing_trial = StoragePath(output_dir) / "scored-trial.json"
    existing_trial.write_text('{"reward": 0.0}')
    submitted = []

    def launch(batch, _client):
        evaluation = replace(batch.evaluations[0], executor=_successful_evaluation)
        execution = replace(batch, records_prefix=str(tmp_path / "records"), evaluations=(evaluation,))
        _evaluate_with_session(execution, _remote_session(), orchestrator_job_id="/test")
        submitted.append(evaluation.identity)
        return SimpleNamespace(group_id=batch.group_id, model_name=batch.model.name, evaluations=())

    monkeypatch.setattr("experiments.evaluation.cli.open_iris_client", lambda **_kwargs: nullcontext(object()))
    monkeypatch.setattr("experiments.evaluation.cli.launch_group", launch)
    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model",
            "qwen3-8b",
            "--harbor-config",
            str(policy),
            "--resume-results-path",
            output_dir,
            "--no-wait",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (StoragePath(output_dir) / "endpoint.txt").read_text() == _remote_session().model.endpoint.base_url
    assert json.loads(existing_trial.read_text()) == {"reward": 0.0}
    record = read_record(str(tmp_path / "records" / submitted[0].run_id / "record.json"))
    assert record.results_path == output_dir


def test_launch_rejects_resume_for_multiple_evaluations(monkeypatch):
    _install_fake_harbor_preflight(monkeypatch)
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")

    result = CliRunner().invoke(
        cli,
        [
            "launch",
            "--model",
            "qwen3-8b",
            "--evals",
            "aime-harbor,tb2",
            "--resume-results-path",
            "memory://existing-harbor-results",
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "requires exactly one Harbor evaluation" in result.output


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


def test_final_batch_configuration_binds_step_execution(tmp_path, monkeypatch):
    executed: list[str] = []
    priorities: list[int] = []

    def executor(_session, output_dir, _env_vars, *, backend_state, judge=None):
        executed.append(output_dir)
        return EvaluationOutcome(metrics={"task": {"accuracy": 0.5}})

    def inference_config(model, *_args, priority, **_kwargs):
        priorities.append(priority)
        return SimpleNamespace(model=SimpleNamespace(model_id=model.name))

    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda _cluster: "https://iris.example")
    monkeypatch.setattr("marin.evaluation.runner.record_rollout_run", lambda _record: None)
    _patch_inference_runtime(monkeypatch, lambda _config: nullcontext(_remote_session()))
    monkeypatch.setattr("marin.evaluation.runner.inference_config_for_model", inference_config)
    spec = LaunchSpec(
        model=models()["qwen3-8b"],
        evals=("mmlu-smoke",),
        evalchemy_definitions=(),
        harbor_definitions=(),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix=str(tmp_path / "records"),
        submission_cluster="marin",
        federated_cluster=None,
        priority_band=job_pb2.PRIORITY_BAND_INHERIT,
    )
    prepared = build_evaluation_batch(spec, LaunchProvenance(git_sha="abc", launch_host="host"), "tester")
    original = prepared.evaluations[0]
    final_evaluation = Evaluation(
        identity=original.identity,
        executor=executor,
        endpoint_route=original.endpoint_route,
        secret_env_keys=original.secret_env_keys,
    )
    batch = replace(
        prepared,
        evaluations=(final_evaluation,),
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        max_concurrent=1,
    )

    paths = _run_test_batch(batch)

    assert len(paths) == 1
    assert read_record(paths[0]).status is RunStatus.SUCCEEDED
    assert _step_status(batch, 0) == STATUS_SUCCESS
    assert executed == [final_evaluation.identity.output_dir]
    assert priorities == [job_pb2.PRIORITY_BAND_INTERACTIVE]
