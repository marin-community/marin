# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable behavior of the serve-once evaluation executor."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fray.types import JobStatus
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint
from marin.evaluation.records import EvalRef, Provenance, RunStatus
from marin.evaluation.runner import (
    Evaluation,
    EvaluationBatch,
    EvaluationError,
    EvaluationIdentity,
    EvaluationOutcome,
    run_evaluation_batch,
)
from marin.inference.iris import RemoteInferenceStartupError
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem import prefix_join


@dataclass(frozen=True)
class _Executor:
    name: str
    error: EvaluationError | None = None
    mechanism: str = "test"

    def __call__(self, model, output_dir, env_vars) -> EvaluationOutcome:
        if self.error is not None:
            raise self.error
        return EvaluationOutcome(metrics={"task": {"acc": 0.5}}, jobs={"eval": f"/{self.name}"})


class _Session:
    jobs = ()
    model = RunningModel(
        endpoint=OpenAIEndpoint(base_url="https://iris.example/proxy/t/token/endpoint/v1", model="model"),
        tokenizer="tokenizer",
    )

    def check_alive(self) -> None:
        return None


def _evaluation(executor: _Executor) -> Evaluation:
    return Evaluation(
        identity=EvaluationIdentity(
            run_id=f"run-{executor.name}",
            created_at="2026-07-24T00:00:00+00:00",
            output_dir=f"gs://bucket/{executor.name}/results",
            eval_ref=EvalRef(name=executor.name, mechanism=executor.mechanism),
        ),
        executor=executor,
    )


def _batch(*executors: _Executor) -> EvaluationBatch:
    return EvaluationBatch(
        group_id="group",
        user="tester",
        version=None,
        description=None,
        records_prefix="gs://bucket/runs",
        model=ModelConfig(
            name="model",
            location="model",
            tokenizer="tokenizer",
            resource_hint=ResourceHint(hbm_gb=3),
        ),
        accelerator=AcceleratorChoice(platform=Platform.TPU, tpu_type="v6e-4", region="us-central1"),
        capability_origin="https://iris.example",
        api_model="model",
        evaluations=tuple(_evaluation(executor) for executor in executors),
        provenance=Provenance(git_sha="abc", eval_image="image", launch_host="host"),
    )


def _patch_runtime(monkeypatch, records: list, session: _Session | None = None):
    session = session or _Session()

    @contextmanager
    def inference(config):
        assert config is inference_config
        yield session

    inference_config = object()
    monkeypatch.setattr("marin.evaluation.runner.inference_config_for_model", lambda *args, **kwargs: inference_config)
    monkeypatch.setattr("marin.evaluation.runner.remote_inference", inference)
    monkeypatch.setattr("marin.evaluation.runner.configure_coreweave_s3", lambda: None)
    monkeypatch.setattr("marin.evaluation.runner.iris_ctx", lambda: SimpleNamespace(job_id="/group"))

    def write(record, prefix):
        assert prefix == "gs://bucket/runs"
        records.append(record)
        return prefix_join(prefix, f"{record.run_id}/record.json")

    monkeypatch.setattr("marin.evaluation.runner.write_record", write)


def test_evaluation_failure_is_recorded_and_later_evaluations_continue(monkeypatch):
    records = []
    _patch_runtime(monkeypatch, records)
    failure = EvaluationError(
        "bad answer",
        status=RunStatus.FAILED,
        jobs={"eval": "/failed"},
        log_tails={"eval": ("tail",)},
    )

    with pytest.raises(RuntimeError, match="1 of 2 evals failed"):
        run_evaluation_batch(_batch(_Executor("one", error=failure), _Executor("two")))

    assert [record.status for record in records] == [RunStatus.FAILED, RunStatus.SUCCEEDED]
    assert records[0].jobs["eval"] == "/failed"
    assert records[0].log_tails == {"eval": ("tail",)}
    assert records[1].jobs["eval"] == "/two"


def test_inference_startup_failure_records_job_even_when_logs_are_unavailable(monkeypatch):
    class _FailedJob:
        job_id = "/serve-failed"

        def wait(self, timeout=None, *, raise_on_failure=True):
            return JobStatus.FAILED

        def status(self):
            return JobStatus.FAILED

        def logs(self, max_lines: int = 0) -> tuple[str, ...]:
            raise RuntimeError("log service unavailable")

        def terminate(self) -> None:
            return None

    records = []
    _patch_runtime(monkeypatch, records)

    @contextmanager
    def failed_inference(config):
        raise RemoteInferenceStartupError("server failed", jobs=(_FailedJob(),))
        yield

    monkeypatch.setattr("marin.evaluation.runner.remote_inference", failed_inference)

    with pytest.raises(RuntimeError, match="evaluation batch inference failed"):
        run_evaluation_batch(_batch(_Executor("one")))

    assert records[0].status is RunStatus.INFRA_FAILED
    assert records[0].jobs["serve"] == "/serve-failed"
    assert records[0].log_tails["serve"] == ()
