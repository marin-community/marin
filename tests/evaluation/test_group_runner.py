# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable behavior of the serve-once evaluation executor."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from marin.evaluation.evalchemy import EvalchemyRunConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.model_config import GenerationConfig, ModelConfig
from marin.evaluation.records import EvalRef, HardwareRef, ModelRef, Provenance, RunStatus
from marin.evaluation.runner import (
    EvalchemyRunner,
    EvalRunner,
    Evaluation,
    EvaluationBatch,
    EvaluationError,
    EvaluationIdentity,
    EvaluationOutcome,
    run_evaluation_batch,
)
from marin.evaluation.serving_config import EvaluationServingConfig, ServeSpec
from marin.inference.types import OpenAIEndpoint, RunningModel


@dataclass(frozen=True)
class _Runner:
    name: str
    error: EvaluationError | None = None
    mechanism: str = "test"
    max_eval_instances: int | None = None
    task_names: tuple[str, ...] = ()

    def bind(self, model, *, limit, region):
        return self

    def reference(self) -> EvalRef:
        return EvalRef(name=self.name, mechanism=self.mechanism)

    def run(self, model, output_dir, env_vars) -> EvaluationOutcome:
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

    def endpoint_generation(self) -> frozenset[str]:
        return frozenset({"endpoint"})

    def wait_for_restart(self, previous: frozenset[str], timeout: float) -> bool:
        return False


def _evaluation(runner: EvalRunner) -> Evaluation:
    return Evaluation(
        identity=EvaluationIdentity(
            run_id=f"run-{runner.name}",
            created_at="2026-07-24T00:00:00+00:00",
            output_dir=f"gs://bucket/{runner.name}/results",
            eval_ref=runner.reference(),
        ),
        runner=runner,
    )


def _batch(*runners: EvalRunner) -> EvaluationBatch:
    return EvaluationBatch(
        group_id="group",
        user="tester",
        version=None,
        description=None,
        records_prefix="gs://bucket/runs",
        serving=EvaluationServingConfig(
            weights="model",
            tokenizer="tokenizer",
            spec=ServeSpec(tpu_type="v6e-4"),
            endpoint_origin="https://iris.example",
        ),
        evaluations=tuple(_evaluation(runner) for runner in runners),
        model_ref=ModelRef(name="model", location="model", backend="vllm"),
        hardware_ref=HardwareRef(
            platform="tpu",
            accelerator="v6e-4",
            region_or_cluster="us-central1",
        ),
        provenance=Provenance(git_sha="abc", eval_image="image", launch_host="host"),
    )


def test_evalchemy_model_generation_overrides_merge_with_suite_settings():
    runner = EvalchemyRunner(
        EvalchemyRunConfig(
            name="eval",
            tasks=(EvalTaskConfig("task", 0),),
            max_gen_toks=512,
            extra_gen_kwargs={"temperature": "0", "repetition_penalty": "1.0"},
        )
    )
    model = ModelConfig(
        name="model",
        location="org/model",
        generation=GenerationConfig(
            max_gen_toks=1024,
            extra_gen_kwargs={"repetition_penalty": "1.1", "skip_special_tokens": "false"},
        ),
    )

    bound = runner.bind(model, limit=3, region="us-central1")

    assert bound.config.max_gen_toks == 1024
    assert bound.config.max_eval_instances == 3
    assert bound.config.runtime.region == "us-central1"
    assert bound.config.extra_gen_kwargs == {
        "temperature": "0",
        "repetition_penalty": "1.1",
        "skip_special_tokens": "false",
    }


def _patch_runtime(monkeypatch, records: list, session: _Session | None = None):
    session = session or _Session()

    @contextmanager
    def inference(*args, **kwargs):
        assert kwargs["endpoint_origin"] == "https://iris.example"
        yield session

    monkeypatch.setattr(
        EvaluationServingConfig,
        "resolve",
        lambda config, env_vars: SimpleNamespace(start=lambda: inference(endpoint_origin=config.endpoint_origin)),
    )
    monkeypatch.setattr("marin.evaluation.runner.configure_coreweave_s3", lambda: None)
    monkeypatch.setattr("marin.evaluation.runner.iris_ctx", lambda: SimpleNamespace(job_id="/group"))

    def write(record, prefix):
        assert prefix == "gs://bucket/runs"
        records.append(record)
        return f"{prefix}/{record.run_id}/record.json"

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
        run_evaluation_batch(_batch(_Runner("one", error=failure), _Runner("two")))

    assert [record.status for record in records] == [RunStatus.FAILED, RunStatus.SUCCEEDED]
    assert records[0].jobs["eval"] == "/failed"
    assert records[0].log_tails == {"eval": ("tail",)}
    assert records[1].jobs["eval"] == "/two"


def test_evaluation_retries_after_the_inference_endpoint_is_replaced(monkeypatch):
    records = []

    class _RestartedSession(_Session):
        def wait_for_restart(self, previous: frozenset[str], timeout: float) -> bool:
            assert previous == frozenset({"endpoint"})
            assert timeout > 0
            return True

    class _RetryRunner:
        name = "one"
        mechanism = "test"
        max_eval_instances = None
        task_names = ()

        def __init__(self) -> None:
            self.calls = 0

        def bind(self, model, *, limit, region):
            return self

        def reference(self) -> EvalRef:
            return EvalRef(name=self.name, mechanism=self.mechanism)

        def run(self, model, output_dir, env_vars) -> EvaluationOutcome:
            self.calls += 1
            if self.calls == 1:
                raise EvaluationError("endpoint unavailable", status=RunStatus.FAILED)
            return EvaluationOutcome(metrics={"task": {"acc": 0.5}})

    session = _RestartedSession()
    _patch_runtime(monkeypatch, records, session)
    runner = _RetryRunner()

    run_evaluation_batch(_batch(runner))

    assert runner.calls == 2
    assert records[0].status is RunStatus.SUCCEEDED
