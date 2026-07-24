# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the generic serve-once evaluation group."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from marin.evaluation.definitions import (
    EvalRunIdentity,
    ResolvedEvalchemyRun,
)
from marin.evaluation.evalchemy import (
    EndpointMovedError,
    EvalchemyOutcome,
    EvalchemyRunConfig,
    EvalPipelineError,
    PipelineStage,
)
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.group_runner import EvalGroupParams, run_eval_group
from marin.evaluation.records import (
    EvalRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
)
from marin.evaluation.serving_config import EvaluationServingConfig, ServeSpec
from marin.inference.types import OpenAIEndpoint, RunningModel


class _Result:
    def task_metrics(self) -> dict[str, dict[str, float]]:
        return {"task": {"acc": 0.5}}


class _Session:
    jobs = ()
    wait_count = 0

    def resolve_model(self, *args, **kwargs) -> RunningModel:
        return RunningModel(
            endpoint=OpenAIEndpoint(base_url="http://inference/v1", model="model"),
            tokenizer="tokenizer",
        )

    def wait_model(self, timeout: float) -> RunningModel:
        assert timeout > 0
        self.wait_count += 1
        return self.resolve_model()

    def endpoint_departed(self, model: RunningModel) -> bool:
        return False

    def check_alive(self) -> None:
        return None


def _run(name: str) -> ResolvedEvalchemyRun:
    return ResolvedEvalchemyRun(
        identity=EvalRunIdentity(
            run_id=f"run-{name}",
            created_at="2026-07-24T00:00:00+00:00",
            output_dir=f"gs://bucket/{name}/results",
            eval_ref=EvalRef(name=name, mechanism="evalchemy"),
        ),
        config=EvalchemyRunConfig(
            name=name,
            tasks=(EvalTaskConfig("task", 0),),
        ),
    )


def _group(*runs: ResolvedEvalchemyRun) -> EvalGroupParams:
    return EvalGroupParams(
        group_id="group",
        user="tester",
        version=None,
        description=None,
        records_prefix="gs://bucket/runs",
        serving=EvaluationServingConfig(
            model="model",
            tokenizer="tokenizer",
            spec=ServeSpec(tpu_type="v6e-4"),
        ),
        runs=runs,
        model_ref=ModelRef(name="model", location="model", backend="vllm"),
        hardware_ref=HardwareRef(
            platform="tpu",
            accelerator="v6e-4",
            region_or_cluster="us-central1",
        ),
        provenance=Provenance(git_sha="abc", eval_image="image", launch_host="host"),
    )


def _patch_runtime(monkeypatch, session: _Session, records: list):
    @contextmanager
    def inference(*args, **kwargs):
        yield session

    monkeypatch.setattr("marin.evaluation.group_runner.remote_inference", inference)
    monkeypatch.setattr(
        "marin.evaluation.group_runner.resolve_inference_launch",
        lambda config: SimpleNamespace(
            model=None,
            engine=None,
            iris=None,
            instances=1,
            broker=None,
            endpoint_access=0,
        ),
    )
    monkeypatch.setattr("marin.evaluation.group_runner.configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(
        "marin.evaluation.group_runner.iris_ctx",
        lambda: SimpleNamespace(job_id="/group"),
    )

    def write(record, prefix):
        assert prefix == "gs://bucket/runs"
        records.append(record)
        return f"{prefix}/{record.run_id}/record.json"

    monkeypatch.setattr("marin.evaluation.group_runner.write_record", write)


def test_endpoint_move_retries_once_and_records_success(monkeypatch):
    records = []
    session = _Session()
    _patch_runtime(monkeypatch, session, records)
    calls = 0

    def evalchemy(model, config, output_dir, observer):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise EndpointMovedError(
                "moved",
                stage=PipelineStage.EVAL,
                jobs={"eval": "/stale"},
                log_tails={},
            )
        return EvalchemyOutcome(jobs={"eval": "/fresh"}, result=_Result())

    monkeypatch.setattr("marin.evaluation.group_runner.run_evalchemy", evalchemy)

    paths = run_eval_group(_group(_run("one")))

    assert paths == ["gs://bucket/runs/run-one/record.json"]
    assert session.wait_count == 1
    assert records[0].status is RunStatus.SUCCEEDED
    assert records[0].jobs["eval"] == "/fresh"


def test_eval_failure_is_recorded_and_later_runs_continue(monkeypatch):
    records = []
    _patch_runtime(monkeypatch, _Session(), records)

    def evalchemy(model, config, output_dir, observer):
        if config.name == "one":
            raise EvalPipelineError(
                "bad answer",
                stage=PipelineStage.EVAL,
                jobs={"eval": "/failed"},
                log_tails={"eval": ("tail",)},
            )
        return EvalchemyOutcome(jobs={"eval": "/succeeded"}, result=_Result())

    monkeypatch.setattr("marin.evaluation.group_runner.run_evalchemy", evalchemy)

    with pytest.raises(RuntimeError, match="1 of 2 evals failed"):
        run_eval_group(_group(_run("one"), _run("two")))

    assert [record.status for record in records] == [RunStatus.FAILED, RunStatus.SUCCEEDED]
    assert records[0].log_tails == {"eval": ("tail",)}
