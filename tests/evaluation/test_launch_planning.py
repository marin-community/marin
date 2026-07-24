# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launcher behavior at the experiment-to-library boundary."""

from marin.evaluation.hardware import Platform
from marin.evaluation.records import Provenance
from marin.evaluation.runner import EvalchemyExecutor, HarborExecutor

from experiments.evaluation.launch import LaunchSpec, build_evaluation_batch


def _provenance() -> Provenance:
    return Provenance(git_sha="abc", eval_image="img", launch_host="host")


def _spec(model: str, *evals: str, platform: Platform = Platform.TPU, accelerator: str | None = None) -> LaunchSpec:
    return LaunchSpec(
        model=model,
        evals=evals,
        platform=platform,
        accelerator=accelerator,
        limit=1,
        records_prefix="gs://bucket/runs",
        cluster="marin",
    )


def test_batch_uses_the_capability_origin_of_the_routed_cluster(monkeypatch):
    monkeypatch.setattr(
        "experiments.evaluation.launch._capability_origin",
        lambda cluster: f"https://{cluster}.example",
    )

    batch = build_evaluation_batch(
        _spec("qwen3-0.6b", "mmlu-smoke", platform=Platform.GPU, accelerator="H100x1"),
        _provenance(),
        "tester",
    )

    assert batch.accelerator.target_cluster == "cw-us-east-02a"
    assert batch.capability_origin == "https://cw-us-east-02a.example"
    assert batch.accelerator.label == "H100x1"


def test_harbor_executor_receives_model_agent_settings(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda cluster: "https://iris.example")

    batch = build_evaluation_batch(_spec("qwen3-8b", "tb2-lite"), _provenance(), "tester")

    (evaluation,) = batch.evaluations
    assert isinstance(evaluation.executor, HarborExecutor)
    assert evaluation.executor.config.dataset == "hf://DCAgent2/terminal_bench_2"
    assert evaluation.executor.config.task_limit == 1
    assert "enable_thinking" in evaluation.executor.config.agent_kwargs["extra_body"]
    assert batch.api_model == "qwen3-8b"


def test_evalchemy_definition_merges_model_generation_settings(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._capability_origin", lambda cluster: "https://iris.example")

    batch = build_evaluation_batch(
        _spec("snowball-sft", "mmlu-smoke", platform=Platform.GPU),
        _provenance(),
        "tester",
    )

    (evaluation,) = batch.evaluations
    assert isinstance(evaluation.executor, EvalchemyExecutor)
    assert evaluation.executor.config.extra_gen_kwargs == {
        "skip_special_tokens": "false",
        "repetition_penalty": "1.1",
    }
