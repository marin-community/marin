# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launcher behavior at the experiment-to-library boundary."""

from marin.evaluation.hardware import Platform
from marin.evaluation.records import Provenance

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


def test_batch_uses_the_endpoint_origin_of_the_routed_cluster(monkeypatch):
    clusters = []
    monkeypatch.setattr(
        "experiments.evaluation.launch._endpoint_origin",
        lambda cluster: clusters.append(cluster) or f"https://{cluster}.example",
    )

    batch = build_evaluation_batch(
        _spec("qwen3-0.6b", "mmlu-smoke", platform=Platform.GPU, accelerator="H100x1"),
        _provenance(),
        "tester",
    )

    assert clusters == ["cw-us-east-02a"]
    assert batch.target_cluster == "cw-us-east-02a"
    assert batch.serving.endpoint_origin == "https://cw-us-east-02a.example"
    assert batch.hardware_ref.accelerator == "H100x1"


def test_harbor_runner_receives_model_agent_settings(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._endpoint_origin", lambda cluster: "https://iris.example")

    batch = build_evaluation_batch(_spec("qwen3-8b", "tb2-lite"), _provenance(), "tester")

    (evaluation,) = batch.evaluations
    assert evaluation.runner.config.dataset == "hf://DCAgent2/terminal_bench_2"
    assert evaluation.runner.config.task_limit == 1
    assert "enable_thinking" in evaluation.runner.config.agent_kwargs["extra_body"]
    assert batch.serving.api_model == "qwen3-8b"


def test_batch_preserves_model_revision_and_allows_mixed_runners(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._endpoint_origin", lambda cluster: "https://iris.example")

    batch = build_evaluation_batch(
        _spec("llama-3.1-8b-base", "mmlu-smoke", "tb2-lite"),
        _provenance(),
        "tester",
    )

    assert batch.serving.revision == "d04e592"
    assert [evaluation.runner.mechanism for evaluation in batch.evaluations] == ["evalchemy", "harbor"]
