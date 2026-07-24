# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stable launcher behavior that the library-boundary refactor must preserve."""

from iris.cluster.types import EndpointAccess
from marin.evaluation.definitions import ResolvedHarborRun
from marin.evaluation.hardware import Platform
from marin.evaluation.records import Provenance

from experiments.evaluation.launch import LaunchSpec, _group_params, plan_runs


def _spec(eval_key: str) -> LaunchSpec:
    return LaunchSpec(
        model="qwen3-0.6b",
        evals=(eval_key,),
        platform=Platform.TPU,
        accelerator=None,
        limit=1,
        records_prefix="gs://bucket/runs",
        cluster="marin",
    )


def test_evalchemy_plan_resolves_model_suite_and_slice():
    (plan,) = plan_runs(_spec("mmlu-smoke"))

    assert (plan.model_key, plan.eval_key) == ("qwen3-0.6b", "mmlu-smoke")
    assert plan.model.location == "Qwen/Qwen3-0.6B"
    assert plan.accel.tpu_type == "v5litepod-4"
    assert plan.accel.region == "us-west4"
    assert plan.serve.tpu_type == "v5litepod-4"
    assert plan.serve.endpoint_access == EndpointAccess.ENDPOINT_ACCESS_PRIVATE
    assert plan.limit == 1


def test_harbor_group_preserves_public_served_identity(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.launch._mint_origin_for", lambda cluster: "https://iris.example")
    provenance = Provenance(git_sha="abc", eval_image="img", launch_host="host")

    params = _group_params(plan_runs(_spec("tb2-lite")), _spec("tb2-lite"), provenance, "tester")

    assert params.capability_origin == "https://iris.example"
    assert params.serving.spec.endpoint_access == EndpointAccess.ENDPOINT_ACCESS_LINK
    assert params.serving.served_model_name == "qwen3-0.6b"
    (run,) = params.runs
    assert isinstance(run, ResolvedHarborRun)
    assert run.identity.output_dir.startswith("gs://bucket/runs/")
    assert run.identity.output_dir.endswith("/results")
    assert run.identity.eval_ref.name == "tb2-lite"
    assert run.config.dataset == "hf://DCAgent2/terminal_bench_2"
    assert run.config.task_limit == 1
