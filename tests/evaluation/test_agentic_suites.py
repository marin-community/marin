# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agentic benchmarks route through the group launcher as Harbor suites.

The benchmarks absorbed from #7246 (tb2, swebench, ...) are Hugging Face repositories of Harbor task
directories: the launcher serves the model once and drives an in-sandbox agent against them, exactly
like the existing Harbor path. These check the registry wiring and that a model's agent_kwargs reach
the Harbor run -- the pure planning pieces that need no cluster.
"""

from dataclasses import replace

from iris.cluster.types import EndpointAccess
from marin.evaluation.records import Provenance

from experiments.evaluation.evals import EVALS, SUITES, EvalMechanism
from experiments.evaluation.hardware import Platform
from experiments.evaluation.launch import LaunchSpec, _group_params, plan_runs


def test_agentic_suites_are_harbor_datasets():
    for key in SUITES["agentic"]:
        suite = EVALS[key]
        assert suite.mechanism is EvalMechanism.HARBOR
        assert suite.harbor is not None
        # Every agentic benchmark runs its trials in a Daytona sandbox against the served endpoint.
        assert suite.harbor.env == "daytona"
        assert suite.harbor.dataset.startswith("hf://")
        assert suite.harbor.version == "main"


def test_tb2_lite_caps_instances_for_validation():
    lite = EVALS["tb2-lite"]
    assert lite.harbor.dataset == "hf://DCAgent2/terminal_bench_2"
    assert lite.max_eval_instances == 2


def _provenance() -> Provenance:
    return Provenance(git_sha="abc", eval_image="img", launch_host="host")


def test_model_agent_kwargs_flow_into_the_harbor_run():
    # qwen3-8b carries enable_thinking in its AgentConfig; a Harbor run of it must forward that to the
    # agent, so a launch does not have to restate per-model agent settings on every benchmark.
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("tb2-lite",),
        platform=Platform.TPU,
        accelerator=None,
        limit=None,
        records_prefix="gs://bucket/runs",
        cluster="marin",
    )
    params = _group_params(plan_runs(spec), spec, _provenance(), "tester")
    (run,) = params.runs
    assert run.harbor is not None
    assert "enable_thinking" in run.harbor.agent_kwargs["extra_body"]
    assert params.session.serve.endpoint_access == EndpointAccess.ENDPOINT_ACCESS_LINK


def test_suite_level_agent_kwargs_override_model_level():
    # A model-level agent kwarg is the default; a suite that sets the same key wins.
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("tb2-override",),
        platform=Platform.TPU,
        accelerator=None,
        limit=None,
        records_prefix="gs://bucket/runs",
        cluster="marin",
    )
    EVALS["tb2-override"] = replace(
        EVALS["tb2-lite"],
        name="tb2-override",
        harbor=replace(EVALS["tb2-lite"].harbor, agent_kwargs={"extra_body": "SUITE"}),
    )
    try:
        params = _group_params(plan_runs(spec), spec, _provenance(), "tester")
        (run,) = params.runs
        assert run.harbor.agent_kwargs["extra_body"] == "SUITE"
    finally:
        del EVALS["tb2-override"]
