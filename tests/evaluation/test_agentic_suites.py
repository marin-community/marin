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
from marin.evaluation.definitions import HarborDefinition, ResolvedHarborRun
from marin.evaluation.hardware import Platform
from marin.evaluation.records import Provenance

from experiments.evaluation.evals import EVALS, SUITES
from experiments.evaluation.launch import LaunchSpec, _group_params, plan_runs


def test_agentic_suites_are_harbor_datasets():
    for key in SUITES["agentic"]:
        suite = EVALS[key]
        assert isinstance(suite, HarborDefinition)
        # Every agentic benchmark runs its trials in a Daytona sandbox against the served endpoint.
        assert suite.config.env == "daytona"
        assert suite.config.dataset.startswith("hf://")
        assert suite.config.version == "main"


def test_tb2_lite_caps_instances_for_validation():
    lite = EVALS["tb2-lite"]
    assert isinstance(lite, HarborDefinition)
    assert lite.config.dataset == "hf://DCAgent2/terminal_bench_2"
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
    assert isinstance(run, ResolvedHarborRun)
    assert "enable_thinking" in run.config.agent_kwargs["extra_body"]
    assert params.serving.spec.endpoint_access == EndpointAccess.ENDPOINT_ACCESS_LINK


def test_suite_level_agent_kwargs_override_model_level():
    # A model-level agent kwarg is the default; a suite that sets the same key wins.
    spec = LaunchSpec(
        model="qwen3-8b",
        evals=("tb2-lite",),
        platform=Platform.TPU,
        accelerator=None,
        limit=None,
        records_prefix="gs://bucket/runs",
        cluster="marin",
    )
    (plan,) = plan_runs(spec)
    definition = plan.suite
    assert isinstance(definition, HarborDefinition)
    override_plan = replace(
        plan,
        eval_key="tb2-override",
        suite=replace(
            definition,
            name="tb2-override",
            config=replace(definition.config, agent_kwargs={"extra_body": "SUITE"}),
        ),
    )
    params = _group_params([override_plan], spec, _provenance(), "tester")
    (run,) = params.runs
    assert isinstance(run, ResolvedHarborRun)
    assert run.config.agent_kwargs["extra_body"] == "SUITE"
