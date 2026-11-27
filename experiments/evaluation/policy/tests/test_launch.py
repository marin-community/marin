# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A policy launch puts each model on one server at the policy's window, and leaves out what a model can't run."""

from dataclasses import replace
from types import MappingProxyType

import pytest
from evalrecords.records import EvaluationOverrides
from iris.rpc import job_pb2
from marin.evaluation.hardware import Platform

from experiments.evaluation.policy.launch import (
    EVALUATIONS_PER_SERVER,
    evaluation_context,
    launch_specs,
    model_evaluations,
    policy_model,
    run_key,
    server_groups,
)
from experiments.evaluation.policy.versions import POLICIES

POLICY = POLICIES["marin-2026-11-20-trial"]
QWEN_NAME = "qwen3.6-35b-a3b"
EVALUATIONS = {evaluation.name: evaluation for evaluation in POLICY.evaluations}


def _with_run(evaluation, **settings):
    run = EvaluationOverrides.model_validate({**evaluation.run.model_dump(), **settings})
    return evaluation.model_copy(update={"run": run})


def test_a_model_runs_every_evaluation_on_one_server_at_the_policys_window(qwen):
    plan = model_evaluations(POLICY, qwen)
    suffixes = {evaluation.name: "0123456789ab-a1" for evaluation in plan.evaluations}
    (spec,) = launch_specs(POLICY, plan, limit=20, priority_band=job_pb2.PRIORITY_BAND_BATCH, run_suffixes=suffixes)
    assert len(spec.evals) == len(POLICY.evaluations)
    assert spec.model.serve.max_model_len == POLICY.serving_window
    assert spec.max_concurrent == EVALUATIONS_PER_SERVER


def test_judges_with_different_keys_get_separate_servers():
    # A launch hands its judges one key; evaluations without a judge join the first server.
    other_key = _with_run(EVALUATIONS["olympiadbench"], judge_api_key=("env:OTHER",))
    groups = {
        tuple(evaluation.name for evaluation in group)
        for group in server_groups([EVALUATIONS["gsm8k-0shot"], EVALUATIONS["financebench"], other_key])
    }
    assert groups == {("gsm8k-0shot", "financebench"), ("olympiadbench",)}


def test_a_protocol_window_is_used_exactly_and_an_agent_gets_the_smaller_of_its_limit_and_the_window(qwen):
    assert evaluation_context(qwen, EVALUATIONS["mrcr-32k"]) == 32_704
    assert evaluation_context(qwen, EVALUATIONS["swebench-verified"]) == 32_768
    assert evaluation_context(qwen, EVALUATIONS["aime24"]) == POLICY.serving_window


def test_a_baseline_served_below_a_protocol_window_must_exclude_it(served):
    # At 65,536 tokens, MRCR's 73,728-token protocol and NUPA's longest window don't fit.
    served(65_536)
    model = policy_model(POLICY, POLICY.baseline(QWEN_NAME), baseline=True)
    with pytest.raises(ValueError, match="exclude it"):
        model_evaluations(POLICY, model)
    reasons = {(QWEN_NAME, "mrcr"): "too long", (QWEN_NAME, "nupa"): "too long"}
    plan = model_evaluations(replace(POLICY, exclusions=MappingProxyType(reasons)), model)
    assert plan.excluded == {"mrcr": "too long", "nupa": "too long"}


def test_a_supplied_model_leaves_out_what_it_cant_run(served):
    served(32_768)
    checkpoint = policy_model(POLICY, replace(POLICY.baseline(QWEN_NAME), name="qwen-step-100"), baseline=False)
    plan = model_evaluations(POLICY, checkpoint)
    assert set(plan.excluded) == {"mmlu-pro", "mrcr", "mrcr-65k", "nupa"}
    assert "mrcr-32k" in {evaluation.name for evaluation in plan.evaluations}


def test_an_output_limit_must_fit_the_context(qwen):
    aime = EVALUATIONS["aime24"]
    capped = _with_run(aime, max_tokens=80_000)
    policy = replace(
        POLICY, evaluations=tuple(capped if evaluation is aime else evaluation for evaluation in POLICY.evaluations)
    )
    with pytest.raises(ValueError, match="aime24"):
        model_evaluations(policy, qwen)


def test_rewriting_an_object_store_checkpoint_changes_its_run_key(served, tmp_path):
    weights = tmp_path / "checkpoint" / "shards" / "model.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    # Training checkpoints carry a producer identity, and shards can sit in a subdirectory; the digest covers both.
    checkpoint = replace(
        POLICY.baseline(QWEN_NAME),
        name="qwen-step-100",
        location=f"file://{tmp_path / 'checkpoint'}",
        identity="producer",
    )
    before = policy_model(POLICY, checkpoint, baseline=False)
    weights.write_bytes(b"retrained weights")
    after = policy_model(POLICY, checkpoint, baseline=False)
    assert run_key(before, Platform.GPU) != run_key(after, Platform.GPU)
