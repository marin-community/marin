# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every policy version resolves to the windows the worker enforces, and the trial shows what each model runs."""

import pytest
from evalrecords.records import EvaluationSelection, TaskEvidence
from marin.evaluation.model_config import prompt_format

from experiments.evaluation.curated import agent_ref_for, resolve_selection, selection_dataset, settings_for
from experiments.evaluation.policy.cli import _row
from experiments.evaluation.policy.launch import PolicyModel, evaluation_context, model_evaluations, policy_model
from experiments.evaluation.policy.versions import POLICIES

POLICY = POLICIES["marin-2026-11-20-trial"]
EVERY_MODEL = pytest.mark.parametrize(
    ("version", "name"),
    [(policy.version, name) for policy in POLICIES.values() for name in policy.baselines],
)


def _enforced_context(model: PolicyModel, evaluation: EvaluationSelection) -> int:
    model_format = prompt_format(model.model)
    resolved = resolve_selection(evaluation, model_format)
    settings = settings_for(
        evaluation,
        resolved,
        model.model,
        limit=None,
        seed=evaluation.run.seed,
        default_seed=None,
        task_machine_backend=None,
        task_evidence=TaskEvidence.REFERENCE,
    )
    agent = agent_ref_for(resolved, model.model, settings)
    if agent is not None:
        return agent.max_input_tokens
    return selection_dataset(evaluation, model_format).protocol.context_window or model.window


@EVERY_MODEL
def test_each_context_is_the_window_the_worker_enforces(served, version, name):
    policy = POLICIES[version]
    model = policy_model(policy, policy.baseline(name), baseline=True)
    for evaluation in model_evaluations(policy, model).evaluations:
        assert evaluation_context(model, evaluation) == _enforced_context(model, evaluation), evaluation.name


def test_show_prints_the_thinking_arguments_each_model_is_sent(served):
    # Glimmer replaces the standard thinking-off switch with its own argument, as the worker sends it.
    gsm8k = next(evaluation for evaluation in POLICY.evaluations if evaluation.name == "gsm8k-0shot")
    think = {
        name: _row(policy_model(POLICY, POLICY.baseline(name), baseline=True), gsm8k).think for name in POLICY.baselines
    }
    assert think == {
        "muse-glimmer-30b": "reasoning_strength=low",
        "nvidia-nemotron-3.5-super-vl-120b-a12b-bf16": "off",
        "qwen3.6-35b-a3b": "off",
    }
