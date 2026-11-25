# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 2026-11-20 trial policy resolves to the catalog selections and windows each candidate runs with."""

import pytest
from evalrecords.records import TaskEvidence
from marin.evaluation.model_config import prompt_format

from experiments.evaluation.curated import agent_ref_for, resolve_selection, selection_dataset, settings_for
from experiments.evaluation.policy.cli import policy_rows
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve
from experiments.evaluation.policy.versions.marin_2026_11_20_trial.policy import POLICY

CANDIDATES = pytest.mark.parametrize("candidate", POLICY.candidates, ids=lambda candidate: candidate.name)


def _enforced_context(item: ResolvedPolicyEvaluation) -> int:
    model = item.candidate.model
    selection = item.evaluation.selection
    model_format = prompt_format(model)
    resolved = resolve_selection(selection, model_format)
    settings = settings_for(
        selection,
        resolved,
        model,
        limit=None,
        seed=item.seed,
        default_seed=None,
        task_machine_backend=None,
        task_evidence=TaskEvidence.REFERENCE,
    )
    agent = agent_ref_for(resolved, model, settings)
    if agent is not None:
        return agent.max_input_tokens
    return selection_dataset(selection, model_format).protocol.context_window or model.serve.max_model_len


@CANDIDATES
def test_each_context_is_the_window_the_worker_enforces(candidate):
    for item in resolve(POLICY, candidate):
        assert item.context == _enforced_context(item), item.evaluation.name


def test_show_prints_the_thinking_arguments_each_candidate_is_sent():
    # Glimmer replaces the standard thinking-off switch with its own argument, as the worker sends it.
    think = {
        candidate.name: next(row.think for row in policy_rows(POLICY, candidate) if row.evaluation == "gsm8k-0shot")
        for candidate in POLICY.candidates
    }
    assert think == {
        "muse-glimmer-30b": "reasoning_strength=low",
        "nvidia-nemotron-3.5-super-vl-120b-a12b-bf16": "off",
        "qwen3.6-35b-a3b": "off",
    }
