# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 2026-11-20 trial policy resolves to the catalog selections and windows each candidate runs with."""

import pytest
from evalrecords.records import PromptFormat, TaskEvidence

from experiments.evaluation.curated import agent_ref_for, resolve_selection, selection_dataset, settings_for
from experiments.evaluation.policies.marin_2026_11_20_trial.policy import POLICY
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve

CANDIDATES = pytest.mark.parametrize("candidate", POLICY.candidates, ids=lambda candidate: candidate.name)


def _enforced_context(item: ResolvedPolicyEvaluation) -> int:
    """The window the worker enforces: an agent's input limit, else the protocol's window, else the server's."""
    model = item.candidate.model
    selection = item.evaluation.definition
    model_format = PromptFormat.CHAT if model.apply_chat_template else PromptFormat.COMPLETION
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
