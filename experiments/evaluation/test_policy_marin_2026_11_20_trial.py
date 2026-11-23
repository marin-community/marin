# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 2026-11-20 trial policy resolves to runnable catalog selections and Harbor configs for every candidate."""

import pytest
from evalrecords.records import EvaluationSelection
from marin.evaluation.harbor.driver_config import preflight_harbor_configs

from experiments.evaluation.evals import HarborDefinition, harbor_model_agent_kwargs
from experiments.evaluation.policies.marin_2026_11_20_trial.policy import POLICY
from experiments.evaluation.policy.cli import selection_dataset
from experiments.evaluation.policy.resolve import resolve

CANDIDATES = pytest.mark.parametrize("candidate", POLICY.candidates, ids=lambda candidate: candidate.name)


@CANDIDATES
def test_taskspec_contexts_are_the_ones_the_server_and_protocol_enforce(candidate):
    """A TaskSpec run's window is the protocol's when it declares one, else the served window."""
    served = candidate.model.serve.max_model_len
    for item in resolve(POLICY, candidate):
        selection = item.evaluation.definition
        if not isinstance(selection, EvaluationSelection):
            continue
        config = selection_dataset(selection, candidate.model)
        assert item.context == (config.protocol.context_window or served), item.evaluation.name


@CANDIDATES
def test_every_evaluation_fits_the_served_window(candidate):
    served = candidate.model.serve.max_model_len
    assert all(item.context <= served for item in resolve(POLICY, candidate))


@pytest.mark.integration
@pytest.mark.timeout(600)
@CANDIDATES
def test_harbor_configs_pass_preflight_with_the_candidate(candidate, monkeypatch):
    monkeypatch.setenv("DAYTONA_TARGET", "policy-test")
    harbor = [item for item in resolve(POLICY, candidate) if isinstance(item.evaluation.definition, HarborDefinition)]
    kwargs = harbor_model_agent_kwargs(candidate.model)
    configs = preflight_harbor_configs(
        [(item.evaluation.definition.config_path, kwargs) for item in harbor], judge_model=None
    )
    assert [(config.max_input_tokens, config.max_output_tokens) for config in configs] == [
        (item.context, item.max_output) for item in harbor
    ]
