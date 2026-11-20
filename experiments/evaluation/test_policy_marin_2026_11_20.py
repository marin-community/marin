# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 2026-11-20 policy resolves to runnable catalog selections and Harbor configs for every candidate."""

import pytest
import yaml
from evalrecords.records import EvaluationSelection, PromptFormat
from marin.evaluation.harbor.driver_config import preflight_harbor_configs

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.evals import HarborDefinition, harbor_model_agent_kwargs
from experiments.evaluation.policies.marin_2026_11_20.policy import POLICY
from experiments.evaluation.policy.resolve import resolve

CANDIDATES = pytest.mark.parametrize("candidate", POLICY.candidates, ids=lambda candidate: candidate.name)


@CANDIDATES
def test_taskspec_contexts_are_the_ones_the_server_and_protocol_enforce(candidate):
    """A TaskSpec run's window is the protocol's when it declares one, else the served window."""
    served = candidate.model.serve.max_model_len
    model_format = PromptFormat.CHAT if candidate.model.apply_chat_template else PromptFormat.COMPLETION
    for item in resolve(POLICY, candidate):
        selection = item.evaluation.definition
        if not isinstance(selection, EvaluationSelection):
            continue
        sources = evaluation_benchmarks()[selection.benchmark].resolve(
            selection.protocol, selection.num_fewshot, selection.prompt_format, model_format=model_format
        )
        config = sources[0].config
        assert config is not None
        assert item.context == (config.protocol.context_window or served), item.evaluation.name


@CANDIDATES
def test_harbor_limits_match_their_configs_and_fit_the_server(candidate):
    served = candidate.model.serve.max_model_len
    for item in resolve(POLICY, candidate):
        definition = item.evaluation.definition
        if not isinstance(definition, HarborDefinition):
            continue
        (agent,) = yaml.safe_load(definition.config_path.read_text())["agents"]
        model_info = agent["kwargs"]["model_info"]
        assert (model_info["max_input_tokens"], model_info["max_output_tokens"]) == (item.context, item.max_output)
        assert item.context <= served, item.evaluation.name


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
