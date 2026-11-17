# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from pathlib import Path

import pytest
from evalrecords.records import EvaluationSelection
from marin.evaluation.model_config import ModelConfig

from experiments.evaluation.evals import HarborDefinition
from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve
from experiments.evaluation.policy.schema import (
    EvalPolicy,
    PolicyCandidate,
    PolicyEvaluation,
    PolicyExclusion,
    PolicyOverride,
)

LARGE = PolicyCandidate(model=ModelConfig(name="large", location="org/large"), max_context=200_000)
SMALL = PolicyCandidate(model=ModelConfig(name="small", location="org/small"), max_context=60_000)

CONTEXT = 50_000
PLAIN = PolicyEvaluation(definition=EvaluationSelection(benchmark="plain"), context=CONTEXT, max_output=None, seed=0)
SEEDED = PolicyEvaluation(definition=EvaluationSelection(benchmark="seeded"), context=CONTEXT, max_output=None, seed=7)
LONG = PolicyEvaluation(
    definition=HarborDefinition(name="long", config_path=Path("long.yaml")), context=100_000, max_output=30_000, seed=0
)
SMALL_LONG = PolicyOverride(
    candidate=SMALL, evaluation=LONG, context=60_000, max_output=10_000, reason="window too small"
)


def _policy(**exceptions) -> EvalPolicy:
    return EvalPolicy(version="test", candidates=(LARGE, SMALL), in_distribution=(PLAIN, SEEDED, LONG), **exceptions)


def _settings(resolved: tuple[ResolvedPolicyEvaluation, ...]) -> dict[str, tuple[int, int | None, int]]:
    return {item.evaluation.name: (item.context, item.max_output, item.seed) for item in resolved}


def test_candidate_runs_every_evaluation_with_its_declared_settings():
    assert _settings(resolve(_policy(), LARGE)) == {
        "plain": (50_000, None, 0),
        "seeded": (50_000, None, 7),
        "long": (100_000, 30_000, 0),
    }


def test_context_beyond_the_candidate_window_needs_an_override():
    with pytest.raises(ValueError, match="small on long"):
        resolve(_policy(), SMALL)
    with pytest.raises(ValueError, match="small on long"):
        resolve(_policy(overrides=(replace(SMALL_LONG, context=20_000, max_output=None),)), SMALL)

    assert _settings(resolve(_policy(overrides=(SMALL_LONG,)), SMALL)) == {
        "plain": (50_000, None, 0),
        "seeded": (50_000, None, 7),
        "long": (60_000, 10_000, 0),
    }


def test_exceptions_apply_only_to_the_named_candidate():
    policy = _policy(
        exclusions=(PolicyExclusion(candidate=SMALL, evaluation=SEEDED, reason="unsupported"),),
        overrides=(SMALL_LONG,),
    )

    assert _settings(resolve(policy, SMALL)) == {"plain": (50_000, None, 0), "long": (60_000, 10_000, 0)}
    assert _settings(resolve(policy, LARGE)) == _settings(resolve(_policy(), LARGE))


@pytest.mark.parametrize(
    "exceptions",
    [
        pytest.param(
            {"overrides": (SMALL_LONG, replace(SMALL_LONG, max_output=5_000))}, id="two-overrides-for-one-pair"
        ),
        pytest.param(
            {
                "exclusions": (PolicyExclusion(candidate=SMALL, evaluation=LONG, reason="unsupported"),),
                "overrides": (SMALL_LONG,),
            },
            id="excluded-and-overridden",
        ),
        pytest.param(
            {"overrides": (replace(SMALL_LONG, evaluation=replace(LONG, max_output=5_000)),)},
            id="stale-copy-of-an-evaluation",
        ),
    ],
)
def test_ambiguous_or_stale_exceptions_are_rejected(exceptions):
    with pytest.raises(ValueError):
        _policy(**exceptions)
