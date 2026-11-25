# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from marin.evaluation.model_config import ModelConfig

from experiments.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve
from experiments.evaluation.policy.schema import (
    EvalPolicy,
    PolicyCandidate,
    PolicyEvaluation,
    PolicyExclusion,
    PolicyOverride,
)

REVISION = "0" * 40
LARGE = PolicyCandidate(model=ModelConfig(name="large", location="org/large", revision=REVISION), max_context=200_000)
SMALL = PolicyCandidate(model=ModelConfig(name="small", location="org/small", revision=REVISION), max_context=60_000)

CONTEXT = 50_000
# Real catalog benchmarks, since a context left unset comes from the benchmark's declared window.
PLAIN = PolicyEvaluation("aime24")
SEEDED = PolicyEvaluation("math500", seed=7)
LONG = PolicyEvaluation("gpqa-diamond", context=100_000, max_output=30_000)
SMALL_LONG = PolicyOverride(
    candidate=SMALL, evaluation=LONG, context=60_000, max_output=10_000, reason="window too small"
)


def _policy(*evaluations: PolicyEvaluation, **exceptions) -> EvalPolicy:
    return EvalPolicy(
        version="test",
        context=CONTEXT,
        candidates=(LARGE, SMALL),
        in_distribution=evaluations or (PLAIN, SEEDED, LONG),
        **exceptions,
    )


def _settings(resolved: tuple[ResolvedPolicyEvaluation, ...]) -> dict[str, tuple[int, int | None, int | None]]:
    return {item.evaluation.name: (item.context, item.max_output, item.seed) for item in resolved}


def test_candidate_runs_every_evaluation_with_its_declared_settings():
    assert _settings(resolve(_policy(), LARGE)) == {
        "aime24": (50_000, None, None),
        "math500": (50_000, None, 7),
        "gpqa-diamond": (100_000, 30_000, None),
    }


def test_an_unset_context_is_the_benchmarks_window_and_a_different_one_is_rejected():
    windowed = PolicyEvaluation("mrcr", protocol="32k")
    assert _settings(resolve(_policy(windowed), LARGE)) == {"mrcr-32k": (32_704, None, None)}
    with pytest.raises(ValueError, match="mrcr-32k"):
        resolve(_policy(replace(windowed, context=30_000)), LARGE)


def test_an_override_may_run_an_agent_below_its_input_limit():
    # The worker gives an agent the smaller of its input limit and the served window.
    agent = PolicyEvaluation("ds-1000", protocol="terminus-2")
    tiny = PolicyCandidate(model=ModelConfig(name="tiny", location="org/tiny", revision=REVISION), max_context=16_384)
    policy = EvalPolicy(version="test", context=CONTEXT, candidates=(LARGE, tiny), in_distribution=(agent,))
    with pytest.raises(ValueError, match="tiny on ds-1000"):
        resolve(policy, tiny)

    smaller = PolicyOverride(candidate=tiny, evaluation=agent, context=16_384, reason="its window")
    assert _settings(resolve(replace(policy, overrides=(smaller,)), tiny)) == {
        "ds-1000-terminus-2": (16_384, None, None)
    }
    above = PolicyOverride(candidate=LARGE, evaluation=agent, context=40_000, reason="more room")
    with pytest.raises(ValueError, match="large on ds-1000"):
        resolve(replace(policy, overrides=(above,)), LARGE)


def test_a_candidate_too_small_for_a_protocol_window_must_be_excluded():
    windowed = PolicyEvaluation("mrcr", protocol="65k")
    with pytest.raises(ValueError, match="small on mrcr-65k"):
        resolve(_policy(windowed), SMALL)
    excluded = PolicyExclusion(candidate=SMALL, evaluation=windowed, reason="window too small")
    assert _settings(resolve(_policy(windowed, exclusions=(excluded,)), SMALL)) == {}


def test_context_beyond_the_candidate_window_needs_an_override():
    with pytest.raises(ValueError, match="small on gpqa-diamond"):
        resolve(_policy(), SMALL)
    with pytest.raises(ValueError, match="small on gpqa-diamond"):
        resolve(_policy(overrides=(replace(SMALL_LONG, context=20_000, max_output=None),)), SMALL)

    assert _settings(resolve(_policy(overrides=(SMALL_LONG,)), SMALL)) == {
        "aime24": (50_000, None, None),
        "math500": (50_000, None, 7),
        "gpqa-diamond": (60_000, 10_000, None),
    }


def test_exceptions_apply_only_to_the_named_candidate():
    policy = _policy(
        exclusions=(PolicyExclusion(candidate=SMALL, evaluation=SEEDED, reason="unsupported"),),
        overrides=(SMALL_LONG,),
    )

    assert _settings(resolve(policy, SMALL)) == {"aime24": (50_000, None, None), "gpqa-diamond": (60_000, 10_000, None)}
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
