# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from marin.evaluation.model_config import ModelConfig, ServeConfig

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


def test_an_agent_runs_at_the_smallest_of_its_limit_the_request_and_the_served_window():
    # The worker gives an agent the smaller of its input limit and the served window.
    agent = PolicyEvaluation("ds-1000", protocol="terminus-2")
    tiny = PolicyCandidate(model=ModelConfig(name="tiny", location="org/tiny", revision=REVISION), max_context=16_384)
    policy = EvalPolicy(version="test", context=CONTEXT, candidates=(LARGE, tiny), in_distribution=(agent,))
    assert _settings(resolve(policy, tiny)) == {"ds-1000-terminus-2": (16_384, None, None)}
    assert _settings(resolve(policy, LARGE)) == {"ds-1000-terminus-2": (32_768, None, None)}

    smaller = PolicyOverride(candidate=LARGE, evaluation=agent, context=20_000, reason="less room")
    assert _settings(resolve(replace(policy, overrides=(smaller,)), LARGE)) == {
        "ds-1000-terminus-2": (20_000, None, None)
    }


def test_a_candidate_served_below_a_protocol_window_must_be_excluded():
    # The candidate is served at 32,768 of a native 200,000; MRCR-65k's items need 65,472.
    served = PolicyCandidate(
        model=ModelConfig(
            name="served", location="org/served", revision=REVISION, serve=ServeConfig(max_model_len=32_768)
        ),
        max_context=200_000,
    )
    windowed, agent = PolicyEvaluation("mrcr", protocol="65k"), PolicyEvaluation(
        "swebench-verified", protocol="mini-swe-agent"
    )
    policy = EvalPolicy(version="test", context=CONTEXT, candidates=(served,), in_distribution=(windowed, agent))
    with pytest.raises(ValueError, match="served on mrcr-65k"):
        resolve(policy, served)

    excluded = replace(policy, exclusions=(PolicyExclusion(candidate=served, evaluation=windowed, reason="window"),))
    assert _settings(resolve(excluded, served)) == {"swebench-verified-mini-swe-agent": (32_768, None, None)}


def test_an_agent_without_an_input_limit_runs_at_the_policy_context():
    assert _settings(resolve(_policy(PolicyEvaluation("terminal-bench-2.0")), LARGE)) == {
        "terminal-bench-2.0": (CONTEXT, None, None)
    }


def test_a_candidate_runs_at_most_at_its_window_and_overrides_still_apply():
    assert _settings(resolve(_policy(), SMALL)) == {
        "aime24": (50_000, None, None),
        "math500": (50_000, None, 7),
        "gpqa-diamond": (60_000, 30_000, None),
    }
    assert _settings(resolve(_policy(overrides=(SMALL_LONG,)), SMALL))["gpqa-diamond"] == (60_000, 10_000, None)
    # An override's output limit must still fit the context it leaves.
    with pytest.raises(ValueError, match="small on gpqa-diamond"):
        resolve(_policy(overrides=(replace(SMALL_LONG, context=20_000, max_output=None),)), SMALL)


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
