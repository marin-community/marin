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
WINDOW = 50_000
LARGE = PolicyCandidate(model=ModelConfig(name="large", location="org/large", revision=REVISION), max_context=200_000)
SMALL = PolicyCandidate(model=ModelConfig(name="small", location="org/small", revision=REVISION), max_context=30_000)

PLAIN = PolicyEvaluation("aime24")
SEEDED = PolicyEvaluation("math500", seed=7)
CAPPED = PolicyEvaluation("gpqa-diamond", max_output=20_000)
SMALL_CAPPED = PolicyOverride(candidate=SMALL, evaluation=CAPPED, max_output=10_000, reason="window too small")


def _policy(*evaluations: PolicyEvaluation, candidates=(LARGE, SMALL), **exceptions) -> EvalPolicy:
    return EvalPolicy(
        version="test",
        serving_window=WINDOW,
        candidates=candidates,
        in_distribution=evaluations or (PLAIN, SEEDED, CAPPED),
        **exceptions,
    )


def _settings(resolved: tuple[ResolvedPolicyEvaluation, ...]) -> dict[str, tuple[int, int | None, int | None]]:
    return {item.evaluation.name: (item.context, item.max_output, item.seed) for item in resolved}


def test_a_candidate_runs_every_evaluation_at_its_serving_window():
    assert _settings(resolve(_policy(), LARGE)) == {
        "aime24": (WINDOW, None, None),
        "math500": (WINDOW, None, 7),
        "gpqa-diamond": (WINDOW, 20_000, None),
    }


def test_a_smaller_candidate_is_served_at_its_own_window_and_an_override_sets_its_output_limit():
    assert _settings(resolve(_policy(overrides=(SMALL_CAPPED,)), SMALL)) == {
        "aime24": (30_000, None, None),
        "math500": (30_000, None, 7),
        "gpqa-diamond": (30_000, 10_000, None),
    }


def test_the_catalog_serve_window_caps_the_serving_window():
    served = PolicyCandidate(
        model=ModelConfig(
            name="served", location="org/served", revision=REVISION, serve=ServeConfig(max_model_len=40_000)
        ),
        max_context=200_000,
    )
    assert _settings(resolve(_policy(PLAIN, candidates=(served,)), served)) == {"aime24": (40_000, None, None)}


def test_a_protocol_window_is_used_exactly_and_a_candidate_served_below_it_is_excluded():
    windowed = PolicyEvaluation("mrcr", protocol="32k")
    assert _settings(resolve(_policy(windowed), LARGE)) == {"mrcr-32k": (32_704, None, None)}
    with pytest.raises(ValueError, match="small on mrcr-32k"):
        resolve(_policy(windowed), SMALL)
    excluded = PolicyExclusion(candidate=SMALL, evaluation=windowed, reason="window")
    assert _settings(resolve(_policy(windowed, exclusions=(excluded,)), SMALL)) == {}


def test_an_agent_runs_at_the_smaller_of_its_input_limit_and_the_serving_window():
    agent = PolicyEvaluation("ds-1000", protocol="terminus-2")
    assert _settings(resolve(_policy(agent), LARGE)) == {"ds-1000-terminus-2": (32_768, None, None)}
    assert _settings(resolve(_policy(agent), SMALL)) == {"ds-1000-terminus-2": (30_000, None, None)}
    # An agent without a declared input limit, like Terminal-Bench 2.0's, runs at the serving window.
    uncapped = PolicyEvaluation("terminal-bench-2.0")
    assert _settings(resolve(_policy(uncapped), LARGE)) == {"terminal-bench-2.0": (WINDOW, None, None)}


def test_an_output_limit_must_fit_the_context():
    with pytest.raises(ValueError, match="small on gpqa-diamond"):
        resolve(_policy(replace(CAPPED, max_output=40_000)), SMALL)


def test_exceptions_apply_only_to_the_named_candidate():
    policy = _policy(
        exclusions=(PolicyExclusion(candidate=SMALL, evaluation=SEEDED, reason="unsupported"),),
        overrides=(SMALL_CAPPED,),
    )
    assert _settings(resolve(policy, SMALL)) == {"aime24": (30_000, None, None), "gpqa-diamond": (30_000, 10_000, None)}
    assert _settings(resolve(policy, LARGE)) == _settings(resolve(_policy(), LARGE))


@pytest.mark.parametrize(
    "exceptions",
    [
        pytest.param(
            {"overrides": (SMALL_CAPPED, replace(SMALL_CAPPED, max_output=5_000))}, id="two-overrides-for-one-pair"
        ),
        pytest.param(
            {
                "exclusions": (PolicyExclusion(candidate=SMALL, evaluation=CAPPED, reason="unsupported"),),
                "overrides": (SMALL_CAPPED,),
            },
            id="excluded-and-overridden",
        ),
        pytest.param(
            {"overrides": (replace(SMALL_CAPPED, evaluation=replace(CAPPED, max_output=5_000)),)},
            id="stale-copy-of-an-evaluation",
        ),
    ],
)
def test_ambiguous_or_stale_exceptions_are_rejected(exceptions):
    with pytest.raises(ValueError):
        _policy(**exceptions)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"benchmark": "no-such-benchmark"}, id="unknown-benchmark"),
        pytest.param({"benchmark": "mrcr", "protocol": "no-such"}, id="undeclared-variant"),
        pytest.param({"benchmark": "gsm8k", "num_fewshot": 3}, id="undeclared-few-shot-count"),
        pytest.param({"benchmark": "gsm8k", "protocol": ""}, id="empty-protocol"),
        pytest.param({"benchmark": "financebench"}, id="judged-benchmark-without-a-judge"),
        pytest.param({"benchmark": "gsm8k", "judge": "together"}, id="judge-on-an-unjudged-benchmark"),
        pytest.param({"benchmark": "financebench", "judge": "nobody"}, id="unknown-judge"),
    ],
)
def test_an_evaluation_the_catalog_cannot_run_as_declared_is_rejected(fields):
    with pytest.raises(ValueError):
        PolicyEvaluation(**fields)
