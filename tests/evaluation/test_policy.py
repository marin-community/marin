# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy declarations and their resolution into per-candidate settings."""

from dataclasses import replace

import pytest
from marin.evaluation.policy.resolve import ResolvedPolicyEvaluation, resolve
from marin.evaluation.policy.schema import (
    EvalPolicy,
    PolicyCandidate,
    PolicyDefaults,
    PolicyEvaluation,
    PolicyExclusion,
    PolicyOverride,
    VendoredEvalchemy,
    VendoredHarbor,
)

LARGE = PolicyCandidate(name="large", model="org/large", max_context=200_000)
SMALL = PolicyCandidate(name="small", model="org/small", max_context=30_000)

PLAIN = PolicyEvaluation(name="plain", run=VendoredEvalchemy("plain"))
SEEDED = PolicyEvaluation(name="seeded", run=VendoredEvalchemy("seeded"), seed=7)
LONG = PolicyEvaluation(name="long", run=VendoredHarbor("long"), context=100_000, max_output=30_000)
SMALL_LONG = PolicyOverride(candidate=SMALL, evaluation=LONG, max_output=10_000, reason="window too small")


def _policy(**exceptions) -> EvalPolicy:
    return EvalPolicy(
        version="test",
        defaults=PolicyDefaults(context=50_000, max_output=None, seed=0),
        candidates=(LARGE, SMALL),
        in_distribution=(PLAIN, SEEDED, LONG),
        **exceptions,
    )


def _settings(resolved: tuple[ResolvedPolicyEvaluation, ...]) -> dict[str, tuple[int, int | None, int]]:
    return {item.evaluation.name: (item.context, item.max_output, item.seed) for item in resolved}


def test_settings_fall_back_from_evaluation_to_policy_defaults():
    assert _settings(resolve(_policy(), LARGE)) == {
        "plain": (50_000, None, 0),
        "seeded": (50_000, None, 7),
        "long": (100_000, 30_000, 0),
    }


def test_candidate_window_caps_context_and_an_unfit_output_limit_needs_an_override():
    with pytest.raises(ValueError, match="small on long"):
        resolve(_policy(), SMALL)

    assert _settings(resolve(_policy(overrides=(SMALL_LONG,)), SMALL)) == {
        "plain": (30_000, None, 0),
        "seeded": (30_000, None, 7),
        "long": (30_000, 10_000, 0),
    }


def test_exceptions_apply_only_to_the_named_candidate():
    policy = _policy(
        exclusions=(PolicyExclusion(candidate=SMALL, evaluation=SEEDED, reason="unsupported"),),
        overrides=(SMALL_LONG,),
    )

    assert _settings(resolve(policy, SMALL)) == {"plain": (30_000, None, 0), "long": (30_000, 10_000, 0)}
    assert _settings(resolve(policy, LARGE)) == _settings(resolve(_policy(), LARGE))


@pytest.mark.parametrize(
    ("exceptions", "error"),
    [
        pytest.param(
            {"overrides": (SMALL_LONG, replace(SMALL_LONG, max_output=5_000))},
            "more than one exception",
            id="two-overrides-for-one-pair",
        ),
        pytest.param(
            {
                "exclusions": (PolicyExclusion(candidate=SMALL, evaluation=LONG, reason="unsupported"),),
                "overrides": (SMALL_LONG,),
            },
            "more than one exception",
            id="excluded-and-overridden",
        ),
        pytest.param(
            {"overrides": (replace(SMALL_LONG, evaluation=replace(LONG, max_output=5_000)),)},
            "not in this policy",
            id="stale-copy-of-an-evaluation",
        ),
    ],
)
def test_ambiguous_or_stale_exceptions_are_rejected(exceptions, error):
    with pytest.raises(ValueError, match=error):
        _policy(**exceptions)


def test_override_cannot_raise_context_beyond_the_candidate_window():
    with pytest.raises(ValueError, match="larger than the candidate's window"):
        replace(SMALL_LONG, context=50_000)
