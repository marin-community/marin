# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy evaluations group onto servers whose window each run actually sees."""

from dataclasses import replace

import pytest
from iris.rpc import job_pb2

from experiments.evaluation.policies import POLICIES
from experiments.evaluation.policies.marin_2026_11_20_trial.policy import POLICY, QWEN
from experiments.evaluation.policy.launch import launch_specs, policy_selection, server_groups
from experiments.evaluation.policy.resolve import resolve

RESOLVED = {item.evaluation.name: item for item in resolve(POLICY, QWEN)}
SERVED = QWEN.model.serve.max_model_len


def _windows(names: list[str], **contexts: int) -> set[tuple[int, bool, tuple[str, ...]]]:
    items = [replace(RESOLVED[name], context=contexts.get(name, RESOLVED[name].context)) for name in names]
    return {
        (group.window, group.agentic, tuple(item.evaluation.name for item in group.evaluations))
        for group in server_groups(items)
    }


def test_benchmarks_that_enforce_their_window_share_the_served_window_apart_from_agents():
    assert _windows(["gsm8k-0shot", "mrcr-32k", "swebench-verified"]) == {
        (SERVED, False, ("gsm8k-0shot", "mrcr-32k")),
        (SERVED, True, ("swebench-verified",)),
    }


def test_a_smaller_unenforced_context_gets_a_server_of_exactly_that_window():
    assert _windows(["gsm8k-0shot", "aime24"], aime24=8192) == {
        (SERVED, False, ("gsm8k-0shot",)),
        (8192, False, ("aime24",)),
    }


def test_a_context_that_disagrees_with_the_benchmark_window_is_rejected():
    with pytest.raises(ValueError, match="mrcr-32k"):
        _windows(["mrcr-32k"], **{"mrcr-32k": 30_000})


def test_policy_output_limit_and_seed_become_run_settings():
    selection = policy_selection(replace(RESOLVED["mrcr-32k"], max_output=4096, seed=7))
    assert (selection.run.max_tokens, selection.run.seed) == (4096, 7)
    assert selection.protocol == "32k"


def test_a_launch_covers_only_the_named_evaluations():
    specs = launch_specs(
        POLICY,
        QWEN,
        evaluations={"aime24", "swebench-verified"},
        limit=20,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )
    assert sorted(selection.name for spec in specs for selection in spec.evals) == ["aime24", "swebench-verified"]


def test_unknown_evaluations_are_rejected():
    with pytest.raises(ValueError, match="bixbench"):
        launch_specs(POLICY, QWEN, evaluations={"bixbench"}, limit=None, priority_band=job_pb2.PRIORITY_BAND_BATCH)


@pytest.mark.parametrize(
    ("policy", "candidate"),
    [(policy, candidate) for policy in POLICIES.values() for candidate in policy.candidates],
    ids=lambda value: getattr(value, "version", None) or value.name,
)
def test_every_policy_launches_every_evaluation_of_every_candidate(policy, candidate):
    """Each declared context matches the window its run sees, so every evaluation lands on a server."""
    specs = launch_specs(policy, candidate, evaluations=None, limit=None, priority_band=job_pb2.PRIORITY_BAND_BATCH)
    launched = sorted(selection.name for spec in specs for selection in spec.evals)
    assert launched == sorted(item.evaluation.name for item in resolve(policy, candidate))
