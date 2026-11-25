# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Policy evaluations group by requested server window."""

from dataclasses import replace

import pytest
from iris.rpc import job_pb2

from experiments.evaluation.evals import JUDGES
from experiments.evaluation.policy.launch import launch_specs, server_groups
from experiments.evaluation.policy.resolve import resolve
from experiments.evaluation.policy.versions.marin_2026_11_20_trial.policy import POLICY, QWEN

RESOLVED = {item.evaluation.name: item for item in resolve(POLICY, QWEN)}
SERVED = QWEN.model.serve.max_model_len


@pytest.fixture
def checkpoint_window(monkeypatch):
    """Set the window the checkpoint clamps the candidate's served window to."""

    def clamp(window: int) -> None:
        monkeypatch.setattr(
            "experiments.evaluation.policy.launch.resolved_serve_config",
            lambda model: replace(model.serve, max_model_len=min(window, model.serve.max_model_len)),
        )

    clamp(SERVED)
    return clamp


def _windows(names: list[str], **contexts: int) -> set[tuple[int, bool, tuple[str, ...]]]:
    items = [replace(RESOLVED[name], context=contexts.get(name, RESOLVED[name].context)) for name in names]
    return {
        (group.window, group.agentic, tuple(item.evaluation.name for item in group.evaluations))
        for group in server_groups(items, SERVED)
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


def test_an_agent_run_below_its_input_limit_gets_a_server_of_that_window():
    # The agent's input is the smaller of its limit and the server's window, so the server sets it.
    assert _windows(["swebench-verified"], **{"swebench-verified": 16_384}) == {(16_384, True, ("swebench-verified",))}


def test_judges_with_different_keys_get_separate_servers(monkeypatch):
    # A launch hands its judges one key; evaluations without a judge join the first server.
    other_key = JUDGES["together"].model_copy(update={"judge_api_key": ("env:OTHER",)})
    monkeypatch.setitem(JUDGES, "other", other_key)
    olympiad = RESOLVED["olympiadbench"]
    olympiad = replace(olympiad, evaluation=replace(olympiad.evaluation, judge="other"))
    groups = {
        tuple(item.evaluation.name for item in group.evaluations)
        for group in server_groups([RESOLVED["gsm8k-0shot"], RESOLVED["financebench"], olympiad], SERVED)
    }
    assert groups == {("gsm8k-0shot", "financebench"), ("olympiadbench",)}
    assert (
        len(server_groups([RESOLVED["gsm8k-0shot"], RESOLVED["financebench"], RESOLVED["olympiadbench"]], SERVED)) == 1
    )


def test_a_launch_covers_only_the_named_evaluations(checkpoint_window):
    specs = launch_specs(
        POLICY,
        QWEN,
        evaluations={"aime24", "swebench-verified"},
        limit=20,
        priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
    )
    assert sorted(selection.name for spec in specs for selection in spec.evals) == ["aime24", "swebench-verified"]


def test_unknown_evaluations_are_rejected(checkpoint_window):
    with pytest.raises(ValueError, match="bixbench"):
        launch_specs(POLICY, QWEN, evaluations={"bixbench"}, limit=None, priority_band=job_pb2.PRIORITY_BAND_BATCH)


def test_a_checkpoint_that_clamps_the_served_window_below_an_agent_window_is_rejected(checkpoint_window):
    checkpoint_window(16384)
    with pytest.raises(ValueError, match="swebench-verified"):
        launch_specs(
            POLICY, QWEN, evaluations={"swebench-verified"}, limit=None, priority_band=job_pb2.PRIORITY_BAND_BATCH
        )
