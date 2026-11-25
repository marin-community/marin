# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A policy launch puts each candidate on one server, split only by judge key."""

from dataclasses import replace

import pytest
from iris.rpc import job_pb2

from experiments.evaluation.evals import JUDGES
from experiments.evaluation.policy.launch import EVALUATIONS_PER_SERVER, launch_specs, server_groups
from experiments.evaluation.policy.resolve import resolve
from experiments.evaluation.policy.versions import POLICIES

POLICY = POLICIES["marin-2026-11-20-trial"]
QWEN = next(candidate for candidate in POLICY.candidates if candidate.name == "qwen3.6-35b-a3b")
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


def test_a_candidate_runs_on_one_server_at_its_serving_window(checkpoint_window):
    (spec,) = launch_specs(POLICY, QWEN, evaluations=None, limit=20, priority_band=job_pb2.PRIORITY_BAND_BATCH)
    assert len(spec.evals) == len(POLICY.in_distribution)
    assert spec.model.serve.max_model_len == SERVED
    assert spec.max_concurrent == EVALUATIONS_PER_SERVER


def test_judges_with_different_keys_get_separate_servers(monkeypatch):
    # A launch hands its judges one key; evaluations without a judge join the first server.
    other_key = JUDGES["together"].model_copy(update={"judge_api_key": ("env:OTHER",)})
    monkeypatch.setitem(JUDGES, "other", other_key)
    olympiad = RESOLVED["olympiadbench"]
    olympiad = replace(olympiad, evaluation=replace(olympiad.evaluation, judge="other"))
    groups = {
        tuple(item.evaluation.name for item in group)
        for group in server_groups([RESOLVED["gsm8k-0shot"], RESOLVED["financebench"], olympiad])
    }
    assert groups == {("gsm8k-0shot", "financebench"), ("olympiadbench",)}


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


def test_a_checkpoint_that_serves_less_than_the_catalog_window_is_rejected(checkpoint_window):
    checkpoint_window(16384)
    with pytest.raises(ValueError, match="swebench-verified"):
        launch_specs(
            POLICY, QWEN, evaluations={"swebench-verified"}, limit=None, priority_band=job_pb2.PRIORITY_BAND_BATCH
        )
