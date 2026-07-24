# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The eval group and its later serving child must share the requested Iris policy."""

from experiments.evaluation.hardware import Platform
from experiments.evaluation.launch import LaunchSpec, launch_group, plan_runs


def _spec(**overrides) -> LaunchSpec:
    values = dict(
        model="qwen3-8b",
        evals=("grug-opencode-id",),
        platform=Platform.GPU,
        accelerator="H100x8",
        limit=2,
        records_prefix="s3://marin-us-east-02a/test/evals",
        cluster="marin",
        priority=2,  # PRIORITY_BAND_INTERACTIVE
        max_retries_failure=6,
    )
    values.update(overrides)
    return LaunchSpec(**values)


def test_plan_carries_explicit_priority_and_retries_to_the_serving_child():
    serve = plan_runs(_spec())[0].serve

    assert serve.priority == 2
    assert serve.max_retries_failure == 6


def test_group_submission_carries_the_same_priority_and_retries(monkeypatch):
    captured = {}

    class _Job:
        def __str__(self) -> str:
            return "/test/eval-group"

    class _Client:
        def submit(self, **kwargs):
            captured.update(kwargs)
            return _Job()

    monkeypatch.setattr("experiments.evaluation.launch._git_sha", lambda: "test")
    launch_group(_spec(), _Client())

    assert captured["max_retries_failure"] == 6
    assert captured["priority_band"] == 2
