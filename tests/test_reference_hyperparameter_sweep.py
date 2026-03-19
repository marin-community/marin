# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for reference hyperparameter sweep resource allocation."""

import os

import pytest


@pytest.fixture(autouse=True)
def _ci_env(monkeypatch):
    monkeypatch.setenv("CI", "1")


def test_train_step_uses_cpu_for_outer_dispatch():
    """The outer remote() wrapper for training must use CPU resources.

    The inner Fray dispatch (via run_grug_base_trial) allocates the TPU.
    Using with_tpu() in both places doubles TPU allocation per trial.
    Regression test for https://github.com/marin-community/marin/issues/3853.
    """
    from experiments.references.reference_hyperparameter_sweep import _build_base_launch_config, _build_train_step
    from marin.execution.remote import RemoteCallable

    base_config = _build_base_launch_config()
    step = _build_train_step(
        loop_index=0,
        suggestion_index=0,
        suggestions_path="gs://fake/suggestions.json",
        base_launch_config=base_config,
    )

    fn = step.fn
    assert isinstance(fn, RemoteCallable)
    assert fn.resources.device.kind == "cpu", (
        f"Outer dispatch should use CPU, not {fn.resources.device.kind}. "
        "Inner Fray dispatch handles TPU allocation."
    )
