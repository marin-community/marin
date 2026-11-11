# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The environment-fidelity checks reject substitution and missing environments."""

import pytest

from experiments.post_training.task_curation.environment_fidelity import (
    environment_unused,
    missing_environment,
    provably_no_op_against,
)

AGENT = "ghcr.io/marin-community/iris-task@sha256:" + "a" * 64


def test_a_stock_agent_image_recipe_is_provably_no_op():
    ok, reason = provably_no_op_against(f"FROM {AGENT}\nWORKDIR /app\n", [], agent_image=AGENT)
    assert ok, reason


@pytest.mark.parametrize(
    ("recipe", "files"),
    [
        (f"FROM {AGENT}\nRUN pip install requests==2.32.3\n", []),
        ("FROM python:3.13-slim-bookworm\nWORKDIR /app\n", []),
        (f"FROM {AGENT}\nCOPY data/ /app/data/\n", ["setup_files/data.csv"]),
    ],
)
def test_anything_the_image_cannot_already_satisfy_is_not_a_no_op(recipe, files):
    ok, reason = provably_no_op_against(recipe, files, agent_image=AGENT)
    assert not ok and reason


def test_a_spec_that_replaces_its_archive_environment_is_flagged():
    archive = {"environment/Dockerfile": b"FROM ubuntu:24.04\nRUN pip install pandas\n"}
    replaced = {"environment_requirements": {"docker_image": AGENT}}
    assert "declares no build context" in environment_unused(replaced, archive)
    faithful = {"environment_requirements": {"docker_build": {"files": [{"path": "Dockerfile", "source": {}}]}}}
    assert environment_unused(faithful, archive) is None


def test_an_agentic_task_without_an_environment_is_rejected():
    assert missing_environment({"environment_requirements": {"capabilities": ["agent:pi"]}})
    assert missing_environment({"environment_requirements": {"docker_image": AGENT}}) is None
