# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for Marin's Harbor task execution boundary."""

import json
import shutil

import pytest

from tests.evaluation.harbor_runtime import external_python as _external_python
from tests.evaluation.harbor_runtime import preflight as _preflight


@pytest.fixture
def iris_policy(tmp_path):
    task = tmp_path / "dataset" / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (task / "instruction.md").write_text("Write an answer in /workspace.\n")
    (task / "task.toml").write_text(
        '[task]\nname = "test/offline"\n[environment]\n'
        f'docker_image = "example.test/task@sha256:{"a" * 64}"\nallow_internet = false\nworkdir = "/workspace"\n'
    )
    policy = tmp_path / "policy-source.json"
    policy.write_text(
        json.dumps(
            {
                "environment": {"import_path": "shellbox.backends.iris.environment:IrisEnvironment"},
                "agents": [{"name": "terminus-2", "kwargs": {"llm_backend": "openai"}}],
                "datasets": [{"path": "dataset"}],
            }
        )
    )
    return policy


def test_iris_preflight_resolves_offline_tasks_without_connecting_to_a_controller(tmp_path, iris_policy):
    completed = _preflight(tmp_path, [(iris_policy, {})])
    result = json.loads(completed.stdout)[0]

    assert result["environment"] == "shellbox.backends.iris.environment:IrisEnvironment"
    assert result["benchmark_metadata"]["n_benchmark"] == 1
    assert result["verifier_env_keys"] == []


@pytest.mark.parametrize(
    "environment",
    [
        {"type": "docker"},
        {"import_path": "harbor.environments.docker.docker:DockerEnvironment"},
    ],
)
def test_preflight_rejects_docker_execution_before_task_launch(tmp_path, iris_policy, environment):
    document = json.loads(iris_policy.read_text())
    document["environment"] = environment
    iris_policy.write_text(json.dumps(document))

    completed = _preflight(tmp_path, [(iris_policy, {})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.mark.parametrize(
    ("section", "replacement"),
    [
        ("agents", [{"name": "mini-swe-agent"}]),
        ("agents", [{"name": "terminus-2", "env": {"STOLEN": "${AWS_SECRET_ACCESS_KEY}"}}]),
        ("agents", [{"name": "terminus-2", "kwargs": {"extra_env": {"STOLEN": "${HF_TOKEN}"}}}]),
        ("agents", [{"name": "terminus-2", "mcp_servers": [{"name": "host", "url": "http://host"}]}]),
        ("verifier", {"env": {"OPENAI_API_KEY": "${AWS_SECRET_ACCESS_KEY}"}}),
        ("verifier", {"import_path": "host.verifier:Verifier"}),
    ],
)
def test_iris_preflight_rejects_agent_and_verifier_escape_routes(tmp_path, iris_policy, section, replacement):
    document = json.loads(iris_policy.read_text())
    document[section] = replacement
    iris_policy.write_text(json.dumps(document))

    completed = _preflight(tmp_path, [(iris_policy, {})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.mark.parametrize(
    "task_configuration",
    [
        '[environment]\ndocker_image = "example.test/task:latest"\nallow_internet = false\nworkdir = "/workspace"\n',
        f'[environment]\ndocker_image = "example.test/task@sha256:{"a" * 64}"\n'
        'allow_internet = true\nworkdir = "/workspace"\n',
        f'[environment]\ndocker_image = "example.test/task@sha256:{"a" * 64}"\n'
        'allow_internet = false\nworkdir = "/workspace"\n'
        '[environment.env]\nSTOLEN = "${AWS_SECRET_ACCESS_KEY}"\n',
        f'[environment]\ndocker_image = "example.test/task@sha256:{"a" * 64}"\n'
        'allow_internet = false\nworkdir = "/workspace"\n'
        '[verifier.env]\nSTOLEN = "${HF_TOKEN}"\n',
    ],
)
def test_iris_preflight_checks_each_resolved_task_before_submission(tmp_path, iris_policy, task_configuration):
    invalid_task = tmp_path / "dataset" / "z-invalid-task"
    shutil.copytree(tmp_path / "dataset" / "task", invalid_task)
    (invalid_task / "task.toml").write_text(task_configuration)

    completed = _preflight(tmp_path, [(iris_policy, {})], check=False)

    assert completed.returncode == 2
    assert completed.stdout == ""


def test_iris_runtime_rejects_hosted_judge_overlay(tmp_path, iris_policy):
    preflight = json.loads(_preflight(tmp_path, [(iris_policy, {})]).stdout)[0]
    policy = tmp_path / "stable-policy.json"
    policy.write_text(preflight["stable_policy_json"])
    overlay = tmp_path / "overlay.json"
    overlay.write_text(
        json.dumps(
            {
                "job_name": "test",
                "jobs_dir": str(tmp_path / "jobs"),
                "dataset_path": str(tmp_path / "dataset"),
                "endpoint_url": "https://model.test/v1",
                "served_model": "test",
                "task_limit": None,
                "model_agent_kwargs": {},
                "verifier_env": {"OPENAI_API_KEY": "hosted-judge-secret"},
                "archive_root": str(tmp_path / "archive"),
                "archive_dataset": "test",
            }
        )
    )
    completed = _external_python(
        "-c",
        "from pathlib import Path; "
        "from marin.evaluation.harbor.trial_driver import effective_job_config; "
        f"effective_job_config(Path({str(policy)!r}), Path({str(overlay)!r}))",
        check=False,
    )

    assert completed.returncode != 0
