# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for operator environment injection (defaults.inject_env)."""

from pathlib import Path

import pytest
from iris.cluster.config import IrisClusterConfig, load_config
from iris.cluster.inject_env import (
    collect_inject_env,
    merge_injected_into_task_env,
    projects_task_env_secret,
    with_injected_task_env,
)


def test_collect_inject_env_reads_named_vars(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://bucket/marin")
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    assert collect_inject_env(["MARIN_PREFIX", "WANDB_API_KEY"]) == {
        "MARIN_PREFIX": "s3://bucket/marin",
        "WANDB_API_KEY": "secret",
    }


def test_collect_inject_env_aborts_on_missing(monkeypatch):
    monkeypatch.setenv("PRESENT", "1")
    monkeypatch.delenv("ABSENT_A", raising=False)
    monkeypatch.delenv("ABSENT_B", raising=False)
    with pytest.raises(ValueError, match="ABSENT_A, ABSENT_B"):
        collect_inject_env(["PRESENT", "ABSENT_A", "ABSENT_B"])


def test_collect_inject_env_empty_is_noop():
    assert collect_inject_env([]) == {}


def test_merge_injected_is_default_literal_wins():
    config = IrisClusterConfig()
    config.defaults.task_env["MARIN_PREFIX"] = "s3://pinned/prefix"
    merge_injected_into_task_env(config, {"MARIN_PREFIX": "s3://shell/prefix", "WANDB_API_KEY": "k"})
    # An explicit literal is not overridden by the operator's shell.
    assert config.defaults.task_env["MARIN_PREFIX"] == "s3://pinned/prefix"
    # New keys are added and mirrored to worker.task_env (the worker bootstrap channel).
    assert config.defaults.task_env["WANDB_API_KEY"] == "k"
    assert config.defaults.worker.task_env["WANDB_API_KEY"] == "k"


def test_with_injected_task_env_no_inject_returns_input():
    config = IrisClusterConfig()
    config.defaults.task_env["MARIN_PREFIX"] = "s3://x"
    assert with_injected_task_env(config) is config


def test_with_injected_task_env_copies_and_folds(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "from-shell")
    config = IrisClusterConfig()
    config.defaults.inject_env.append("WANDB_API_KEY")
    merged = with_injected_task_env(config)
    assert merged is not config  # original is left untouched
    assert "WANDB_API_KEY" not in config.defaults.task_env
    assert merged.defaults.task_env["WANDB_API_KEY"] == "from-shell"


def test_projects_task_env_secret_predicate():
    s3 = IrisClusterConfig()
    s3.storage.remote_state_dir = "s3://bucket/state"
    assert projects_task_env_secret(s3)

    injected = IrisClusterConfig()
    injected.storage.remote_state_dir = "gs://bucket/state"
    injected.defaults.inject_env.append("WANDB_API_KEY")
    assert projects_task_env_secret(injected)

    neither = IrisClusterConfig()
    neither.storage.remote_state_dir = "gs://bucket/state"
    assert not projects_task_env_secret(neither)


def test_inject_env_round_trips_through_load_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """\
platform:
  manual: {}

defaults:
  inject_env:
    - MARIN_PREFIX
    - CW_KEY_ID
  worker:
    docker_image: gcr.io/project/iris-worker:latest
    port: 10001
    controller_address: "http://10.0.0.1:10000"

scale_groups:
  manual_hosts:
    num_vms: 1
    resources:
      cpu: 16
      ram: 32GB
      disk: 100GB
      device_type: cpu
      device_count: 0
      capacity_type: on-demand
    slice_template:
      manual:
        hosts: [10.0.0.1]
        ssh_user: ubuntu
"""
    )
    config = load_config(config_path)
    assert list(config.defaults.inject_env) == ["MARIN_PREFIX", "CW_KEY_ID"]
