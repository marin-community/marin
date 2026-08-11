# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.ci.pulumi_rollouts import rollout_for_service, rollout_item, rollouts_for_paths


def test_rollouts_for_paths_selects_direct_and_shared_dependencies() -> None:
    selected = rollouts_for_paths(
        (
            "infra/echo/api/app.py",
            "infra/pulumi/src/iac/gcp/cloud_run.py",
            "lib/ducky/src/ducky/server.py",
        )
    )

    assert [rollout.name for rollout in selected] == ["ducky", "echo", "grafana"]


def test_rollouts_for_paths_ignores_unregistered_projects() -> None:
    assert rollouts_for_paths(("infra/evaldash/__main__.py", "docs/index.md")) == ()


def test_rollout_item_encodes_supported_deploy_generation() -> None:
    assert rollout_item(rollout_for_service("xprof"), "42")["config_map"] == (
        '{"xprof:deploy_generation":{"value":"42"}}'
    )


def test_rollout_item_rejects_unsupported_deploy_generation() -> None:
    with pytest.raises(ValueError):
        rollout_item(rollout_for_service("echo"), "42")
