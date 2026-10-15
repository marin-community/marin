# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for service rollout selection."""

from typing import cast

import pytest

from scripts.ci.pulumi_rollouts import (
    rollout_for_service,
    rollouts_for_paths,
    workflow_payload,
    workflow_payload_for_paths,
)


@pytest.mark.parametrize(
    "changed_path",
    [
        "infra/marina/apps/evaldash/samples.py",
        "lib/finestore/src/finestore/reader.py",
        "lib/marin/src/marin/evaluation/archive.py",
        "lib/iris/src/iris/rpc/controller_connect.py",
        "lib/finelog/src/finelog/rpc/logs_pb2.py",
        "lib/rigging/src/rigging/filesystem/factory.py",
    ],
)
def test_marina_rollout_tracks_image_inputs(changed_path: str) -> None:
    assert "marina" in {rollout.name for rollout in rollouts_for_paths([changed_path])}


def test_marina_rollout_ignores_marin_modules_outside_the_image() -> None:
    assert "marina" not in {rollout.name for rollout in rollouts_for_paths(["lib/marin/src/marin/training/training.py"])}


@pytest.mark.parametrize(
    "changed_path",
    [
        "infra/marina/applets/rl_data_catalog/server/app.py",
        "infra/marina/applets/filer/dist/app.js",
        "infra/marina/examples/problem-set-applet/applet.toml",
    ],
)
def test_applet_changes_do_not_roll_out_the_shared_runtime(changed_path: str) -> None:
    payload = workflow_payload_for_paths([changed_path])
    assert payload["deploy"] == {"include": []}
    tests = cast(dict[str, list[dict[str, object]]], payload["test"])
    assert [item["name"] for item in tests["include"]] == ["marina"]
    mixed = workflow_payload_for_paths([changed_path, "infra/marina/src/marina/server.py"])
    deploy = cast(dict[str, list[dict[str, object]]], mixed["deploy"])
    assert [item["name"] for item in deploy["include"]] == ["marina"]


def test_rollout_payload_passes_deploy_cli_config() -> None:
    payload = workflow_payload([rollout_for_service("ducky")], deploy_generation="17")
    deploy = cast(dict[str, list[dict[str, object]]], payload["deploy"])
    item = deploy["include"][0]

    assert item["name"] == "ducky"
    assert item["config"] == "ducky:deploy_generation=17"


def test_cloud_run_change_triggers_all_service_rollouts() -> None:
    selected = {rollout.name for rollout in rollouts_for_paths(["infra/pulumi/src/iac/gcp/cloud_run.py"])}

    assert {"marina", "grafana"} <= selected
