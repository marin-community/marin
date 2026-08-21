# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for service rollout selection."""

from typing import cast

import pytest

from scripts.ci.pulumi_rollouts import rollout_for_service, rollouts_for_paths, workflow_payload


@pytest.mark.parametrize(
    "changed_path",
    [
        "infra/evaldash/src/samples.py",
        "lib/finestore/src/finestore/reader.py",
        "lib/marin/src/marin/evaluation/archive.py",
        "lib/iris/src/iris/rpc/controller_connect.py",
        "lib/finelog/src/finelog/rpc/logs_pb2.py",
        "lib/rigging/src/rigging/filesystem/factory.py",
    ],
)
def test_evaldash_rollout_tracks_image_inputs(changed_path: str) -> None:
    assert "evaldash" in {rollout.name for rollout in rollouts_for_paths([changed_path])}


def test_evaldash_rollout_ignores_marin_modules_outside_the_image() -> None:
    assert "evaldash" not in {
        rollout.name for rollout in rollouts_for_paths(["lib/marin/src/marin/training/training.py"])
    }


def test_rollout_payload_passes_deploy_cli_config() -> None:
    payload = workflow_payload([rollout_for_service("ducky")], deploy_generation="17")
    deploy = cast(dict[str, list[dict[str, object]]], payload["deploy"])
    item = deploy["include"][0]

    assert item["name"] == "ducky"
    assert item["config"] == "ducky:deploy_generation=17"
    assert "config_map" not in item
    assert "stack" not in item
