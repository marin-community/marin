# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from rigging.grants import Permission

from infra.auth.cedar import CedarPolicy, ProviderIdentity

POLICY_DIRECTORY = Path(__file__).resolve().parents[1] / "policy"


@pytest.fixture
def policy():
    return CedarPolicy(
        json.loads((POLICY_DIRECTORY / "schema.json").read_text()),
        (POLICY_DIRECTORY / "example.cedar").read_text(),
        json.loads((POLICY_DIRECTORY / "example-entities.json").read_text()),
        [ProviderIdentity("diagnostics", (Permission("cw_s3", frozenset({"s3:GetObject"}), "s3://logs/incidents/"),))],
    )


def test_shipped_policy_separates_research_submission_from_alert_investigation(policy):
    assert policy.check("example-researcher", "iris", "job.submit", "/analysis/experiment/")
    assert not policy.check("example-researcher", "iris", "job.submit", "/other/experiment/")
    assert not policy.check("service:grafana", "iris", "job.submit", "/analysis/experiment/")
    assert policy.check("service:grafana", "loom", "session.create", "/diagnostics/")
    assert not policy.check("service:grafana", "loom", "session.create", "/unrestricted/")
    assert policy.allows_provider("service:grafana", "diagnostics")
    assert not policy.check("service:grafana", "cw_s3", "s3:PutObject", "s3://logs/incidents/probe")


def test_undeclared_action_and_resource_type_cannot_gain_authority(policy):
    assert not policy.check("example-researcher", "iris", "job.sumbit", "/analysis/")
    assert not policy.check("example-researcher", "iris", "job.submit", "s3://logs/incidents/")
    assert not policy.check("example-researcher", "marina", "app.release", "/echo/")
