# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from stack_outputs import require_workload_profile, workload_client


def test_workload_client_selects_the_pulumi_owned_binding():
    clients = [
        {"name": "other", "loomUrl": "https://other.example", "profile": "other"},
        {
            "name": "grafana-alerts",
            "loomUrl": "https://loom.example.com",
            "profile": "ops",
            "profiles": ["ops", "hero-ops"],
            "serviceAccount": "marin-grafana@example.iam.gserviceaccount.com",
        },
    ]

    assert workload_client(clients, "grafana-alerts") == {
        "loomUrl": "https://loom.example.com",
        "profile": "ops",
        "profiles": ["ops", "hero-ops"],
    }


def test_required_additional_workload_profile_must_be_federated():
    client = {"profiles": ["ops", "hero-ops"]}

    assert require_workload_profile(client, "hero-ops") == "hero-ops"
    with pytest.raises(ValueError, match="does not grant"):
        require_workload_profile(client, "missing")


@pytest.mark.parametrize("clients", [None, [], [{"name": "grafana-alerts"}]])
def test_workload_client_rejects_missing_or_incomplete_bindings(clients):
    with pytest.raises(ValueError):
        workload_client(clients, "grafana-alerts")
