# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from stack_outputs import workload_client


def test_workload_client_selects_the_pulumi_owned_binding():
    clients = [
        {"name": "other", "loomUrl": "https://other.example", "profile": "other"},
        {
            "name": "grafana-alerts",
            "loomUrl": "https://loom.example.com",
            "profile": "grafana_alert",
            "serviceAccount": "marin-grafana@example.iam.gserviceaccount.com",
        },
    ]

    assert workload_client(clients, "grafana-alerts") == {
        "loomUrl": "https://loom.example.com",
        "profile": "grafana_alert",
    }


@pytest.mark.parametrize("clients", [None, [], [{"name": "grafana-alerts"}]])
def test_workload_client_rejects_missing_or_incomplete_bindings(clients):
    with pytest.raises(ValueError):
        workload_client(clients, "grafana-alerts")
