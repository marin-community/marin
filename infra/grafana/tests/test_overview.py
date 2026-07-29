# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime

from overview import provisioning_query, provisioning_rows


def _sample(metric: str, value: float, **labels: str) -> dict[str, object]:
    return {
        "metric": metric,
        "value": value,
        "collected_at": 1_784_257_200_000,
        **{f"label_{key}": label for key, label in labels.items()},
    }


def test_provisioning_rows_restore_fleet_and_pool_status():
    rows = [
        _sample("provision_ready", 9, scope="fleet"),
        _sample("provision_stockout", 2, scope="fleet"),
        _sample("provision_error", 1, scope="fleet"),
        _sample("provision_preempted", 3, scope="fleet"),
        _sample("provision_outcomes", 12, scope="fleet"),
        _sample("provision_success_ratio", 0.75, scope="fleet"),
        _sample("provision_pools_placing", 4, scope="fleet"),
        _sample("provision_pools_stockout_dead", 2, scope="fleet"),
        _sample("provision_window_hours", 3, scope="fleet"),
        _sample("provision_latency_seconds", 45, scope="fleet", quantile="p50"),
        _sample("provision_latency_seconds", 120, scope="fleet", quantile="p95"),
        _sample("provision_ready", 3, resource_type="v5p-8", scale_group="a", zone="us-east5-a"),
        _sample("provision_stockout", 1, resource_type="v5p-8", scale_group="a", zone="us-east5-a"),
        _sample("provision_outcomes", 4, resource_type="v5p-8", scale_group="a", zone="us-east5-a"),
        _sample(
            "provision_latency_seconds",
            32,
            resource_type="v5p-8",
            scale_group="a",
            zone="us-east5-a",
            quantile="p50",
        ),
    ]

    fleet, pool = provisioning_rows(rows)

    assert fleet.scope == "fleet"
    assert fleet.success_ratio == 0.75
    assert (fleet.ready, fleet.stockout, fleet.error, fleet.preempted) == (9, 2, 1, 3)
    assert (fleet.pools_placing, fleet.pools_no_ready_outcome) == (4, 2)
    assert (fleet.latency_p50_seconds, fleet.latency_p95_seconds, fleet.window_hours) == (45, 120, 3)

    assert (pool.resource_type, pool.scale_group, pool.zone) == ("v5p-8", "a", "us-east5-a")
    assert pool.success_ratio == 0.75
    assert pool.latency_p50_seconds == 32


def test_provisioning_rows_return_no_rows_without_a_recent_cycle():
    assert provisioning_rows([]) == []


def test_provisioning_query_uses_one_shared_latest_cycle():
    sql = provisioning_query(datetime(2026, 7, 29, 3, 0, tzinfo=UTC))

    assert sql.count("2026-07-29 03:00:00") == 2
    assert "collected_at = (" in sql
    assert "SELECT MAX(collected_at)" in sql
