from iac.config import KueueProvisioningSpec


def test_kueue_provisioning_defaults_include_fleet_resync_headroom() -> None:
    spec = KueueProvisioningSpec()

    assert (
        spec.manager_memory_limit,
        spec.client_connection.qps,
        spec.client_connection.burst,
    ) == ("8Gi", 1000.0, 2000)
