# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""federation_dns_target — the computed CoreWeave CNAME target for a cluster's federation host.

Guards the assumption FederationDns is built on: CoreWeave's allocated hostname always follows
`<host-label>.<tenant>-<cks-cluster-name>.coreweave.app`. If CoreWeave ever changes that naming
scheme, this test — not a live cluster — is where it should first show up as a failure.
"""

import pytest
from iac.coreweave.dns import federation_dns_target


@pytest.mark.parametrize(
    "cluster, cks_cluster_name, expected",
    [
        ("cw-rno2a", "marin-rn02a", "iris-cw-rno2a.208261-marin-rn02a.coreweave.app"),
        ("cw-us-east-02a", "marin-gpu", "iris-cw-us-east-02a.208261-marin-gpu.coreweave.app"),
        (
            "cw-us-east-08a",
            "marin-us-east-08a",
            "iris-cw-us-east-08a.208261-marin-us-east-08a.coreweave.app",
        ),
        ("cw-us-west-04a", "marin", "iris-cw-us-west-04a.208261-marin.coreweave.app"),
    ],
)
def test_federation_dns_target_matches_the_live_oa_dev_zone(cluster, cks_cluster_name, expected):
    assert federation_dns_target(cluster, cks_cluster_name) == expected
