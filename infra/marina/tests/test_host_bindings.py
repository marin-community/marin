# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marina.host_bindings import dns_zone_name, host_resource_slug

ZONES = ("oa.dev", "marin.community", "marina.oa.dev")


@pytest.mark.parametrize(
    ("hostname", "zone"),
    [
        ("atlas.oa.dev", "oa.dev"),
        ("atlas.marin.community", "marin.community"),
        ("filer.marina.oa.dev", "marina.oa.dev"),
        ("marin.community", "marin.community"),
    ],
)
def test_dns_zone_name_selects_most_specific_containing_zone(hostname: str, zone: str) -> None:
    assert dns_zone_name(hostname, ZONES) == zone


@pytest.mark.parametrize("hostname", ["atlas.openathena.ai", "atlas.notoa.dev"])
def test_dns_zone_name_rejects_host_outside_every_zone(hostname: str) -> None:
    with pytest.raises(ValueError, match=hostname):
        dns_zone_name(hostname, ZONES)


def test_host_resource_slugs_keep_primary_names_and_separate_other_zones() -> None:
    hosts = {
        "public.applets.marina.oa.dev": "oa.dev",
        "atlas.oa.dev": "oa.dev",
        "atlas.marin.community": "marin.community",
    }

    slugs = [host_resource_slug(host, zone, "oa.dev") for host, zone in hosts.items()]

    assert slugs == ["public", "atlas", "atlas-marin-community"]
