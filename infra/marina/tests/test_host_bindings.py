# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marina.host_bindings import (
    AppletHostAccess,
    applet_domains_env,
    applet_host_access,
    dns_zone_name,
    domain_mapping_name,
    host_resource_slug,
)


def test_public_hosts_reach_only_the_public_reader_and_stay_bindable_through_the_service() -> None:
    hosts = applet_host_access(["filer.marina.oa.dev"], ["atlas.oa.dev"])

    assert applet_domains_env(hosts) == {
        AppletHostAccess.AUTHENTICATED: "filer.marina.oa.dev,atlas.oa.dev",
        AppletHostAccess.PUBLIC: "atlas.oa.dev",
    }


def test_a_host_cannot_be_both_authenticated_and_public() -> None:
    with pytest.raises(ValueError, match=r"atlas\.oa\.dev"):
        applet_host_access(["atlas.oa.dev"], ["atlas.oa.dev"])


def test_moving_a_host_between_surfaces_changes_its_mapping_resource() -> None:
    # Pulumi ignores route changes on an existing mapping, so a one-deploy move must surface as a
    # new resource (which Cloud Run rejects as a duplicate) rather than a silent no-op.
    assert domain_mapping_name("atlas", AppletHostAccess.AUTHENTICATED) != domain_mapping_name(
        "atlas", AppletHostAccess.PUBLIC
    )
    assert domain_mapping_name("filer", AppletHostAccess.AUTHENTICATED) == "filer-domain"


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
