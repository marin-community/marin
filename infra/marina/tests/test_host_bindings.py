# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marina.host_bindings import AppletHostAccess, applet_domains_env, applet_host_access, domain_mapping_name


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
    assert domain_mapping_name("atlas.oa.dev", AppletHostAccess.AUTHENTICATED) != domain_mapping_name(
        "atlas.oa.dev", AppletHostAccess.PUBLIC
    )
    assert domain_mapping_name("filer.marina.oa.dev", AppletHostAccess.AUTHENTICATED) == "filer-domain"
