# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import click
import pytest

from lib.iris.scripts.iap_gclb import (
    LEGACY_TOKEN_PROXY_PATHS,
    TOKEN_PROXY_ROUTE_DESCRIPTION,
    _reconcile_token_proxy_rule,
    _reconcile_url_map_test,
    _remove_url_map_tests,
)


def test_token_proxy_rule_migrates_local_path_rule_and_adds_federated_match() -> None:
    matcher = {
        "name": "marin-dev",
        "defaultService": "iap-backend",
        "pathRules": [
            {
                "paths": sorted(LEGACY_TOKEN_PROXY_PATHS),
                "service": "proxy-backend",
            }
        ],
    }

    assert _reconcile_token_proxy_rule(matcher, "proxy-backend")
    assert "pathRules" not in matcher
    assert matcher["routeRules"] == [
        {
            "description": TOKEN_PROXY_ROUTE_DESCRIPTION,
            "priority": 0,
            "matchRules": [
                {"pathTemplateMatch": "/proxy/t/**"},
                {"pathTemplateMatch": "/proxy/*/t/**"},
            ],
            "service": "proxy-backend",
        }
    ]
    assert not _reconcile_token_proxy_rule(matcher, "proxy-backend")


def test_token_proxy_rule_does_not_replace_unrelated_proxy_routes() -> None:
    matcher = {
        "name": "marin-dev",
        "pathRules": [{"paths": ["/proxy/private/*"], "service": "iap-backend"}],
    }

    with pytest.raises(click.ClickException):
        _reconcile_token_proxy_rule(matcher, "proxy-backend")

    assert matcher["pathRules"] == [{"paths": ["/proxy/private/*"], "service": "iap-backend"}]


def test_url_map_tests_preserve_capability_and_private_routing_boundaries() -> None:
    url_map = {
        "tests": [
            {
                "host": "iris-dev.oa.dev",
                "path": "/existing",
                "service": "existing-backend",
            }
        ]
    }

    assert _reconcile_url_map_test(
        url_map,
        host="iris-dev.oa.dev",
        path="/proxy/test-peer/t/test-token/test-endpoint",
        service="proxy-backend",
    )
    assert _reconcile_url_map_test(
        url_map,
        host="iris-dev.oa.dev",
        path="/proxy/system.log-server/",
        service="iap-backend",
    )
    assert not _reconcile_url_map_test(
        url_map,
        host="iris-dev.oa.dev",
        path="/proxy/system.log-server/",
        service="iap-backend",
    )
    assert url_map["tests"] == [
        {
            "host": "iris-dev.oa.dev",
            "path": "/existing",
            "service": "existing-backend",
        },
        {
            "host": "iris-dev.oa.dev",
            "path": "/proxy/test-peer/t/test-token/test-endpoint",
            "service": "proxy-backend",
        },
        {
            "host": "iris-dev.oa.dev",
            "path": "/proxy/system.log-server/",
            "service": "iap-backend",
        },
    ]

    assert _remove_url_map_tests(
        url_map,
        host="iris-dev.oa.dev",
        paths=(
            "/proxy/test-peer/t/test-token/test-endpoint",
            "/proxy/system.log-server/",
        ),
    )
    assert url_map["tests"] == [
        {
            "host": "iris-dev.oa.dev",
            "path": "/existing",
            "service": "existing-backend",
        }
    ]
