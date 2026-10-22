# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The egress rule syntax users write in ``--egress-allow`` and cluster config."""

import pytest
from iris.cluster.egress import BASE_EGRESS_ALLOWLIST, format_egress_rule, parse_egress_rule, parse_egress_rules


@pytest.mark.parametrize(
    ("text", "canonical"),
    [
        pytest.param("API.OpenAI.com", "api.openai.com", id="domain-lowercased"),
        pytest.param("example.com.", "example.com", id="trailing-dot"),
        pytest.param("*.example.com:443,8443", "*.example.com:443,8443", id="pattern-with-ports"),
        pytest.param("*.*.cdn.example.com", "*.*.cdn.example.com", id="two-wildcard-labels"),
        pytest.param("203.0.113.7", "203.0.113.7/32", id="bare-ipv4"),
        pytest.param("203.0.113.0/24:5432", "203.0.113.0/24:5432", id="network-with-port"),
        pytest.param("2001:db8::/32", "2001:db8::/32", id="ipv6-without-ports"),
        pytest.param("[2001:db8::1]:443", "[2001:db8::1/128]:443", id="bracketed-ipv6-address"),
    ],
)
def test_parse_egress_rule_round_trips_through_canonical_form(text, canonical):
    rule = parse_egress_rule(text)

    assert format_egress_rule(rule) == canonical
    assert parse_egress_rule(canonical) == rule


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("*.com", id="pattern-over-a-tld"),
        pytest.param("api.*.example.com", id="wildcard-inside"),
        pytest.param("ex ample.com", id="invalid-label"),
        pytest.param("0.0.0.0/0", id="every-address"),
        pytest.param("203.0.113.1/24", id="host-bits-set"),
        pytest.param("example.com:0", id="port-zero"),
        pytest.param("example.com:443,443", id="repeated-port"),
        pytest.param("example.com:https", id="named-port"),
        pytest.param("[2001:db8::1", id="unclosed-bracket"),
    ],
)
def test_parse_egress_rule_rejects_rules_that_policies_cannot_express(text):
    with pytest.raises(ValueError):
        parse_egress_rule(text)


def test_base_allowlist_parses():
    """Every node builds its policy from the base list, so one bad entry would break every ALLOWLIST task."""
    assert len(parse_egress_rules(BASE_EGRESS_ALLOWLIST)) == len(BASE_EGRESS_ALLOWLIST)
