# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invariants that hold across the whole datakit source catalog."""

from marin.datakit.normalize import NORMALIZE_IDENTITY_ATTRS
from marin.datakit.sources import all_sources


def test_every_source_normalizes_its_output():
    """A source whose terminal step skips normalize ships unsorted, duplicated ids (#8110)."""
    unnormalized = [
        name
        for name, source in all_sources().items()
        if not NORMALIZE_IDENTITY_ATTRS <= source.normalized.hash_attrs.keys() or not source.normalized.deps
    ]
    assert unnormalized == []


def test_code_alchemy_registry_components():
    """Code Alchemy subsets remain independent normalized mixture components."""
    expected_counts = {
        "code-alchemy/code-dev": 251.195609089,
        "code-alchemy/code-dialogue": 509.618631817,
        "code-alchemy/code-enhance": 112.823840552,
        "code-alchemy/code-qa": 34.821683237,
        "code-alchemy/code-trace": 8.078120885,
    }

    sources = all_sources()
    code_alchemy_names = {name for name in sources if name.startswith("code-alchemy/")}

    assert code_alchemy_names == expected_counts.keys()
    assert len([source.name for source in sources.values()]) == len({source.name for source in sources.values()})
    for name, expected_count in expected_counts.items():
        source = sources[name]
        assert source.rough_token_count_b == expected_count
        assert source.normalized.name == f"normalized/{name}"
        assert NORMALIZE_IDENTITY_ATTRS <= source.normalized.hash_attrs.keys()
