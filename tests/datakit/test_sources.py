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
