# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invariants that hold across the whole datakit source catalog."""

from marin.datakit.sources import all_sources

# The hash attributes ``normalize_step`` always declares. A terminal step without
# them was not built by normalize, so its output carries none of normalize's
# guarantees: unsorted shards and duplicate ids reach every consumer (#8110).
NORMALIZE_IDENTITY_ATTRS = frozenset(
    {"text_field", "id_field", "target_partition_bytes", "max_whitespace_run_chars", "dedup_mode"}
)


def test_every_source_normalizes_its_output():
    unnormalized = [
        name
        for name, source in all_sources().items()
        if not NORMALIZE_IDENTITY_ATTRS <= source.normalized.hash_attrs.keys() or not source.normalized.deps
    ]
    assert unnormalized == []
