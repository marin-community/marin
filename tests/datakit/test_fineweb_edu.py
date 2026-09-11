# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FineWeb-Edu canonical pipeline."""

from marin.datakit.canonical.fineweb_edu import download
from marin.datakit.normalize import normalize_step


def test_normalize_subset_distinct_cache_keys():
    """Different subsets share a step name but produce distinct output paths.

    Catches regressions where ``subset`` stops propagating into ``input_path``
    (and thus into ``hash_attrs``), which would collapse caches across subsets.
    """
    dl = download()
    data_step = normalize_step(
        name="normalized/fineweb_edu",
        download=dl,
        relative_input_path="data",
    )
    sample_step = normalize_step(
        name="normalized/fineweb_edu",
        download=dl,
        relative_input_path="sample/10BT",
    )
    assert data_step.name == sample_step.name
    assert data_step.output_path != sample_step.output_path
