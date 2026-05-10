# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the Datakit Testbed ferry DAG."""

import pytest
from marin.datakit.sources import DatakitSource, all_sources

from experiments.datakit_testbed.sampler import build_testbed_steps

_ALL = all_sources()
_ALL_LIST = list(_ALL.values())


def _source(name: str) -> DatakitSource:
    return _ALL[name]


def test_dag_empty_source_list_raises():
    with pytest.raises(ValueError, match="at least one source"):
        build_testbed_steps(sources=[])


def test_dag_single_source_emits_full_chain_in_order():
    """Per-source step ordering: download → normalize → sample."""
    src = _source("nemotron_cc_code_v1/all")
    steps = build_testbed_steps(sources=[src])

    names = [s.name for s in steps]
    assert names == [
        "raw/nemotron_cc_code_v1",
        "normalized/nemotron_cc_code_v1/all",
        "data/datakit/normalized/nemotron_cc_code_v1/all",
    ]


def test_dag_stops_at_sample():
    """No testbed-specific stages beyond sample (guards against re-adding noop_dedup/consolidate)."""
    steps = build_testbed_steps(sources=_ALL_LIST)
    testbed_steps = [s for s in steps if s.name.startswith("data/datakit/normalized/")]
    assert len(testbed_steps) == len(_ALL_LIST), "expected exactly one testbed step per source (the sampler)"


def test_dag_nemotron_family_subsets_share_one_download_stepspec():
    """Every Nemotron-CC v2.1 subset's chain starts with the SAME StepSpec object.

    Subsets share one family download because
    :func:`nemotron_v2_normalize_steps` builds the download once and passes it
    to each subset's normalize step. Duplicate downloads in the returned list
    are the same Python object, which the executor trivially dedupes.
    """
    v21_sources = tuple(s for s in _ALL.values() if s.name.startswith("nemotron_cc_v2_1/"))
    assert len(v21_sources) > 1, "registry must have multiple v2.1 subsets"

    first_download = v21_sources[0].normalize_steps[0]
    for src in v21_sources[1:]:
        assert src.normalize_steps[0] is first_download, "v2.1 subsets must share one download StepSpec"


def test_dag_starcoder2_subsets_get_distinct_download_names():
    """StarCoder2-Extras stages each subset under its own path; the download
    names must reflect that disambiguation so per-subset StepSpecs don't collide."""
    subsets = tuple(s for s in _ALL.values() if s.name.startswith("starcoder2/"))
    assert len(subsets) > 1

    download_names = {s.normalize_steps[0].name for s in subsets}
    assert len(download_names) == len(subsets), f"expected distinct downloads, got {download_names}"
