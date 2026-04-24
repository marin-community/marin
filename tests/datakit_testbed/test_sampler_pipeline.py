# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the Datakit Testbed ferry DAG."""

import pytest

from experiments.datakit_testbed.sampler import build_testbed_steps
from marin.datakit.sources import DatakitSource, all_sources

_ALL = all_sources()
_ALL_LIST = list(_ALL.values())


def _source(name: str) -> DatakitSource:
    return _ALL[name]


def test_dag_empty_source_list_raises():
    with pytest.raises(ValueError, match="at least one source"):
        build_testbed_steps("run0", sources=[])


def test_dag_single_source_shape():
    src = _source("nemotron_cc_code_v1/all")
    steps = build_testbed_steps("run0", sources=[src])

    names = [s.name for s in steps]
    assert names == [
        "raw/nemotron_cc_code_v1",
        "normalized/nemotron_cc_code_v1/all",
        "datakit-testbed/nemotron_cc_code_v1/all",
    ]


def test_dag_has_one_sample_step_per_source():
    steps = build_testbed_steps("run0", sources=_ALL_LIST)
    sample_names = {s.name for s in steps if s.name.startswith("datakit-testbed/")}
    assert sample_names == {f"datakit-testbed/{s.name}" for s in _ALL_LIST}


def test_dag_nemotron_family_subsets_share_one_download_stepspec():
    """Every Nemotron-CC v2.1 subset's chain starts with the SAME StepSpec object.

    Subsets share one family download because
    :func:`nemotron_v2_normalize_steps` builds the download once and passes it
    to each subset's normalize step. The ferry appends each source's
    ``normalize_steps`` as-is; duplicate downloads in the returned list are the
    same Python object, which the executor trivially dedupes.
    """
    v21_sources = tuple(s for s in _ALL.values() if s.name.startswith("nemotron_cc_v2_1/"))
    assert len(v21_sources) > 1, "registry must have multiple v2.1 subsets"

    first_download = v21_sources[0].normalize_steps[0]
    for src in v21_sources[1:]:
        assert src.normalize_steps[0] is first_download, "v2.1 subsets must share one download StepSpec"

    steps = build_testbed_steps("run0", sources=v21_sources)
    normalize_steps = [s for s in steps if s.name.startswith("normalized/")]
    assert len(normalize_steps) == len(v21_sources)


def test_dag_stops_at_sample():
    """The ferry emits sample steps only (no other testbed-specific stages)."""
    steps = build_testbed_steps("run0", sources=_ALL_LIST)
    testbed_steps = [s for s in steps if s.name.startswith("datakit-testbed/")]
    assert len(testbed_steps) == len(_ALL_LIST), "expected exactly one testbed step per source (the sampler)"


def test_dag_output_paths_namespaced_by_run_id():
    steps = build_testbed_steps("abc123", sources=_ALL_LIST)
    # Canonical source-pipeline artifacts (download, any transform/preprocess,
    # normalize) are run-independent. Only the testbed-specific sample stage
    # must land under datakit-testbed/abc123/...
    for step in steps:
        if step.name.startswith(("raw/", "processed/", "normalized/")):
            continue
        assert "/abc123/" in step.output_path, f"{step.name} not namespaced: {step.output_path}"


def test_dag_starcoder2_subsets_get_distinct_download_names():
    """StarCoder2-Extras stages each subset under its own path; the download
    names must reflect that disambiguation so per-subset StepSpecs don't collide."""
    subsets = tuple(s for s in _ALL.values() if s.name.startswith("starcoder2/"))
    assert len(subsets) > 1

    download_names = {s.normalize_steps[0].name for s in subsets}
    assert len(download_names) == len(subsets), f"expected distinct downloads, got {download_names}"


def test_dag_full_testbed_builds():
    """Smoke test: building with the full registry does not raise."""
    steps = build_testbed_steps("run0", sources=_ALL_LIST)
    sample_steps = [s for s in steps if s.name.startswith("datakit-testbed/")]
    assert len(sample_steps) == len(_ALL)
