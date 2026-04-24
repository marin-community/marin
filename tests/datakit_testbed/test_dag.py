# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the Datakit Testbed ferry DAG."""

import pytest

from experiments.datakit_testbed.dag import build_testbed_steps
from marin.datakit.sources import DatakitSource, all_sources

_ALL = all_sources()


def _source(name: str) -> DatakitSource:
    return _ALL[name]


def test_dag_empty_source_list_raises():
    with pytest.raises(ValueError, match="at least one source"):
        build_testbed_steps("run0", sources=[])


def test_dag_single_source_shape():
    src = _source("nemotron_cc_code_v1/all")
    dag = build_testbed_steps("run0", sources=[src])

    names = [s.name for s in dag.all_steps]
    assert names == [
        "raw/nemotron_cc_code_v1",
        "data/normalized/nemotron_cc_code_v1/all",
        "datakit-testbed/sample/nemotron_cc_code_v1/all",
    ]
    assert set(dag.sampled_by_source.keys()) == {"nemotron_cc_code_v1/all"}


def test_dag_has_one_sample_step_per_source():
    dag = build_testbed_steps("run0")
    sample_names = {s.name for s in dag.all_steps if "/sample/" in s.name}
    assert sample_names == {f"datakit-testbed/sample/{s.name}" for s in _ALL.values()}


def test_dag_nemotron_family_subsets_share_one_download_stepspec():
    """Every Nemotron-CC v2.1 subset's chain starts with the SAME StepSpec object.

    Subsets share one family download because
    :func:`nemotron_v2_normalize_steps` builds the download once and passes it
    to each subset's normalize step. The ferry appends each source's
    ``normalize_steps`` as-is; duplicate downloads in ``all_steps`` are the
    same Python object, which the executor trivially dedupes.
    """
    v21_sources = tuple(s for s in _ALL.values() if s.name.startswith("nemotron_cc_v2_1/"))
    assert len(v21_sources) > 1, "registry must have multiple v2.1 subsets"

    first_download = v21_sources[0].normalize_steps[0]
    for src in v21_sources[1:]:
        assert src.normalize_steps[0] is first_download, "v2.1 subsets must share one download StepSpec"

    dag = build_testbed_steps("run0", sources=v21_sources)
    normalize_steps = [s for s in dag.all_steps if s.name.startswith("data/normalized/")]
    assert len(normalize_steps) == len(v21_sources)


def test_dag_stops_at_sample():
    dag = build_testbed_steps("run0")
    # The ferry stops at sample; tokenize lives in the training harness.
    non_source_stages = {s.name.split("/", 2)[1] for s in dag.all_steps if s.name.startswith("datakit-testbed/")}
    assert non_source_stages == {"sample"}


def test_dag_output_paths_namespaced_by_run_id():
    dag = build_testbed_steps("abc123")
    # Canonical source-pipeline artifacts (download, any transform/preprocess,
    # normalize) are run-independent. Only the testbed-specific sample stage
    # must land under datakit-testbed/abc123/...
    for step in dag.all_steps:
        if step.name.startswith(("raw/", "processed/", "data/normalized/")):
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
    """Smoke test: building the default (pinned) testbed does not raise."""
    dag = build_testbed_steps("run0")
    assert len(dag.sampled_by_source) == len(_ALL)
