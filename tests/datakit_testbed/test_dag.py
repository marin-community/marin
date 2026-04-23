# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural tests for the Datakit Testbed ferry DAG."""

import pytest

from experiments.datakit_testbed.dag import build_testbed_steps
from marin.datakit.sources import DatakitSource, pinned_sources

_PINNED = pinned_sources()


def _source(name: str) -> DatakitSource:
    return _PINNED[name]


def test_dag_empty_source_list_raises():
    with pytest.raises(ValueError, match="at least one source"):
        build_testbed_steps("run0", sources=[])


def test_dag_single_source_shape():
    src = _source("nemotron_cc_code_v1/all")
    dag = build_testbed_steps("run0", sources=[src])

    names = [s.name for s in dag.all_steps]
    assert names == [
        "datakit-testbed/download/nvidia__Nemotron-CC-Code-v1__nemotron_cc_code_v1-c55cd9",
        "normalized/nemotron_cc_code_v1/all",
        "datakit-testbed/sample/nemotron_cc_code_v1/all",
        "datakit-testbed/noop_dedup",
        "datakit-testbed/consolidate/nemotron_cc_code_v1/all",
    ]
    assert set(dag.consolidated_by_source.keys()) == {"nemotron_cc_code_v1/all"}


def test_dag_has_one_sample_step_per_source():
    dag = build_testbed_steps("run0")
    sample_names = {s.name for s in dag.all_steps if "/sample/" in s.name}
    assert sample_names == {f"datakit-testbed/sample/{s.name}" for s in _PINNED.values()}


def test_dag_groups_downloads_by_hf_id_revision():
    """Every Nemotron-CC v2.1 subset shares one family download."""
    v21_sources = tuple(s for s in _PINNED.values() if s.name.startswith("nemotron_cc_v2_1/"))
    assert len(v21_sources) > 1, "registry must have multiple v2.1 subsets"

    dag = build_testbed_steps("run0", sources=v21_sources)

    download_steps = [s for s in dag.all_steps if "/download/" in s.name]
    assert len(download_steps) == 1, "expected exactly one download for a single family"

    normalize_steps = [s for s in dag.all_steps if s.name.startswith("normalized/")]
    assert len(normalize_steps) == len(v21_sources)


def test_dag_single_shared_dedup_step():
    dag = build_testbed_steps("run0")
    dedup_steps = [s for s in dag.all_steps if s.name == "datakit-testbed/noop_dedup"]
    assert len(dedup_steps) == 1


def test_dag_per_source_consolidate():
    dag = build_testbed_steps("run0")
    consolidate_names = {s.name for s in dag.all_steps if "/consolidate/" in s.name}
    expected_names = {f"datakit-testbed/consolidate/{s.name}" for s in _PINNED.values()}
    assert consolidate_names == expected_names
    # The ferry stops at consolidate; tokenize lives in the training harness.
    tokenize_names = {s.name for s in dag.all_steps if "/tokenize/" in s.name}
    assert tokenize_names == set()


def test_dag_output_paths_namespaced_by_run_id():
    dag = build_testbed_steps("abc123")
    # Downloads and normalize outputs are shared canonical artifacts — not run-id scoped.
    # Every other step (sample, dedup, consolidate) must land under datakit-testbed/abc123/...
    for step in dag.all_steps:
        if "/download/" in step.name or step.name.startswith("normalized/"):
            continue
        assert "/abc123/" in step.output_path, f"{step.name} not namespaced: {step.output_path}"


def test_dag_same_repo_different_staged_paths_get_separate_downloads():
    """StarCoder2-Extras-shaped case: same (hf_repo, revision), different staged_path per subset."""
    a = DatakitSource(name="A", hf_dataset_id="foo/bar", revision="abc1234", staged_path="raw/foo/sub_a")
    b = DatakitSource(name="B", hf_dataset_id="foo/bar", revision="abc1234", staged_path="raw/foo/sub_b")
    dag = build_testbed_steps("run0", sources=[a, b])
    download_names = sorted(s.name for s in dag.all_steps if "/download/" in s.name)
    assert download_names == [
        "datakit-testbed/download/foo__bar__sub_a",
        "datakit-testbed/download/foo__bar__sub_b",
    ]


def test_dag_full_testbed_builds():
    """Smoke test: building the default (pinned) testbed does not raise."""
    dag = build_testbed_steps("run0")
    assert len(dag.consolidated_by_source) == len(_PINNED)
