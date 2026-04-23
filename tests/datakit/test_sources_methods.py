# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for ``DatakitSource.download_step`` and ``.normalize_step``."""

import pytest

from marin.datakit.normalize import DedupMode
from marin.datakit.sources import DatakitSource, all_sources


def test_download_step_builds_for_pinned_source():
    src = all_sources()["coderforge"]
    step = src.download_step()
    assert step.name.startswith("raw/")
    # Staged-path sources pin the output; unstaged ones use the default.
    if src.staged_path is not None:
        assert step.override_output_path == src.staged_path


def test_download_step_disambiguates_same_repo_different_staging():
    """StarCoder2-Extras stages each subset separately; names must not collide."""
    cpp = all_sources()["starcoder2/ir_cpp"]
    kaggle = all_sources()["starcoder2/kaggle"]
    assert cpp.hf_dataset_id == kaggle.hf_dataset_id
    assert cpp.revision == kaggle.revision
    assert cpp.staged_path != kaggle.staged_path
    # Different staging → different download names, via the tail suffix.
    assert cpp.download_step().name != kaggle.download_step().name


def test_normalize_step_names_and_chains_on_download():
    src = all_sources()["coderforge"]
    norm = src.normalize_step()
    assert norm.name == "normalized/coderforge"
    # Normalize depends on its download, not on any run-specific step.
    assert len(norm.deps) == 1
    assert norm.deps[0].name.startswith("raw/")


def test_normalize_step_reuses_caller_supplied_download():
    """Shared family downloads must be honored — normalize should chain to the
    same StepSpec passed in, not build a fresh one."""
    src = all_sources()["nemotron_cc_v2_1/high_quality"]
    dl = src.download_step()
    norm = src.normalize_step(dl)
    assert norm.deps[0] is dl


def test_normalize_step_applies_data_subdir():
    """data_subdir ⇒ normalize's input_path joins download.output_path + subdir."""
    src = all_sources()["nemotron_cc_v2_1/high_quality"]
    assert src.data_subdir == "High-Quality"
    dl = src.download_step()
    norm = src.normalize_step(dl)
    assert norm.hash_attrs["input_path"].endswith("/High-Quality")


def test_normalize_step_preserves_schema_fields_in_hash():
    """text_field / id_field / file_extensions must flow into normalize's hash
    so different schemas produce different cache dirs."""
    src = all_sources()["starcoder2/ir_cpp"]  # text_field="content" is a common override
    norm = src.normalize_step()
    assert norm.hash_attrs["text_field"] == src.text_field
    assert norm.hash_attrs["id_field"] == src.id_field
    # DedupMode is part of normalize's hash — default is EXACT.
    assert norm.hash_attrs["dedup_mode"] == DedupMode.EXACT


def test_download_step_raises_for_unpinned_source():
    src = all_sources()["hplt_v3"]
    assert src.revision is None
    with pytest.raises(ValueError, match="unpinned"):
        src.download_step()


def test_download_step_raises_for_api_sourced():
    src = all_sources()["nsf_awards"]
    assert src.hf_dataset_id == ""
    with pytest.raises(ValueError, match="hf_dataset_id"):
        src.download_step()


def test_normalize_step_falls_back_to_self_download():
    """Omitting the download arg builds one via download_step()."""
    src = all_sources()["coderforge"]
    norm = src.normalize_step()
    dl = norm.deps[0]
    # The fallback path should match what download_step() would produce.
    assert dl.name == src.download_step().name
    assert dl.override_output_path == src.download_step().override_output_path


def test_normalize_step_inherits_download_raise_for_unpinned():
    src = all_sources()["hplt_v3"]
    with pytest.raises(ValueError, match="unpinned"):
        src.normalize_step()


def test_methods_do_not_mutate_the_source():
    """The dataclass is frozen; methods must return fresh StepSpecs, not cache into fields."""
    src = DatakitSource(
        name="synthetic",
        hf_dataset_id="fake/dataset",
        revision="abc1234",
    )
    a = src.download_step()
    b = src.download_step()
    # Content-addressed: identical configs hash the same, so executor dedups —
    # but the objects are independent instantiations, not memoized.
    assert a is not b
    assert a.name == b.name
