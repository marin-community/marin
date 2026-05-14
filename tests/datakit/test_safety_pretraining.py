# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Safety Pretraining canonical pipelines."""

import pytest
from marin.datakit.canonical.safety_pretraining import (
    SAFETY_PRETRAINING_FAMILIES,
    download_fineweb_annotated,
    download_moral_education,
    download_refuseweb,
    download_safeweb,
)
from marin.datakit.normalize import normalize_step

_FAMILY_DOWNLOADERS = {
    "moral_education": download_moral_education,
    "safeweb": download_safeweb,
    "refuseweb": download_refuseweb,
    "fineweb_annotated": download_fineweb_annotated,
}


@pytest.mark.parametrize("family", sorted(SAFETY_PRETRAINING_FAMILIES))
def test_download_step_pins_family_revision(family: str):
    """Each ``download_*`` returns a StepSpec whose name and override path
    encode the family's pinned revision, so cache identity is stable across
    invocations."""
    hf_dataset_id, default_revision, _ = SAFETY_PRETRAINING_FAMILIES[family]
    step = _FAMILY_DOWNLOADERS[family]()
    assert step.name == f"raw/{family}"
    assert step.hash_attrs["hf_dataset_id"] == hf_dataset_id
    assert step.hash_attrs["revision"] == default_revision
    assert step.output_path.endswith(f"raw/{family}-{default_revision}")


@pytest.mark.parametrize("family", sorted(SAFETY_PRETRAINING_FAMILIES))
def test_normalize_subsets_have_distinct_output_paths(family: str):
    """Each top-level score-bucket directory must produce a distinct normalize
    output path. Catches regressions where ``relative_input_path`` stops
    flowing into ``hash_attrs`` and collapses subset caches together."""
    _, _, subsets = SAFETY_PRETRAINING_FAMILIES[family]
    download = _FAMILY_DOWNLOADERS[family]()
    steps = [
        normalize_step(
            name=f"normalized/{family}",
            download=download,
            relative_input_path=subset,
        )
        for subset in subsets
    ]
    output_paths = {step.output_path for step in steps}
    assert len(output_paths) == len(subsets)
