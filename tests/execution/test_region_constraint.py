# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for executor region inference logic."""

from dataclasses import dataclass

import pytest
from marin.execution.executor import (
    _infer_gcs_regions,
)
from marin.execution.step_spec import StepSpec


@dataclass(frozen=True)
class FakeConfig:
    input_path: str = ""
    output_path: str = ""


class TestInferGcsRegionsCrossRegion:
    """Verify that _infer_gcs_regions handles cross-region paths correctly."""

    def test_cross_region_raises_by_default(self):
        """Config in one region + output in another raises ValueError."""
        config = FakeConfig(input_path="gs://marin-us-central1/data/tok")
        with pytest.raises(ValueError, match="cross-region"):
            _infer_gcs_regions(
                step_name="tokenize",
                config=config,
                output_path="gs://marin-us-central2/tokenize-abc123",
                deps=None,
            )

    def test_cross_region_allowed_returns_output_region(self):
        """When allow_cross_region=True, returns only the output_path region."""
        config = FakeConfig(input_path="gs://marin-us-central1/data/tok")
        result = _infer_gcs_regions(
            step_name="tokenize",
            config=config,
            output_path="gs://marin-us-central2/tokenize-abc123",
            deps=None,
            allow_cross_region=True,
        )
        assert result == ["us-central2"]

    def test_cross_region_allowed_no_gcs_output(self):
        """When allow_cross_region=True but output is local, single config region is returned."""
        config = FakeConfig(input_path="gs://marin-us-central1/data/tok")
        result = _infer_gcs_regions(
            step_name="tokenize",
            config=config,
            output_path="/tmp/local/tokenize-abc123",
            deps=None,
            allow_cross_region=True,
        )
        # Only one GCS region exists (from config), so no cross-region conflict.
        assert result == ["us-central1"]

    def test_cross_region_allowed_with_deps(self):
        """When allow_cross_region=True with cross-region deps, returns output region."""
        config = FakeConfig(input_path="gs://marin-us-central1/data/tok")
        dep = StepSpec(name="dep", override_output_path="gs://marin-us-central1/dep-out")
        result = _infer_gcs_regions(
            step_name="tokenize",
            config=config,
            output_path="gs://marin-us-central2/tokenize-abc123",
            deps=[dep],
            allow_cross_region=True,
        )
        assert result == ["us-central2"]

    def test_same_region_unaffected_by_allow_flag(self):
        """When all paths are same region, allow_cross_region doesn't change result."""
        config = FakeConfig(input_path="gs://marin-us-central2/data/tok")
        result = _infer_gcs_regions(
            step_name="tokenize",
            config=config,
            output_path="gs://marin-us-central2/tokenize-abc123",
            deps=None,
            allow_cross_region=True,
        )
        assert result == ["us-central2"]

    def test_no_gcs_paths_returns_none(self):
        """When no GCS paths are present, returns None regardless of flag."""
        config = FakeConfig(input_path="/local/data")
        result = _infer_gcs_regions(
            step_name="tokenize",
            config=config,
            output_path="/tmp/tokenize-abc123",
            deps=None,
            allow_cross_region=True,
        )
        assert result is None
