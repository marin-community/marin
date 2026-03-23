# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for region constraint validation in the executor, specifically
the cache-bypass logic that skips region validation for completed steps."""

from unittest.mock import patch

import pytest

from fray.v2.types import ResourceConfig
from marin.execution.executor import _maybe_attach_inferred_region_constraint
from marin.execution.remote import RemoteCallable


def _dummy_fn(output_path: str) -> None:
    pass


def _make_remote_fn(regions: list[str] | None = None) -> RemoteCallable:
    resources = ResourceConfig(regions=regions)
    return RemoteCallable(fn=_dummy_fn, resources=resources)


# Patch targets: skip the Iris backend check so logic runs in unit tests,
# and control the worker region pin and GCS region inference.
_PATCH_IRIS_ACTIVE = patch(
    "marin.execution.executor._iris_backend_is_active", return_value=True
)
_BASE = "marin.execution.executor"


def test_region_mismatch_raises_when_not_cached():
    """Inherited region pin that conflicts with inferred regions raises
    ValueError when the step output is NOT cached."""
    remote_fn = _make_remote_fn()
    with (
        _PATCH_IRIS_ACTIVE,
        patch(f"{_BASE}._iris_worker_region_pin", return_value="us-east5"),
        patch(f"{_BASE}._allowed_regions_for_step", return_value={"us-central2"}),
        patch(f"{_BASE}._step_output_is_cached", return_value=False),
    ):
        with pytest.raises(ValueError, match="pinned to inherited Iris region"):
            _maybe_attach_inferred_region_constraint(
                step_name="tokenized/paloma/4chan",
                remote_fn=remote_fn,
                config={},
                output_path="gs://marin-us-central2/tokenized/paloma/4chan",
                deps=None,
            )


def test_region_mismatch_skipped_when_cached():
    """Inherited region pin that conflicts with inferred regions is skipped
    when the step output IS cached (already completed)."""
    remote_fn = _make_remote_fn()
    with (
        _PATCH_IRIS_ACTIVE,
        patch(f"{_BASE}._iris_worker_region_pin", return_value="us-east5"),
        patch(f"{_BASE}._allowed_regions_for_step", return_value={"us-central2"}),
        patch(f"{_BASE}._step_output_is_cached", return_value=True),
    ):
        result = _maybe_attach_inferred_region_constraint(
            step_name="tokenized/paloma/4chan",
            remote_fn=remote_fn,
            config={},
            output_path="gs://marin-us-central2/tokenized/paloma/4chan",
            deps=None,
        )
    assert result is remote_fn


def test_region_mismatch_with_explicit_regions_raises_when_not_cached():
    """Same conflict with explicit remote_fn regions raises when not cached."""
    remote_fn = _make_remote_fn(regions=["us-central2"])
    with (
        _PATCH_IRIS_ACTIVE,
        patch(f"{_BASE}._iris_worker_region_pin", return_value="us-east5"),
        patch(f"{_BASE}._allowed_regions_for_step", return_value={"us-central2"}),
        patch(f"{_BASE}._step_output_is_cached", return_value=False),
    ):
        with pytest.raises(ValueError, match="pinned to inherited Iris region"):
            _maybe_attach_inferred_region_constraint(
                step_name="tokenized/paloma/4chan",
                remote_fn=remote_fn,
                config={},
                output_path="gs://marin-us-central2/tokenized/paloma/4chan",
                deps=None,
            )


def test_region_mismatch_with_explicit_regions_skipped_when_cached():
    """Same conflict with explicit remote_fn regions is skipped when cached."""
    remote_fn = _make_remote_fn(regions=["us-central2"])
    with (
        _PATCH_IRIS_ACTIVE,
        patch(f"{_BASE}._iris_worker_region_pin", return_value="us-east5"),
        patch(f"{_BASE}._allowed_regions_for_step", return_value={"us-central2"}),
        patch(f"{_BASE}._step_output_is_cached", return_value=True),
    ):
        result = _maybe_attach_inferred_region_constraint(
            step_name="tokenized/paloma/4chan",
            remote_fn=remote_fn,
            config={},
            output_path="gs://marin-us-central2/tokenized/paloma/4chan",
            deps=None,
        )
    assert result is remote_fn


def test_matching_regions_no_cache_check():
    """When inherited region matches inferred region, no cache check needed."""
    remote_fn = _make_remote_fn()
    with (
        _PATCH_IRIS_ACTIVE,
        patch(f"{_BASE}._iris_worker_region_pin", return_value="us-central2"),
        patch(f"{_BASE}._allowed_regions_for_step", return_value={"us-central2"}),
        # _step_output_is_cached should NOT be called when regions match
        patch(f"{_BASE}._step_output_is_cached", side_effect=AssertionError("should not be called")),
    ):
        result = _maybe_attach_inferred_region_constraint(
            step_name="tokenized/paloma/4chan",
            remote_fn=remote_fn,
            config={},
            output_path="gs://marin-us-central2/tokenized/paloma/4chan",
            deps=None,
        )
    assert result is remote_fn
