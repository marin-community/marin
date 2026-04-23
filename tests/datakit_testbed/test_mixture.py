# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Datakit Testbed proportional mixture builder.

Covers the weight-computation logic directly. The
``lm_mixture_data_config`` call inside ``build_testbed_mixture`` loads real
tokenizer steps and is exercised by the training harness end-to-end
rather than mocked here.
"""

import pytest

from experiments.datakit_testbed.mixture import (
    build_testbed_mixture,
    weights_from_rough_counts,
)
from marin.datakit.sources import DatakitSource, all_sources


def _src(name: str, rough: float | None) -> DatakitSource:
    return DatakitSource(name=name, normalize_steps=(), rough_token_count_b=rough)


def test_weights_use_rough_token_counts():
    sources = [_src("a", 100.0), _src("b", 50.0)]
    assert weights_from_rough_counts(sources) == {"a": 100.0, "b": 50.0}


def test_weights_for_full_testbed_source_set():
    """Every canonical testbed source resolves to a positive weight."""
    sources = list(all_sources().values())
    weights = weights_from_rough_counts(sources)
    assert set(weights) == {s.name for s in sources}
    assert all(w > 0 for w in weights.values())


def test_mixture_empty_raises():
    with pytest.raises(ValueError, match="must be non-empty"):
        build_testbed_mixture({})


def test_mixture_explicit_weights_mismatched_keys_raises():
    with pytest.raises(ValueError, match="must match"):
        build_testbed_mixture({"a": object(), "b": object()}, weights={"a": 1.0})  # type: ignore[dict-item]


def test_mixture_missing_source_metadata_raises():
    with pytest.raises(ValueError, match="No DatakitSource metadata"):
        build_testbed_mixture({"has_no_metadata": object()}, sources=[])  # type: ignore[dict-item]
