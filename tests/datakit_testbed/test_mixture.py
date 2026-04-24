# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Datakit Testbed proportional mixture builder.

Covers the weight-computation wiring directly. The
``lm_mixture_data_config`` call inside ``build_testbed_mixture`` loads
real tokenizer steps and reads ``train/.stats.json`` files on disk — it's
exercised by the training harness end-to-end rather than mocked here.
"""

import pytest

from experiments.datakit_testbed.mixture import build_testbed_mixture


def test_mixture_empty_raises():
    with pytest.raises(ValueError, match="must be non-empty"):
        build_testbed_mixture({})


def test_mixture_explicit_weights_mismatched_keys_raises():
    with pytest.raises(ValueError, match="must match"):
        build_testbed_mixture({"a": object(), "b": object()}, weights={"a": 1.0})  # type: ignore[dict-item]
