# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.grug.moe.standalone import bench_mxfp8_dense as benchmark


def test_sample_statistics_reports_median_and_mad():
    median, mad = benchmark.sample_statistics([1.0, 2.0, 3.0, 100.0])

    assert median == 2.5
    assert mad == 1.0


def test_weighted_production_ratio_uses_dense_operation_mix():
    ratio = benchmark.weighted_production_ratio(
        [
            (5, 1.01, 1.0),
            (2, 0.95, 1.0),
        ]
    )

    assert ratio == pytest.approx(6.95 / 7.0)
