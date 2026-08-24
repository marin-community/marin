# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict

from experiments.june_tpu_67b_a2b.moe.launch_datakit_moe_mix import _phase_weights
from experiments.june_tpu_67b_a2b.moe.long_context_datakit_moe_mix import (
    LONG_CONTEXT_SKEW,
    long_context_datakit_components,
    long_context_phase_weights,
)


def _quality_domain_weights(weights: dict[str, float]) -> dict[str, float]:
    quality_domain_weights: defaultdict[str, float] = defaultdict(float)
    for name, weight in weights.items():
        bucket, _ = name.split("-", 1)
        quality_domain_weights[bucket] += weight
    return quality_domain_weights


def test_default_long_context_skew_reproduces_original_mixture():
    original = _phase_weights(1)
    weights = long_context_phase_weights()
    components = long_context_datakit_components()
    quality_domain_weights = _quality_domain_weights(weights)

    assert LONG_CONTEXT_SKEW == 1.0
    assert components.keys() == weights.keys()
    assert len(weights) == 391
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
    for bucket, weight in original.items():
        if bucket == "tail":
            continue
        assert math.isclose(quality_domain_weights[bucket], weight, abs_tol=1e-12)
    assert math.isclose(
        sum(weight for bucket, weight in quality_domain_weights.items() if bucket not in original),
        original["tail"],
        abs_tol=1e-12,
    )

    assert math.isclose(
        weights["c01q0-gt_64k"] / weights["c01q0-lte_64k"],
        57_111_643_914 / 95_502_129_802,
        rel_tol=1e-12,
    )


def test_long_context_skew_doubles_long_to_short_weight_without_changing_bucket_mass():
    baseline = long_context_phase_weights(1.0)
    skewed = long_context_phase_weights(2.0)

    baseline_quality_domain = _quality_domain_weights(baseline)
    skewed_quality_domain = _quality_domain_weights(skewed)
    assert baseline_quality_domain.keys() == skewed_quality_domain.keys()
    for bucket in baseline_quality_domain:
        assert math.isclose(baseline_quality_domain[bucket], skewed_quality_domain[bucket], abs_tol=1e-12)

    long_buckets = {name.removesuffix("-gt_64k") for name in baseline if name.endswith("-gt_64k")}
    for bucket in baseline_quality_domain:
        short_weight = skewed[f"{bucket}-lte_64k"]
        if bucket in long_buckets:
            baseline_ratio = baseline[f"{bucket}-gt_64k"] / baseline[f"{bucket}-lte_64k"]
            skewed_ratio = skewed[f"{bucket}-gt_64k"] / short_weight
            assert math.isclose(skewed_ratio, 2 * baseline_ratio, rel_tol=1e-12)
        else:
            assert math.isclose(short_weight, baseline[f"{bucket}-lte_64k"], abs_tol=1e-12)
