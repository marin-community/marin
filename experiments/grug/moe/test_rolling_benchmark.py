# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from experiments.grug.moe.rolling_benchmark import (
    PlateauRequirements,
    PlateauWindow,
    frozen_cohort_slots,
    histogram_quantile_delta,
    parse_labeled_prometheus,
    prometheus_value,
    prometheus_values_by_label,
)


def test_labeled_prometheus_preserves_engines_and_window_histograms() -> None:
    before = parse_labeled_prometheus(
        """
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{engine="0"} 100
vllm:generation_tokens_total{engine="1"} 200
vllm:num_requests_running{engine="0"} 2
vllm:num_requests_running{engine="1"} 3
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0"} 2
vllm:time_to_first_token_seconds_bucket{engine="0",le="2.0"} 4
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 4
"""
    )
    after = parse_labeled_prometheus(
        """
vllm:generation_tokens_total{engine="0"} 110
vllm:generation_tokens_total{engine="1"} 220
vllm:num_requests_running{engine="0"} 4
vllm:num_requests_running{engine="1"} 5
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0"} 3
vllm:time_to_first_token_seconds_bucket{engine="0",le="2.0"} 8
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 8
"""
    )

    assert prometheus_value(after, "vllm:generation_tokens") == 330
    assert prometheus_values_by_label(after, "vllm:num_requests_running", label="engine") == {
        "0": 4,
        "1": 5,
    }
    assert histogram_quantile_delta(before, after, "vllm:time_to_first_token_seconds", 0.5) == pytest.approx(4 / 3)


def test_frozen_cohort_slots_keep_equal_load_and_cover_each_manifest() -> None:
    requests = [
        {"request_id": f"{cohort}-{index}", "cohort": cohort}
        for cohort in ("short", "medium", "long")
        for index in range(6)
    ]

    slots = frozen_cohort_slots(requests, target_concurrency=6)
    observed = {cohort: set() for cohort in ("short", "medium", "long")}
    for _ in range(3):
        for slot in slots:
            request = slot.next_request()
            observed[slot.cohort].add(request["request_id"])

    assert {cohort: sum(slot.cohort == cohort for slot in slots) for cohort in observed} == {
        "short": 2,
        "medium": 2,
        "long": 2,
    }
    assert all(len(request_ids) == 6 for request_ids in observed.values())


def test_plateau_discards_a_load_dip_and_closes_only_after_all_floors() -> None:
    requirements = PlateauRequirements(
        target_concurrency=20,
        minimum_seconds=120,
        minimum_generated_tokens=250_000,
        required_request_ids=frozenset({"short", "medium", "long"}),
    )
    plateau = PlateauWindow(requirements)

    assert (
        plateau.observe_in_flight(
            now=10,
            in_flight=19,
            generation_counter=1_000,
            prompt_counter=2_000,
        )
        == "opened"
    )
    plateau.record_completion(request_id="short", cohort="short", completion_tokens=2_048, succeeded=True)
    assert plateau.observe_in_flight(now=20, in_flight=18) == "discarded"

    assert (
        plateau.observe_in_flight(
            now=30,
            in_flight=20,
            generation_counter=3_000,
            prompt_counter=4_000,
        )
        == "opened"
    )
    for request_id in ("short", "medium", "long"):
        plateau.record_completion(
            request_id=request_id,
            cohort=request_id,
            completion_tokens=2_048,
            succeeded=True,
        )

    assert not plateau.ready_to_close(now=150, in_flight=19, generation_counter=253_000)
    assert plateau.ready_to_close(now=150, in_flight=20, generation_counter=253_000)
    result = plateau.close(
        now=150,
        in_flight=20,
        generation_counter=253_000,
        prompt_counter=104_000,
    )

    assert result["elapsed_seconds"] == 120
    assert result["generated_tokens"] == 250_000
    assert result["in_flight"]["at_close"] == 20
    assert result["manifest"] == {"expected": 3, "observed": 3, "passed": True}
    assert result["discarded_windows"][0]["reason"] == "in_flight_below_95_percent"
