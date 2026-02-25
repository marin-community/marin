# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

import levanter.callbacks.tensorstore_callbacks as tensorstore_callbacks


def test_tensorstore_metrics_interval_from_env(monkeypatch):
    monkeypatch.delenv("LEVANTER_LOG_TENSORSTORE_METRICS_EVERY", raising=False)
    assert tensorstore_callbacks.tensorstore_metrics_interval_from_env() is None

    monkeypatch.setenv("LEVANTER_LOG_TENSORSTORE_METRICS_EVERY", "0")
    assert tensorstore_callbacks.tensorstore_metrics_interval_from_env() is None

    monkeypatch.setenv("LEVANTER_LOG_TENSORSTORE_METRICS_EVERY", "-5")
    assert tensorstore_callbacks.tensorstore_metrics_interval_from_env() is None

    monkeypatch.setenv("LEVANTER_LOG_TENSORSTORE_METRICS_EVERY", "7")
    assert tensorstore_callbacks.tensorstore_metrics_interval_from_env() == 7


def test_build_tensorstore_metrics_logger_raises_for_nonpositive_interval():
    with pytest.raises(ValueError, match="every must be positive"):
        tensorstore_callbacks.build_tensorstore_metrics_logger(0)


def test_build_tensorstore_metrics_logger_logs_totals_and_deltas(monkeypatch):
    path_to_metric_name = {spec.tensorstore_path: spec.name for spec in tensorstore_callbacks._METRIC_SPECS}
    rounds = [
        {
            "cache_hit_count": 10.0,
            "cache_miss_count": 5.0,
        },
        {
            "cache_hit_count": 15.0,
            "cache_miss_count": 7.0,
        },
    ]
    round_index = {"value": 0}

    def fake_metric_total(metric_path: str, value_key: str):
        del value_key
        metric_name = path_to_metric_name.get(metric_path)
        if metric_name is None:
            return False, 0.0
        if metric_name not in rounds[round_index["value"]]:
            return False, 0.0
        return True, rounds[round_index["value"]][metric_name]

    log_calls: list[tuple[dict[str, float], int | None]] = []

    def fake_log(metrics, step=None):
        log_calls.append((metrics, step))

    monkeypatch.setattr(tensorstore_callbacks, "_metric_total", fake_metric_total)
    monkeypatch.setattr("levanter.tracker.log", fake_log)

    logger = tensorstore_callbacks.build_tensorstore_metrics_logger(every=1)
    logger(step=0)
    round_index["value"] = 1
    logger(step=1)

    assert len(log_calls) == 2

    first_metrics, first_step = log_calls[0]
    assert first_step == 0
    assert first_metrics["data/tensorstore/cache_hit_count_total"] == 10.0
    assert first_metrics["data/tensorstore/cache_miss_count_total"] == 5.0
    assert first_metrics["data/tensorstore/cache_hit_count_delta"] == 0.0
    assert first_metrics["data/tensorstore/cache_miss_count_delta"] == 0.0
    assert first_metrics["data/tensorstore/cache_hit_rate_total"] == pytest.approx(10.0 / 15.0)

    second_metrics, second_step = log_calls[1]
    assert second_step == 1
    assert second_metrics["data/tensorstore/cache_hit_count_total"] == 15.0
    assert second_metrics["data/tensorstore/cache_miss_count_total"] == 7.0
    assert second_metrics["data/tensorstore/cache_hit_count_delta"] == 5.0
    assert second_metrics["data/tensorstore/cache_miss_count_delta"] == 2.0
    assert second_metrics["data/tensorstore/cache_hit_rate_total"] == pytest.approx(15.0 / 22.0)
    assert second_metrics["data/tensorstore/cache_hit_rate_delta"] == pytest.approx(5.0 / 7.0)
