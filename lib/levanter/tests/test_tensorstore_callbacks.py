# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import levanter.callbacks.tensorstore_callbacks as tensorstore_callbacks


class _FakeTrainer:
    def __init__(self):
        self.hooks: list[tuple[object, int]] = []

    def add_hook(self, hook, *, every: int):
        self.hooks.append((hook, every))


def test_install_tensorstore_metrics_hook_logs_only_discovered_metrics(monkeypatch):
    totals = {
        "/tensorstore/cache/hit_count": 10.0,
        "/tensorstore/cache/miss_count": 2.0,
        "/tensorstore/kvstore/gcs/read": 100.0,
    }
    logs = []
    trainer = _FakeTrainer()

    def _fake_collect(pattern: str, include_zero_metrics: bool = False):
        assert include_zero_metrics
        if pattern not in totals:
            return []
        return [{"name": pattern, "values": [{"value": totals[pattern]}]}]

    monkeypatch.setattr(tensorstore_callbacks.ts, "experimental_collect_matching_metrics", _fake_collect)
    monkeypatch.setattr(
        tensorstore_callbacks.levanter.tracker, "log", lambda metrics, step: logs.append((step, metrics))
    )
    monkeypatch.setenv("LEVANTER_LOG_TENSORSTORE_METRICS_EVERY", "5")

    tensorstore_callbacks.install_tensorstore_metrics_hook_if_enabled(trainer)

    assert len(trainer.hooks) == 1
    hook, every = trainer.hooks[0]
    assert every == 5

    hook(SimpleNamespace(step=10))
    totals["/tensorstore/cache/hit_count"] += 3.0
    totals["/tensorstore/cache/miss_count"] += 1.0
    totals["/tensorstore/kvstore/gcs/read"] += 7.0
    hook(SimpleNamespace(step=15))

    assert len(logs) == 2
    _, second = logs[1]
    assert second["data/tensorstore/cache_hit_count_delta"] == 3.0
    assert second["data/tensorstore/cache_miss_count_delta"] == 1.0
    assert second["data/tensorstore/gcs_read_count_delta"] == 7.0
    assert "data/tensorstore/gcs_grpc_read_count_total" not in second
