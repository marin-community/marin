# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ready engine replacements retain previous generations for accounting."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from finestore.rl.difficulty import Solver
from marin.inference.config import EffectiveServing

from experiments.post_training.task_curation.difficulty import engine_pool
from experiments.post_training.task_curation.difficulty.engine_pool import publish_pool_receipt


def test_restarted_pool_replaces_legacy_endpoint_and_retains_all_ready_receipts(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    original = {
        "solver": "large",
        "model": "pinned-model",
        "revision": "pinned-revision",
        "endpoint_name": "failed-endpoint",
        "ready_at": "2026-11-23T12:00:00+00:00",
    }
    (registry / "large.json").write_text(json.dumps(original))
    replacement = {**original, "endpoint_name": "replacement-endpoint", "ready_at": "2026-11-24T15:00:00+00:00"}
    retried = {**replacement, "endpoint_name": "retried-endpoint", "ready_at": "2026-11-24T16:00:00+00:00"}

    publish_pool_receipt(str(registry), Solver.LARGE, replacement)
    assert json.loads((registry / "large.json").read_text()) == replacement
    publish_pool_receipt(str(registry), Solver.LARGE, retried)
    publish_pool_receipt(str(registry), Solver.LARGE, retried)

    assert json.loads((registry / "large.json").read_text()) == retried
    history = [json.loads(path.read_text()) for path in (registry / "history" / "large").glob("*.json")]
    assert len(history) == 3
    assert all(receipt in history for receipt in (original, replacement, retried))


def test_pool_publishes_real_engine_config_after_warmup(tmp_path, monkeypatch):
    def stop_after_ready():
        raise RuntimeError("stop after ready")

    ready = SimpleNamespace(
        effective_serving=EffectiveServing(2, 2, 1, 8, 65536),
        model=SimpleNamespace(endpoint=SimpleNamespace(base_url="http://ready-engine/v1")),
    )

    @contextmanager
    def remote_inference(config):
        yield SimpleNamespace(
            ready_direct_session=lambda: ready,
            endpoint_name="/serve/replacement",
            jobs=[],
            check_alive=stop_after_ready,
        )

    async def warm_throughput(*args):
        return {"requests": 128, "output_tokens": 32768}

    monkeypatch.setattr(engine_pool, "remote_inference", remote_inference)
    monkeypatch.setattr(engine_pool, "warm_throughput", warm_throughput)
    with pytest.raises(RuntimeError, match="stop after ready"):
        engine_pool.run_pool(Solver.LARGE, str(tmp_path))

    receipt = json.loads((tmp_path / "large.json").read_text())
    assert receipt["endpoint_name"] == "/serve/replacement"
    assert receipt["engine"]["extra_metric_families"] == []
    assert receipt["warm_throughput"]["output_tokens"] == 32768
