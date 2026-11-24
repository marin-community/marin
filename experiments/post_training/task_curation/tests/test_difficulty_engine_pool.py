# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ready engine replacements retain previous generations for accounting."""

import json

from finestore.rl.difficulty import Solver

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
