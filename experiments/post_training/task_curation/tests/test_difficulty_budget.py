# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pool cost accounting across controller restarts and serving replacements."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from finestore.rl.difficulty import Solver

from experiments.post_training.task_curation.difficulty import budget
from experiments.post_training.task_curation.difficulty.ledger import read_rows


@pytest.mark.parametrize("platform", ["tpu", "gpu"])
def test_pool_replacement_excludes_unallocated_gap_and_resumes_after_restart(tmp_path, monkeypatch, platform):
    ledger = str(tmp_path / "ledger")
    registry = tmp_path / "pools"
    registry.mkdir()
    receipt_path = registry / "large.json"
    clock = datetime(2026, 11, 27, tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.astimezone(tz)

    monkeypatch.setattr(budget, "datetime", Clock)

    def publish(ready_at, count, revision):
        receipt_path.write_text(
            json.dumps(
                dict(
                    ready_at=ready_at,
                    revision=revision,
                    platform=platform,
                    chip_count=count,
                    gpu_count=0 if platform == "tpu" else count,
                )
            )
        )

    def reconcile():
        # Reload persisted rows as a restarted controller does.
        return budget.record_pool_cost(ledger, str(registry), read_rows(ledger), solvers=frozenset({Solver.LARGE}))

    publish("2026-11-27T00:00:00+00:00", 32, "first-generation")
    clock = datetime(2026, 11, 27, 1, tzinfo=UTC)
    first = reconcile()
    clock = datetime(2026, 11, 27, 2, tzinfo=UTC)
    second = reconcile()
    original_partitions = {path: path.read_bytes() for path in (tmp_path / "ledger/v1/rows").glob("*.jsonl")}

    # The replacement starts two hours after the last allocation was accounted.
    # Use an equivalent non-UTC timestamp to exercise chronological comparison.
    publish("2026-11-26T23:00:00-05:00", 64, "second-generation")
    clock = datetime(2026, 11, 27, 5, tzinfo=UTC)
    replacement = reconcile()
    clock = datetime(2026, 11, 27, 6, tzinfo=UTC)
    resumed = reconcile()
    assert reconcile() == []

    # A warm overlapping replacement must not charge time already accounted.
    publish("2026-11-27T05:30:00+00:00", 32, "third-generation")
    clock = datetime(2026, 11, 27, 7, tzinfo=UTC)
    overlapping = reconcile()
    costs = first + second + replacement + resumed + overlapping
    assert len(costs) == 5
    if platform == "tpu":
        accounts = [json.loads(Path(row.evidence_uri).read_text()) for row in costs if row.evidence_uri is not None]
        assert [value["chip_hours"] for value in accounts] == [32, 32, 64, 64, 32]
        assert datetime.fromisoformat(accounts[2]["start"]) == datetime(2026, 11, 27, 4, tzinfo=UTC)
        assert [row.gpu_hours for row in costs] == [0, 0, 0, 0, 0]
    else:
        assert [row.gpu_hours for row in costs] == [32, 32, 64, 64, 32]
    assert {row.attempt_id for row in read_rows(ledger)} == {row.attempt_id for row in costs}
    assert all(path.read_bytes() == payload for path, payload in original_partitions.items())
