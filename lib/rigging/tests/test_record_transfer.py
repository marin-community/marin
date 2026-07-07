# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest
from rigging import filesystem as rfs
from rigging.filesystem import TransferBudget, TransferBudgetExceeded, record_transfer


@pytest.fixture()
def patched_regions(monkeypatch):
    """VM is in us-central1; bucket lookup is faked from the URL."""
    monkeypatch.setattr(rfs, "cached_marin_region", lambda: "us-central1")
    monkeypatch.setattr(
        rfs,
        "_cached_bucket_location",
        lambda bucket: {
            "marin-us-central1": "us-central1",
            "marin-us-central2": "us-central2",
        }.get(bucket),
    )


def test_record_transfer_charges_cross_region(patched_regions):
    budget = TransferBudget(limit_bytes=1024 * 1024)
    record_transfer(1000, "gs://marin-us-central2/checkpoint", budget=budget)
    assert budget.bytes_used == 1000


def test_record_transfer_skips_same_region_and_local(patched_regions):
    budget = TransferBudget(limit_bytes=1024 * 1024)
    record_transfer(1000, "gs://marin-us-central1/checkpoint", budget=budget)
    record_transfer(1000, "/tmp/checkpoint", budget=budget)
    assert budget.bytes_used == 0


def test_record_transfer_raises_when_budget_exceeded(patched_regions):
    budget = TransferBudget(limit_bytes=500)
    with pytest.raises(TransferBudgetExceeded):
        record_transfer(1000, "gs://marin-us-central2/checkpoint", budget=budget)


def test_transfer_budget_exceeded_round_trips_through_pickle():
    # The exception crosses process boundaries (Zephyr ships shard/coordinator
    # errors via cloudpickle). It must revive without a constructor TypeError.
    original = TransferBudgetExceeded(bytes_used=9_960, attempted=400, limit=10_000, path="gs://marin-us-east5/x")
    revived = pickle.loads(pickle.dumps(original))
    assert (revived.bytes_used, revived.attempted, revived.limit, revived.path) == (
        9_960,
        400,
        10_000,
        "gs://marin-us-east5/x",
    )
    assert str(revived) == str(original)
    assert "Cross-region transfer budget exceeded" in str(revived)
