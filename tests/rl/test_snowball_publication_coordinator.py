# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.launch_snowball_publication_gate import guard_absent


@pytest.mark.parametrize("existing", ["terminal", "receipt", "selection"])
def test_duplicate_storage_blocks_coordinator_before_launch(monkeypatch, existing):
    monkeypatch.setattr(StoragePath, "exists", lambda self: str(self) == existing)
    monkeypatch.setattr(StoragePath, "glob", lambda self: [StoragePath("receipt")] if existing == "receipt" else [])
    with pytest.raises(AssertionError, match=r"blocks? duplicate gate"):
        guard_absent("terminal", "receipts/*", "selection")


def test_empty_storage_allows_coordinator_guard(monkeypatch):
    monkeypatch.setattr(StoragePath, "exists", lambda self: False)
    monkeypatch.setattr(StoragePath, "glob", lambda self: [])
    guard_absent("terminal", "receipts/*", "selection")
