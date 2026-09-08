# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.launch_qwen_weight_sync_gate import guard_absent as qwen_guard_absent
from experiments.post_training.launch_snowball_publication_gate import guard_absent


@pytest.mark.parametrize("existing", ["terminal", "receipt", "selection"])
@pytest.mark.parametrize("guard", [guard_absent, qwen_guard_absent])
def test_duplicate_storage_blocks_coordinator_before_launch(monkeypatch, existing, guard):
    monkeypatch.setattr(StoragePath, "exists", lambda self: str(self) == existing)
    monkeypatch.setattr(StoragePath, "glob", lambda self: [StoragePath("receipt")] if existing == "receipt" else [])
    with pytest.raises(AssertionError, match=r"blocks? duplicate gate"):
        guard("terminal", "receipts/*", "selection")


@pytest.mark.parametrize("guard", [guard_absent, qwen_guard_absent])
def test_empty_storage_allows_coordinator_guard(monkeypatch, guard):
    monkeypatch.setattr(StoragePath, "exists", lambda self: False)
    monkeypatch.setattr(StoragePath, "glob", lambda self: [])
    guard("terminal", "receipts/*", "selection")
