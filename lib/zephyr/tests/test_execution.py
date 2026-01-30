# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the actor-based execution engine (ZephyrContext)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fray.v2 import LocalClient
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, shard_ctx


@pytest.fixture
def client():
    c = LocalClient()
    yield c
    c.shutdown()


@pytest.fixture
def ctx(client):
    zctx = ZephyrContext(client=client, num_workers=2)
    yield zctx
    zctx.shutdown()


def test_simple_map(ctx):
    """Map pipeline produces correct results."""
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    results = list(ctx.execute(ds))
    assert sorted(results) == [2, 4, 6]


def test_filter(ctx):
    """Filter pipeline produces correct results."""
    ds = Dataset.from_list([1, 2, 3, 4, 5]).filter(lambda x: x > 3)
    results = list(ctx.execute(ds))
    assert sorted(results) == [4, 5]


def test_shared_data(client):
    """Workers can access shared data via shard_ctx()."""

    def use_shared(x):
        multiplier = shard_ctx().get_shared("multiplier")
        return x * multiplier

    zctx = ZephyrContext(client=client, num_workers=1)
    zctx.put("multiplier", 10)
    ds = Dataset.from_list([1, 2, 3]).map(use_shared)
    results = list(zctx.execute(ds))
    assert sorted(results) == [10, 20, 30]
    zctx.shutdown()


def test_multi_stage(ctx):
    """Multi-stage pipeline (map + filter) works."""
    ds = Dataset.from_list([1, 2, 3, 4, 5]).map(lambda x: x * 2).filter(lambda x: x > 5)
    results = list(ctx.execute(ds))
    assert sorted(results) == [6, 8, 10]


def test_context_manager(client):
    """ZephyrContext works as context manager."""
    with ZephyrContext(client=client, num_workers=1) as zctx:
        ds = Dataset.from_list([1, 2, 3]).map(lambda x: x + 1)
        results = list(zctx.execute(ds))
    assert sorted(results) == [2, 3, 4]


def test_write_jsonl(tmp_path, ctx):
    """Pipeline writing to jsonl file."""
    output = str(tmp_path / "out-{shard}.jsonl")
    ds = Dataset.from_list([{"a": 1}, {"a": 2}, {"a": 3}]).write_jsonl(output)
    results = list(ctx.execute(ds))
    assert len(results) == 3
    # Verify all files were written and contain correct data
    all_records = []
    for path_str in results:
        written = Path(path_str)
        assert written.exists()
        lines = written.read_text().strip().split("\n")
        all_records.extend(json.loads(line) for line in lines)
    assert sorted(all_records, key=lambda r: r["a"]) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_dry_run(ctx):
    """Dry run shows plan without executing."""
    ds = Dataset.from_list([1, 2, 3]).map(lambda x: x * 2)
    results = list(ctx.execute(ds, dry_run=True))
    assert results == []


def test_flat_map(ctx):
    """FlatMap pipeline produces correct results."""
    ds = Dataset.from_list([1, 2, 3]).flat_map(lambda x: [x, x * 10])
    results = list(ctx.execute(ds))
    assert sorted(results) == [1, 2, 3, 10, 20, 30]


def test_empty_dataset(ctx):
    """Empty dataset produces empty results."""
    ds = Dataset.from_list([])
    results = list(ctx.execute(ds))
    assert results == []
