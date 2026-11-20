# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest
from levanter.store.cache import CacheLedger, CacheMetadata, TreeCache, _merge_sharded_ledgers

from experiments.datakit.store.bucket_writer import write_bucket_cache
from experiments.grug.snowball_ladder import sample_mix
from experiments.grug.snowball_ladder.sample_mix import OutputCache, Recipe

EOS = 7
_EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}


def test_document_copies_match_band_width_in_expectation():
    u = np.random.default_rng(0).random(200_000)
    for band in [(0.0, 0.3), (0.3, 0.9), (0.9, 1.4), (0.0, 1.27), (1.27, 1.6)]:
        copies = sample_mix.document_copies(u, band)
        assert copies.min() >= 0
        assert copies.mean() == pytest.approx(band[1] - band[0], abs=0.005)


def test_bands_inside_unit_interval_are_disjoint():
    u = np.random.default_rng(1).random(100_000)
    main = sample_mix.document_copies(u, (0.0, 0.6))
    cooldown = sample_mix.document_copies(u, (0.6, 0.95))
    assert not np.any((main > 0) & (cooldown > 0))
    assert main.max() == 1 and cooldown.max() == 1


def test_group_weight_splits_by_member_tokens():
    weights = sample_mix.resolve_cell_weights(
        {"c01q0": 0.5, "tail": 0.5}, {"tail": ("c02q0", "c03q0")}, {"c01q0": 10, "c02q0": 30, "c03q0": 10}
    )
    assert weights == pytest.approx({"c01q0": 0.5, "c02q0": 0.375, "c03q0": 0.125})


def test_group_shards_fills_parts_in_order():
    shards = [{"key": f"s{i}", "tokens": tokens} for i, tokens in enumerate([100, 100, 300, 50, 50, 10])]
    groups = sample_mix.group_shards(shards, copies_per_token=0.5, part_tokens=100)
    assert [[s["key"] for s in group] for group in groups] == [["s0", "s1"], ["s2"], ["s3", "s4", "s5"]]
    assert sample_mix.group_shards([], copies_per_token=0.5, part_tokens=100) == []


def _write_cell(store, cell, shard_docs):
    root = sample_mix.cell_dir(store, cell)
    paths, stubs, counts = [], [], []
    for shard, docs in enumerate(shard_docs):
        path = f"{root}/part-{shard:05d}-of-{len(shard_docs):05d}"
        ledger = write_bucket_cache(path, iter(docs), [len(d) for d in docs])
        paths.append(path)
        stubs.append(
            CacheLedger(
                total_num_rows=ledger.total_num_rows,
                shard_rows={},
                finished_shards=[],
                field_counts={},
                metadata=CacheMetadata.empty(),
            )
        )
        counts.append({"input_ids": int(ledger.field_counts["input_ids"])})
    _merge_sharded_ledgers(root, paths, stubs, counts, CacheMetadata.empty())


def _docs(rng, n):
    return [
        np.concatenate([rng.integers(10, 1000, size=rng.integers(5, 200)), [EOS]]).astype(np.int32) for _ in range(n)
    ]


# Each item samples in a freshly spawned process that re-imports this module, about 10s apiece.
@pytest.mark.timeout(180)
def test_end_to_end_on_a_small_store(tmp_path):
    rng = np.random.default_rng(0)
    store = str(tmp_path / "store")
    source = {
        "c00q0": [_docs(rng, 400), _docs(rng, 300), _docs(rng, 200)],
        "c01q4": [_docs(rng, 500)],
        "c02q1": [_docs(rng, 60)],  # small: the main cache must repeat its documents
    }
    for cell, shards in source.items():
        _write_cell(store, cell, shards)
    cell_tokens = {cell: sum(len(d) for shard in shards for d in shard) for cell, shards in source.items()}

    recipe = Recipe(
        store=store,
        caches=(
            OutputCache("main", 40_000, {"c00q0": 0.4, "c01q4": 0.3, "small": 0.3}),
            OutputCache("cooldown", 8_000, {"c00q0": 0.2, "c01q4": 0.8}),
        ),
        groups={"small": ("c02q1",)},
    )
    out = str(tmp_path / "out")
    # A part size larger than any cell makes one item per cell, so c00q0's three shards share one part.
    plan = sample_mix.plan_recipe(recipe, "test", out, seed=3, store=None, target_overrides={}, part_tokens=10**9)
    assert plan["cells"]["c02q1"]["bands"]["main"][1] > 1  # repeats
    assert [(item["cell"], len(item["shards"])) for item in plan["items"]] == [("c00q0", 3), ("c01q4", 1), ("c02q1", 1)]

    sample_mix.sample(out, num_tasks=2, task_index=0)
    sample_mix.sample(out, num_tasks=2, task_index=1)
    report = sample_mix.merge(out)
    sample_mix.validate(out, measure_bytes=True)

    for cache, target in plan["caches"].items():
        assert report["caches"][cache]["total_tokens"] == pytest.approx(target, rel=0.15)
    expected_small = plan["cells"]["c02q1"]["expected_tokens"]["main"]
    assert report["cells"]["c02q1"]["main"]["realized"] == pytest.approx(expected_small, rel=0.2)
    assert expected_small > cell_tokens["c02q1"]

    # Every output document is a source document, and caches share none while bands fit in [0, 1).
    source_docs = {tuple(d) for shards in source.values() for shard in shards for d in shard}
    seen = {}
    for cache in plan["caches"]:
        loaded = TreeCache.load(f"{out}/{cache}", _EXEMPLAR)
        docs = [tuple(np.asarray(d["input_ids"])) for d in loaded.get_batch_sync(list(range(len(loaded))))]
        assert set(docs) <= source_docs
        seen[cache] = set(docs)
    big_cell_docs = {tuple(d) for cell in ("c00q0", "c01q4") for shard in source[cell] for d in shard}
    assert not (seen["main"] & seen["cooldown"] & big_cell_docs)

    # Re-running a finished task is a no-op, and the plan is deterministic given the seed.
    sample_mix.sample(out, num_tasks=1, task_index=0)
    replanned = sample_mix.plan_recipe(
        recipe, "test", str(tmp_path / "out2"), seed=3, store=None, target_overrides={}, part_tokens=10**9
    )
    assert json.dumps(replanned["cells"], sort_keys=True) == json.dumps(plan["cells"], sort_keys=True)
