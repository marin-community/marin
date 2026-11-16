# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import numpy as np
import pytest
from levanter.store.cache import CacheCatalog, SerialCacheWriter, TreeCache

from experiments.sft_new_mix.assemble_candidate import combine_source_inventories, correct_replay_inventory
from experiments.sft_new_mix.candidate_receipts import catalog_from_inventory


def test_catalog_keeps_sources_and_prediction_masks_separate_under_one_ancestor(tmp_path):
    exemplar = {"input_ids": np.empty(0, dtype=np.int32), "loss_weights": np.empty(0, dtype=np.float32)}
    sources = []
    expected = {}
    for name, mode, tokens, weights in (
        ("chat", "chat_assistant_only", [11, 12, 13], [0, 1, 0]),
        ("replay", "full_token", [21, 22, 23, 24], [1, 1, 1, 0]),
    ):
        path = str(tmp_path / "build" / name / "shards" / "a" / "cache")
        row = {"input_ids": np.asarray(tokens, dtype=np.int32), "loss_weights": np.asarray(weights, dtype=np.float32)}
        with SerialCacheWriter(path, exemplar) as writer:
            writer.write_batch([row])
        expected[name] = row
        sources.append(
            {
                "source": "replay/fixture" if mode == "full_token" else name,
                "component": name,
                "complete": True,
                "loss_mode": mode,
                "loss_weights_key": "loss_weights",
                "loss_weights_indexing": "prediction",
                "teacher_only": False,
                "source_revision": "pin",
                "counts": {
                    "input_documents": 2 if mode == "full_token" else 1,
                    "input_tokens": len(tokens) + (2 if mode == "full_token" else 0),
                    "output_documents": 1,
                    "output_tokens": len(tokens),
                    "supervised_prediction_positions": int(sum(weights)),
                },
                "shards": [{"cache_uri": path, "rows": 1, "tokens": len(tokens)}],
            }
        )
    base = {
        "complete": True,
        "sources": sources,
        "transforms": [],
        "source_replacements": [],
        "tokenizer_uri": "fixture",
        "tokenizer_sha256": "fixture",
    }
    with pytest.raises(ValueError, match="missing"):
        combine_source_inventories(base, [], {}, {})
    inventory = combine_source_inventories(
        base, [{"complete": True, "tokenizer_sha256": "fixture", "sources": [sources[0]], "transforms": []}], {}, {}
    )
    catalog, sidecar = catalog_from_inventory(inventory, str(tmp_path))
    restored = CacheCatalog.from_json(catalog.to_json())
    for name, row in expected.items():
        entry = restored.entry("train", name)
        cache = TreeCache.load_from_ledger(entry.cache_dir, exemplar, entry.ledger)
        actual = asyncio.run(cache.get_batch([0]))[0]
        np.testing.assert_array_equal(actual["input_ids"], row["input_ids"])
        np.testing.assert_array_equal(actual["loss_weights"], row["loss_weights"])
    assert sidecar["sources"]["chat"]["loss_mode"] == "chat_assistant_only"
    assert sidecar["sources"]["replay"]["loss_mode"] == "full_token"
    recovered_path = str(tmp_path / "recovered" / "cache")
    with SerialCacheWriter(recovered_path, exemplar) as writer:
        writer.write_batch(
            [{"input_ids": np.asarray([31, 32], np.int32), "loss_weights": np.asarray([1, 0], np.float32)}]
        )
    previous = inventory["sources"][1]
    corrected = {
        "complete": True,
        "tokenizer_sha256": "fixture",
        "transforms": [],
        "replay_restoration_recipe": {"uri": "fixture", "sha256": "fixture"},
        "replay_line_length_rejection_enabled": False,
        "sources": [
            {
                **previous,
                "counts": {
                    **previous["counts"],
                    "output_documents": 2,
                    "output_tokens": 6,
                    "supervised_prediction_positions": 4,
                },
                "shards": previous["shards"] + [{"cache_uri": recovered_path, "rows": 1, "tokens": 2}],
            }
        ],
    }
    final = correct_replay_inventory(inventory, corrected)
    corrected_catalog, _ = catalog_from_inventory(final, str(tmp_path))
    entry = corrected_catalog.entry("train", "replay")
    cache = TreeCache.load_from_ledger(entry.cache_dir, exemplar, entry.ledger)
    rows = asyncio.run(cache.get_batch([0, 1]))
    assert [row["input_ids"].tolist() for row in rows] == [[21, 22, 23, 24], [31, 32]]
    assert [row["loss_weights"].tolist() for row in rows] == [[1, 1, 1, 0], [1, 0]]
    assert final["sources"][0] == inventory["sources"][0]
