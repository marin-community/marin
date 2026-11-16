# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json

import numpy as np
from levanter.store.cache import SerialCacheWriter, TreeCache
from tokenizers import Tokenizer, models, pre_tokenizers

from experiments.sft_new_mix.exp3_build import build_shard, implementation_identity


def test_range_build_preserves_global_row_ids_and_every_prediction_weight(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "enabled": ["context_limit"],
                "think_limits": {},
                "tools_by_source": {},
                "give_up_sources": [],
                "junkgate_enabled": False,
            }
        )
    )
    path = str(tmp_path / "original")
    originals = [np.ones(length, dtype=np.int32) for length in range(2, 10)]
    with SerialCacheWriter(path, {"input_ids": np.empty(0, np.int32)}) as writer:
        writer.write_batch([{"input_ids": row} for row in originals])
    common = {
        "source": "replay/fixture/lte_64k",
        "input_path": path,
        "loss_mode": "full_token",
        "source_revision": "original-pin",
        "passed": False,
        "pass_receipt_sha256": "",
        "output_prefix": str(tmp_path / "candidate"),
        "config_uri": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "implementation_sha256": implementation_identity(),
        "tokenizer_uri": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }
    actual = []
    for start, stop in ((0, 3), (3, 8)):
        receipt = build_shard(
            {
                **common,
                "row_start": start,
                "row_stop": stop,
                "expected_rows": stop - start,
                "expected_tokens": sum(map(len, originals[start:stop])),
            }
        )
        cache = TreeCache.load(
            receipt["output_path"], {"input_ids": np.empty(0, np.int32), "loss_weights": np.empty(0, np.float32)}
        )
        actual.extend(asyncio.run(cache.get_batch(list(range(stop - start)))))
        identity = hashlib.sha256(json.dumps(receipt["job"], sort_keys=True).encode()).hexdigest()
        decisions = (
            tmp_path
            / "candidate"
            / "components"
            / hashlib.sha256(common["source"].encode()).hexdigest()
            / "shards"
            / identity
            / "decisions.jsonl"
        )
        retained = [
            json.loads(line)["row"]
            for line in decisions.read_text().splitlines()
            if json.loads(line)["stage"] == "retained"
        ]
        assert retained == list(range(start, stop))
    for original, row in zip(originals, actual, strict=True):
        np.testing.assert_array_equal(row["input_ids"], original)
        np.testing.assert_array_equal(row["loss_weights"], np.r_[np.ones(len(original) - 1), 0])
