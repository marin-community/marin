# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json

import numpy as np
import pytest
from levanter.store.cache import SerialCacheWriter, TreeCache
from tokenizers import Tokenizer, models, pre_tokenizers

from experiments.sft_new_mix import exp3_build
from experiments.sft_new_mix.exp3_build import build_shard
from experiments.sft_new_mix.exp3_runtime import BUILD_IMPLEMENTATION_FILES, implementation_identity
from experiments.sft_new_mix.score_replay import MODEL_FILES


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
        "implementation_sha256": implementation_identity(BUILD_IMPLEMENTATION_FILES),
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


def build_fixture(tmp_path, enabled=("context_limit",), **config_extra):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "enabled": list(enabled),
                "think_limits": {},
                "tools_by_source": {},
                "give_up_sources": [],
                "junkgate_enabled": False,
                **config_extra,
            }
        )
    )
    path = str(tmp_path / "original")
    originals = [np.ones(length, dtype=np.int32) for length in range(2, 6)]
    with SerialCacheWriter(path, {"input_ids": np.empty(0, np.int32)}) as writer:
        writer.write_batch([{"input_ids": row} for row in originals])
    return originals, {
        "source": "replay/fixture/lte_64k",
        "input_path": path,
        "row_start": 0,
        "row_stop": len(originals),
        "expected_rows": len(originals),
        "expected_tokens": sum(map(len, originals)),
        "loss_mode": "full_token",
        "source_revision": "original-pin",
        "passed": False,
        "pass_receipt_sha256": "",
        "output_prefix": str(tmp_path / "candidate"),
        "config_uri": str(config),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "implementation_sha256": implementation_identity(BUILD_IMPLEMENTATION_FILES),
        "tokenizer_uri": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }


def shard_root(tmp_path, job):
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    return (
        tmp_path / "candidate" / "components" / hashlib.sha256(job["source"].encode()).hexdigest() / "shards" / identity
    )


def test_preempted_cache_without_receipt_is_deleted_and_rebuilt(tmp_path):
    originals, job = build_fixture(tmp_path)
    first = build_shard(job)
    (shard_root(tmp_path, job) / "receipt.json").unlink()
    second = build_shard(job)
    assert second["output_rows"] == first["output_rows"] == len(originals)
    assert second["output_tokens_and_weights_sha256"] == first["output_tokens_and_weights_sha256"]
    cache = TreeCache.load(
        second["output_path"], {"input_ids": np.empty(0, np.int32), "loss_weights": np.empty(0, np.float32)}
    )
    for original, row in zip(originals, asyncio.run(cache.get_batch(list(range(len(originals))))), strict=True):
        np.testing.assert_array_equal(row["input_ids"], original)


def test_junkgate_digest_mismatch_refuses_to_score(tmp_path, monkeypatch):
    model_dir = tmp_path / "junkgate-model"
    model_dir.mkdir()
    for name in MODEL_FILES:
        (model_dir / name).write_bytes(b"stale weights")
    _, job = build_fixture(
        tmp_path,
        junkgate_enabled=True,
        junkgate_cutoff=0.5,
        junkgate_model_dir=str(model_dir),
        junkgate_model_sha256={name: "0" * 64 for name in MODEL_FILES},
    )
    monkeypatch.setattr(exp3_build, "MODEL_DIR", str(model_dir))
    with pytest.raises(ValueError, match="Junkgate model weights changed after cutoff calibration"):
        build_shard(job)
