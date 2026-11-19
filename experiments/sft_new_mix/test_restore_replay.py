# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json

import numpy as np
from levanter.store.cache import SerialCacheWriter, TreeCache
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from experiments.sft_new_mix.candidate_receipts import collect_inventory
from experiments.sft_new_mix.exp3_filters import Stage
from experiments.sft_new_mix.restore_replay import (
    merge_restoration,
    restoration_identity,
    restoration_path,
    restore_shard,
)


def test_restoration_keeps_existing_rows_and_recovers_only_line_rejections(tmp_path):
    vocabulary = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    texts = [
        "An existing short document.",
        "A field researcher returned from the mountain valley with several notebooks and a box of photographs. "
        "During the first week, heavy rain had forced the group to change its route and shelter near an abandoned mill. "
        "The observations described unusual plants growing beside a stream, local farming practices, and the remains "
        "of a wooden bridge. Later visits helped establish which changes followed seasonal flooding and which had "
        "developed over many years. The final account compared those records with maps and interviews, explaining "
        "where the evidence remained incomplete and what another expedition should investigate.",
    ]
    ids = [np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int32) for text in texts]
    original_path = str(tmp_path / "original")
    with SerialCacheWriter(original_path, {"input_ids": np.empty(0, np.int32)}) as writer:
        writer.write_batch([{"input_ids": row} for row in ids])
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": [Stage.SOURCE_EXCLUSIONS, Stage.REPLAY_HEURISTICS, Stage.REPLAY_EMAILS, Stage.CONTEXT_LIMIT],
                "think_limits": {},
                "tools_by_source": {},
                "give_up_sources": [],
                "junkgate_enabled": False,
            }
        )
    )
    source = "replay/fixture/lte_64k"
    original_job = {
        "source": source,
        "input_path": original_path,
        "expected_rows": 2,
        "expected_tokens": sum(map(len, ids)),
        "row_start": 0,
        "row_stop": 2,
        "loss_mode": "full_token",
        "source_revision": "control-pin",
        "passed": False,
        "pass_receipt_sha256": "",
        "output_prefix": str(tmp_path / "old"),
        "config_uri": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "implementation_sha256": "historical-code",
        "tokenizer_uri": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
    }
    old_receipt = tmp_path / "old-receipt.json"
    old_receipt.write_text(
        json.dumps(
            {
                "job": original_job,
                "complete": True,
                "stats": {Stage.REPLAY_HEURISTICS: {"drop_reason:mean_line_length": 1}},
            }
        )
    )
    (tmp_path / "decisions.jsonl").write_text(
        json.dumps({"row": 0, "stage": "retained", "output_row": 0})
        + "\n"
        + json.dumps({"row": 1, "stage": Stage.REPLAY_HEURISTICS, "reason": "mean_line_length"})
        + "\n"
    )
    job = {
        "base_job": original_job,
        "base_receipt": {"uri": str(old_receipt), "sha256": hashlib.sha256(old_receipt.read_bytes()).hexdigest()},
        "config": {"uri": str(config_path), "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()},
        "output_prefix": str(tmp_path / "restored"),
        "implementation_sha256": restoration_identity(),
        "source_row_start": 0,
        "source_row_stop": 2,
    }
    restore_shard(job)
    receipt = json.loads((restoration_path(job) / "receipt.json").read_text())
    assert receipt["selected_original_rows"] == 1
    loss = {
        "loss_mode": "full_token",
        "source_revision": "control-pin",
        "teacher_only": False,
        "loss_weights_key": "loss_weights",
        "loss_weights_indexing": "prediction",
    }
    restored = collect_inventory(
        {"jobs": [receipt["candidate_job"]], "source_loss": {source: loss}, "source_replacements": []}, workers=1
    )
    base = {
        "complete": True,
        "sources": [
            {
                "source": source,
                "component": "fixture",
                **loss,
                "shards": [{"cache_uri": "old-survivor"}],
                "counts": {
                    "input_documents": 2,
                    "input_tokens": sum(map(len, ids)),
                    "output_documents": 1,
                    "output_tokens": len(ids[0]),
                    "supervised_prediction_positions": len(ids[0]) - 1,
                },
            }
        ],
        "transforms": [
            {
                "source": source,
                "transform": Stage.REPLAY_HEURISTICS,
                "loss_mode": "full_token",
                "input_documents": 2,
                "input_tokens": sum(map(len, ids)),
                "output_documents": 1,
                "output_tokens": len(ids[0]),
                "dropped_documents": 1,
                "dropped_tokens": len(ids[1]),
                "drop_reason:mean_line_length": 1,
            }
        ],
    }
    merged = merge_restoration(base, restored)
    counts = merged["sources"][0]["counts"]
    assert counts["input_documents"] == counts["output_documents"] == 2
    assert counts["output_tokens"] == sum(map(len, ids))
    assert merged["sources"][0]["shards"][0] == {"cache_uri": "old-survivor"}
    assert merged["transforms"][0]["dropped_documents"] == 0
    assert merged["transforms"][0]["drop_reason:mean_line_length"] == 0
    cache = TreeCache.load(
        restored["sources"][0]["shards"][0]["cache_uri"],
        {"input_ids": np.empty(0, np.int32), "loss_weights": np.empty(0, np.float32)},
    )
    recovered = asyncio.run(cache.get_batch([0]))[0]
    np.testing.assert_array_equal(recovered["input_ids"], ids[1])
    np.testing.assert_array_equal(recovered["loss_weights"], np.r_[np.ones(len(ids[1]) - 1), 0])
