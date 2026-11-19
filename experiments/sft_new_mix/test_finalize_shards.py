# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
from levanter.store.cache import SerialCacheWriter, TreeCache
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from experiments.sft_new_mix.apply_atlas import bind_disposition, propagate_lexical_duplicates
from experiments.sft_new_mix.exp3_filters import BOS, END_HEADER, EOS, EOT, HEADER, Stage
from experiments.sft_new_mix.exp3_runtime import implementation_identity
from experiments.sft_new_mix.final_receipts import collect_final_inventory
from experiments.sft_new_mix.finalize_shards import FINALIZE_IMPLEMENTATION_FILES, DropStage, transform_shard
from experiments.sft_new_mix.job_manifests import read_job_manifest, stage_job_manifest
from experiments.sft_new_mix.loss_modes import LossMode, prediction_weights


def test_dropped_replacements_are_not_counted_as_selected(tmp_path):
    vocabulary = {character: index for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([BOS, EOS, HEADER, END_HEADER, EOT])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    source = "nemotron_sft_v3/math_v4/train"

    def tokens(name, arguments):
        text = (
            BOS
            + HEADER
            + "user"
            + END_HEADER
            + "\nSolve."
            + EOT
            + HEADER
            + "assistant"
            + END_HEADER
            + "\n<tool_call>"
            + json.dumps({"name": name, "arguments": arguments})
            + "</tool_call>"
            + EOT
            + EOS
        )
        return np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int32)

    prefix = tuple(tokenizer.encode(HEADER + "assistant" + END_HEADER + "\n", add_special_tokens=False).ids)

    def weights(ids):
        return prediction_weights(
            ids,
            loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
            message_start_id=tokenizer.token_to_id(HEADER),
            assistant_prefix=prefix,
            trained_assistant_turns=None,
        )

    original = [tokens("bash", {"command": "pwd"}), tokens("bash", {"command": "pwd"})]
    # The replacement calls an unvetted tool, so the re-filter drops it.
    rejected = tokens("web_search", {"query": "answer"})
    base_path = str(tmp_path / "base")
    exemplar = {"input_ids": np.empty(0, dtype=np.int32), "loss_weights": np.empty(0, dtype=np.float32)}
    with SerialCacheWriter(base_path, exemplar) as writer:
        writer.write_batch([{"input_ids": ids, "loss_weights": weights(ids)} for ids in original])
    decisions = tmp_path / "base-decisions.jsonl"
    decisions.write_text("".join(json.dumps({"row": i, "stage": "retained", "output_row": i}) + "\n" for i in range(2)))
    payload = tmp_path / "replacement.i32"
    payload.write_bytes(rejected.astype("<i4").tobytes())
    original_sha = hashlib.sha256(original[1].astype("<i4").tobytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": [Stage.REAL_TOOLS],
                "think_limits": {},
                "tools_by_source": {source: ["bash"]},
                "give_up_sources": [],
            }
        )
    )
    job = {
        "source": source,
        "loss_mode": "chat_assistant_only",
        "teacher_only": False,
        "passed": False,
        "output_prefix": str(tmp_path / "final"),
        "implementation_sha256": implementation_identity(FINALIZE_IMPLEMENTATION_FILES),
        "tokenizer_uri": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "config_uri": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "overlay_enabled": True,
        "real_tools_enabled": True,
        "drop_rows": {},
        "enabled_drop_stages": [],
        "base_shard": {
            "cache_uri": base_path,
            "rows": 2,
            "tokens": sum(map(len, original)),
            "original_input_path": "pinned-original",
            "decisions_uri": str(decisions),
        },
        "overlays": [
            {
                "source": source,
                "source_shard": "pinned-original",
                "source_row": 1,
                "status": "verified",
                "tool_calls_modified": True,
                "diff": {"modified": True},
                "verified_original_sha256": original_sha,
                "input_ids_sha256": original_sha,
                "payload_uri": str(payload),
                "export_manifest": {"files": {"input_ids.i32": hashlib.sha256(payload.read_bytes()).hexdigest()}},
            }
        ],
    }
    result = transform_shard(read_job_manifest(stage_job_manifest(job)))
    with open(result["receipt_uri"]) as stream:
        receipt = json.load(stream)
    assert receipt["output_rows"] == 1
    assert receipt["verified_modified_rows_selected"] == 0
    cache = TreeCache.load(receipt["output_path"], exemplar)
    actual = asyncio.run(cache.get_batch([0]))
    np.testing.assert_array_equal(actual[0]["input_ids"], original[0])


def test_modified_trace_replaces_one_row_and_actual_calls_control_the_gate(tmp_path):
    vocabulary = {character: index for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([BOS, EOS, HEADER, END_HEADER, EOT])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    source = "nemotron_sft_v3/math_v4/train"

    def tokens(name, arguments):
        # An unused web-search declaration must survive the actual-call gate.
        text = (
            BOS
            + HEADER
            + "system"
            + END_HEADER
            + '\n<tools>{"name":"web_search"}</tools>'
            + EOT
            + HEADER
            + "user"
            + END_HEADER
            + "\nSolve."
            + EOT
            + HEADER
            + "assistant"
            + END_HEADER
            + "\n<tool_call>"
            + json.dumps({"name": name, "arguments": arguments})
            + "</tool_call>"
            + EOT
            + EOS
        )
        return np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int32)

    prefix = tuple(tokenizer.encode(HEADER + "assistant" + END_HEADER + "\n", add_special_tokens=False).ids)

    def weights(ids):
        return prediction_weights(
            ids,
            loss_mode=LossMode.CHAT_ASSISTANT_ONLY,
            message_start_id=tokenizer.token_to_id(HEADER),
            assistant_prefix=prefix,
            trained_assistant_turns=None,
        )

    original = [
        tokens("bash", {"command": "pwd"}),
        tokens("stateful_python_code_exec", {"code": "print(2)"}),
        tokens("web_search", {"query": "answer"}),
        tokens("bash", {"command": "python3 -c 'print(2)'"}),
    ]
    replacement = tokens("bash", {"command": "python3 -c 'print(2)'"})
    base_path = str(tmp_path / "base")
    exemplar = {"input_ids": np.empty(0, dtype=np.int32), "loss_weights": np.empty(0, dtype=np.float32)}
    with SerialCacheWriter(base_path, exemplar) as writer:
        writer.write_batch([{"input_ids": ids, "loss_weights": weights(ids)} for ids in original])
    decisions = tmp_path / "base-decisions.jsonl"
    decisions.write_text("".join(json.dumps({"row": i, "stage": "retained", "output_row": i}) + "\n" for i in range(4)))
    payload = tmp_path / "replacement.i32"
    payload.write_bytes(replacement.astype("<i4").tobytes())
    original_sha = hashlib.sha256(original[1].astype("<i4").tobytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": [Stage.REAL_TOOLS],
                "think_limits": {},
                "tools_by_source": {source: ["bash"]},
                "give_up_sources": [],
            }
        )
    )
    job = {
        "source": source,
        "loss_mode": "chat_assistant_only",
        "teacher_only": False,
        "passed": False,
        "output_prefix": str(tmp_path / "final"),
        "implementation_sha256": implementation_identity(FINALIZE_IMPLEMENTATION_FILES),
        "tokenizer_uri": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "config_uri": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "overlay_enabled": True,
        "real_tools_enabled": True,
        "drop_rows": {},
        "enabled_drop_stages": [],
        "base_shard": {
            "cache_uri": base_path,
            "rows": 4,
            "tokens": sum(map(len, original)),
            "original_input_path": "pinned-original",
            "decisions_uri": str(decisions),
        },
        "overlays": [
            {
                "source": source,
                "source_shard": "pinned-original",
                "source_row": 1,
                "status": "verified",
                "tool_calls_modified": True,
                "diff": {"modified": True},
                "verified_original_sha256": original_sha,
                "input_ids_sha256": original_sha,
                "payload_uri": str(payload),
                "export_manifest": {"files": {"input_ids.i32": hashlib.sha256(payload.read_bytes()).hexdigest()}},
            }
        ],
    }
    result = transform_shard(read_job_manifest(stage_job_manifest(job)))
    with open(result["receipt_uri"]) as stream:
        receipt = json.load(stream)
    assert receipt["output_rows"] == 3
    assert receipt["verified_modified_rows_selected"] == 1
    assert receipt["stats"][Stage.REAL_TOOLS]["drop_reason:unvetted_tools:web_search"] == 1
    cache = TreeCache.load(receipt["output_path"], exemplar)
    actual = asyncio.run(cache.get_batch([0, 1, 2]))
    np.testing.assert_array_equal(actual[0]["input_ids"], original[0])
    np.testing.assert_array_equal(actual[0]["loss_weights"], weights(original[0]))
    np.testing.assert_array_equal(actual[1]["input_ids"], replacement)
    np.testing.assert_array_equal(actual[1]["loss_weights"], weights(replacement))
    np.testing.assert_array_equal(actual[2]["input_ids"], replacement)

    # A pinned Atlas disposition can be rolled back independently of the call gate.
    cache_prefix = hashlib.sha256(receipt["output_path"].encode()).hexdigest()
    drop_rows = {cache_prefix: {}}
    bind_disposition(
        cache_prefix + ":000000000001",
        DropStage.LEXICAL,
        "matched_pinned_eval",
        {cache_prefix: {"rows": 3}},
        drop_rows,
    )
    bind_disposition(
        cache_prefix + ":000000000002",
        DropStage.EXACT_DUPLICATE,
        "exact_duplicate",
        {cache_prefix: {"rows": 3}},
        drop_rows,
    )
    propagate_lexical_duplicates(
        {cache_prefix + ":000000000002": cache_prefix + ":000000000001"},
        {cache_prefix + ":000000000001"},
        {cache_prefix: {"rows": 3}},
        drop_rows,
    )
    atlas_job = {
        **job,
        "overlays": [],
        "overlay_enabled": False,
        "real_tools_enabled": False,
        "base_shard": {
            **job["base_shard"],
            "cache_uri": receipt["output_path"],
            "rows": 3,
            "tokens": len(original[0]) + 2 * len(replacement),
            "receipt_uri": result["receipt_uri"],
            "receipt_sha256": hashlib.sha256(Path(result["receipt_uri"]).read_bytes()).hexdigest(),
        },
        "drop_rows": drop_rows[cache_prefix],
    }
    disabled = transform_shard(atlas_job)
    with open(disabled["receipt_uri"]) as stream:
        assert json.load(stream)["output_rows"] == 3
    final_job = {**atlas_job, "enabled_drop_stages": ["atlas_lexical"]}
    enabled = transform_shard(final_job)
    with open(enabled["receipt_uri"]) as stream:
        final_receipt = json.load(stream)
    assert final_receipt["output_rows"] == 1
    assert final_receipt["stats"]["atlas_lexical"]["input_documents"] == 3
    assert final_receipt["stats"]["atlas_lexical"]["output_documents"] == 1
    both = transform_shard({**atlas_job, "enabled_drop_stages": ["atlas_exact_dedup", "atlas_lexical"]})
    assert json.loads(Path(both["receipt_uri"]).read_text())["output_rows"] == 1
    dedup_only = transform_shard({**atlas_job, "enabled_drop_stages": ["atlas_exact_dedup"]})
    assert json.loads(Path(dedup_only["receipt_uri"]).read_text())["output_rows"] == 2
    inventory = collect_final_inventory(
        {"jobs": [final_job], "scope": "fixture-complete"},
        {
            source: {
                "loss_mode": "chat_assistant_only",
                "teacher_only": False,
                "source_revision": "frozen-test",
                "loss_weights_key": "loss_weights",
                "loss_weights_indexing": "prediction",
            }
        },
        workers=1,
    )
    retained = inventory["sources"][0]
    assert inventory["complete"] is True
    assert retained["shards"][0]["cache_uri"] == final_receipt["output_path"]
    assert retained["counts"]["output_tokens"] == len(original[0])
    assert retained["counts"]["supervised_prediction_positions"] == int(weights(original[0]).sum())
