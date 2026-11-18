# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the option-G comp-prog label rescue on real pipeline rows."""

import asyncio
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from levanter.store.cache import SerialCacheWriter, TreeCache
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

from experiments.sft_new_mix.exp3_build import build_shard
from experiments.sft_new_mix.exp3_filters import BOS, EOS, CompProgLabels, Document, Stage, apply_stage
from experiments.sft_new_mix.exp3_runtime import BUILD_IMPLEMENTATION_FILES, build_config, implementation_identity

CPP = "nemotron_sft_v3/competitive_programming_v2/competitive_coding_cpp_dd8b7ae4"
THINK_LIMIT = 50_000

SPECIALS = (
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|start_think|>",
    "<|end_think|>",
)


def fixture_tokenizer(path) -> str:
    """A byte-level tokenizer that round-trips native transcripts exactly."""
    vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(token, special=True, normalized=False) for token in SPECIALS])
    tokenizer.save(str(path))
    return str(path)


def transcript(user: str, thought: str, answer: str) -> str:
    return (
        f"{BOS}<|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n<|start_think|>{thought}<|end_think|>{answer}<|eot_id|>{EOS}"
    )


def label_records(cache_part: str = "s3://fixture/train/part-00000-of-00001") -> list[dict]:
    """One judged row per cache row; problem P2 keeps only its shortest accepted row."""

    def record(uuid: str, problem: str, rating: int, status: str, give_up: bool, think: int, row: int) -> dict:
        return {
            "lang": "cpp",
            "uuid": uuid,
            "problem": problem,
            "rating": rating,
            "status": status,
            "give_up": give_up,
            "think_tokens": think,
            "cache_source": CPP,
            "cache_part": cache_part,
            "cache_row": row,
        }

    return [
        record("uuid-accepted", "1846A", 800, "accepted", False, 10, 0),
        record("uuid-failed", "1846B", 800, "failed", False, 5, 1),
        record("uuid-give-up", "1846C", 1600, "accepted", True, 30, 2),
        record("uuid-rescued", "1846C", 1600, "accepted", True, 20, 3),
    ]


def label_config(tmp_path, records=None, *, enabled=None):
    records = label_records() if records is None else records
    table = tmp_path / "labels.jsonl.gz"
    table.write_bytes(gzip.compress("\n".join(json.dumps(record) for record in records).encode()))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "enabled": (
                    enabled if enabled is not None else ["comp_prog_label_rescue", "give_up_endings", "think_length"]
                ),
                "think_limits": {CPP: THINK_LIMIT},
                "tools_by_source": {},
                "give_up_sources": [CPP],
                "junkgate_enabled": False,
                "comp_prog_labels": {
                    "table_uri": str(table),
                    "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                },
            }
        )
    )
    return str(config)


def filter_config(config_path: str):
    return build_config(config_path, hashlib.sha256(Path(config_path).read_bytes()).hexdigest())[1]


def stage_apply(tmp_path, text: str, stage: Stage, uuid: str, records=None):
    config = filter_config(label_config(tmp_path, records))
    return apply_stage(Document(CPP, "fixture", text, False, uuid), stage, config, len)


def test_judged_verdicts_replace_the_current_filters(tmp_path):
    accepted = transcript("Solve A", "Reasoning.", "print(1)")
    failed = transcript("Solve B", "Reasoning.", "print(2)")
    kept = stage_apply(tmp_path, accepted, Stage.COMP_PROG_LABEL_RESCUE, "uuid-accepted")
    assert kept.keep and kept.reason == ""
    dropped = stage_apply(tmp_path, failed, Stage.COMP_PROG_LABEL_RESCUE, "uuid-failed")
    assert not dropped.keep and dropped.reason == "label_failed"


def test_judged_rows_skip_the_think_length_prior(tmp_path):
    long_thought = transcript("Solve A", "Reasoning. " * (THINK_LIMIT + 10), "print(1)")
    # Without labels the same row is dropped for thought length.
    plain = Document(CPP, "fixture", long_thought, False, "")
    config = filter_config(label_config(tmp_path))
    assert not apply_stage(plain, Stage.THINK_LENGTH, config, len).keep
    # With a pinned accepted verdict the row keeps the official-test result.
    kept = stage_apply(tmp_path, long_thought, Stage.COMP_PROG_LABEL_RESCUE, "uuid-accepted")
    assert kept.keep
    skipped = stage_apply(tmp_path, long_thought, Stage.THINK_LENGTH, "uuid-accepted")
    assert skipped.keep and skipped.reason == ""


def test_give_up_still_drops_judged_rows(tmp_path):
    give_up = transcript("Solve C", "Hmm. I give up.", "print('guess')")
    dropped = stage_apply(tmp_path, give_up, Stage.GIVE_UP, "uuid-give-up")
    assert not dropped.keep and dropped.reason == "give_up_pattern"


def test_rescued_shortest_passing_row_survives_give_up(tmp_path):
    give_up = transcript("Solve C", "Hmm. I give up.", "print('guess')")
    kept = stage_apply(tmp_path, give_up, Stage.GIVE_UP, "uuid-rescued")
    assert kept.keep and kept.reason == ""
    labels = CompProgLabels.from_records(label_records())
    # 1846C has two accepted rows and both hold give-up phrases; the shortest is rescued.
    assert labels.rescued == frozenset({"uuid-rescued"})


def test_unjudged_rows_keep_the_current_filters(tmp_path):
    give_up = transcript("Solve D", "Hmm. I give up.", "print('guess')")
    dropped = stage_apply(tmp_path, give_up, Stage.GIVE_UP, "")
    assert not dropped.keep and dropped.reason == "give_up_pattern"
    long_thought = transcript("Solve E", "Reasoning. " * (THINK_LIMIT + 10), "print(1)")
    over_limit = stage_apply(tmp_path, long_thought, Stage.THINK_LENGTH, "")
    assert not over_limit.keep and over_limit.reason == "over_think_limit"


def test_label_stage_requires_the_pinned_table(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "enabled": ["comp_prog_label_rescue"],
                "think_limits": {},
                "tools_by_source": {},
                "give_up_sources": [],
                "junkgate_enabled": False,
            }
        )
    )
    with pytest.raises(ValueError, match="must be set together"):
        build_config(str(config_path), hashlib.sha256(config_path.read_bytes()).hexdigest())


def build_fixture_shard(tmp_path, texts: list[str], config_path: str, tokenizer_path: str) -> dict:
    tokenizer_sha = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
    config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    rows = [np.asarray(tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int32) for text in texts]
    cache_path = str(tmp_path / "original")
    with SerialCacheWriter(cache_path, {"input_ids": np.empty(0, np.int32)}) as writer:
        writer.write_batch([{"input_ids": row} for row in rows])
    return {
        "source": CPP,
        "input_path": cache_path,
        "row_start": 0,
        "row_stop": len(rows),
        "expected_rows": len(rows),
        "expected_tokens": sum(len(row) for row in rows),
        "loss_mode": "chat_assistant_only",
        "source_revision": "fixture-pin",
        "passed": False,
        "pass_receipt_sha256": "",
        "output_prefix": str(tmp_path / "candidate"),
        "config_uri": config_path,
        "config_sha256": config_sha,
        "implementation_sha256": implementation_identity(BUILD_IMPLEMENTATION_FILES),
        "tokenizer_uri": tokenizer_path,
        "tokenizer_sha256": tokenizer_sha,
    }


def shard_decisions(tmp_path, job: dict) -> list[dict]:
    identity = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    decisions = (
        Path(tmp_path)
        / "candidate"
        / "components"
        / hashlib.sha256(CPP.encode()).hexdigest()
        / "shards"
        / identity
        / "decisions.jsonl"
    )
    return [json.loads(line) for line in decisions.read_text().splitlines()]


def test_build_applies_option_g_end_to_end(tmp_path):
    """The real shard pipeline keeps passing rows, drops failures and rescues the shortest passing row."""
    tokenizer_path = fixture_tokenizer(tmp_path / "tokenizer.json")
    cache_part = str(tmp_path / "original")
    config = label_config(tmp_path, label_records(cache_part))
    texts = [
        transcript("Solve A", "Reasoning.", "print(1)"),  # judged accepted, kept
        transcript("Solve B", "Reasoning.", "print(2)"),  # judged failed, dropped
        transcript("Solve C", "Hmm. I give up.", "print('no')"),  # judged give-up row, dropped
        transcript("Solve C", "Hmm. I give up. Short.", "print('yes')"),  # rescued shortest passing row
        transcript("Solve D", "Reasoning. " * 6_000, "print(1)"),  # unjudged, over the think limit
        transcript("Solve E", "Reasoning.", "print(1)"),  # unjudged, kept
    ]
    job = build_fixture_shard(tmp_path, texts, str(config), tokenizer_path)
    receipt = build_shard(job)
    decisions = shard_decisions(tmp_path, job)
    by_row = {decision["row"]: decision for decision in decisions}
    assert by_row[1]["stage"] == "comp_prog_label_rescue" and by_row[1]["reason"] == "label_failed"
    assert by_row[2]["stage"] == "give_up_endings" and by_row[2]["reason"] == "give_up_pattern"
    assert by_row[3]["stage"] == "retained"
    assert by_row[4]["stage"] == "think_length" and by_row[4]["reason"] == "over_think_limit"
    retained = [decision["row"] for decision in decisions if decision["stage"] == "retained"]
    assert retained == [0, 3, 5]
    assert receipt["output_rows"] == 3
    cache = TreeCache.load(
        receipt["output_path"], {"input_ids": np.empty(0, np.int32), "loss_weights": np.empty(0, np.float32)}
    )
    rows = asyncio.run(cache.get_batch([0, 1, 2]))
    tokenizer = Tokenizer.from_file(tokenizer_path)
    decoded = [tokenizer.decode(row["input_ids"].tolist(), skip_special_tokens=False) for row in rows]
    assert decoded == [texts[0], texts[3], texts[5]]
    for row in rows:
        supervised = row["loss_weights"].astype(bool)
        assert supervised.sum() > 0
        # Assistant-only masking keeps supervision inside the assistant turn.
        assert not supervised[0]
        text = tokenizer.decode(row["input_ids"].tolist(), skip_special_tokens=False)
        assert "print(" in text
