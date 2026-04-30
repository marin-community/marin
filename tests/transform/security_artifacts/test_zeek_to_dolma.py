# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Zeek → Dolma transform."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from marin.transform.security_artifacts.zeek_to_dolma import (
    ZEEK_RENDER_TAG,
    ZeekToDolmaConfig,
    render_file_to_dolma_blocks,
    render_records_to_dolma_blocks,
)

FIELDS = ("ts", "uid", "id.orig_h", "id.orig_p", "proto", "service")
SAMPLE_RECORDS = [
    {
        "ts": "1700000000.000001",
        "uid": "CabcD1",
        "id.orig_h": "10.0.0.1",
        "id.orig_p": 443,
        "proto": "tcp",
        "service": "ssl",
    },
    {
        "ts": "1700000001.000002",
        "uid": "CabcD2",
        "id.orig_h": "10.0.0.2",
        "id.orig_p": 53,
        "proto": "udp",
        "service": "dns",
    },
    {
        "ts": "1700000002.000003",
        "uid": "CabcD3",
        "id.orig_h": "10.0.0.3",
        "id.orig_p": 80,
        "proto": "tcp",
        # Deliberately leave `service` off to exercise missing-field handling.
    },
]


def _base_config(input_path: str, output_path: str, **overrides) -> ZeekToDolmaConfig:
    kwargs = {
        "input_path": input_path,
        "output_path": output_path,
        "zeek_path": "conn",
        "fields": FIELDS,
        "source_label": "test/zeek/conn",
        "input_format": "jsonl",
        "records_per_block": 2,
    }
    kwargs.update(overrides)
    return ZeekToDolmaConfig(**kwargs)


def test_render_records_groups_into_blocks_and_tags_render_mode():
    cfg = _base_config("ignored", "ignored")

    blocks = render_records_to_dolma_blocks(SAMPLE_RECORDS, cfg)

    # 3 records, records_per_block=2 → 2 blocks (2 + 1).
    assert len(blocks) == 2
    assert all(block["render"] == ZEEK_RENDER_TAG for block in blocks)
    assert all(block["source"] == "test/zeek/conn" for block in blocks)
    first_text = blocks[0]["text"]
    assert "#path\tconn" in first_text
    assert "#fields\tts\tuid\tid.orig_h\tid.orig_p\tproto\tservice" in first_text
    # Two body records in block 0.
    body_lines = [line for line in first_text.splitlines() if not line.startswith("#")]
    assert len(body_lines) == 2
    assert "CabcD1" in body_lines[0]
    assert "CabcD2" in body_lines[1]


def test_render_records_handles_missing_fields_as_unset():
    cfg = _base_config("ignored", "ignored", records_per_block=8)

    blocks = render_records_to_dolma_blocks(SAMPLE_RECORDS, cfg)

    assert len(blocks) == 1
    body_lines = [line for line in blocks[0]["text"].splitlines() if not line.startswith("#")]
    # Third record has no `service` field — should serialize as `-`.
    assert body_lines[-1].endswith("\t-"), body_lines[-1]


def test_render_records_is_deterministic_for_stable_ids():
    cfg = _base_config("ignored", "ignored")

    a = render_records_to_dolma_blocks(SAMPLE_RECORDS, cfg)
    b = render_records_to_dolma_blocks(SAMPLE_RECORDS, cfg)

    assert [block["id"] for block in a] == [block["id"] for block in b]
    assert [block["text"] for block in a] == [block["text"] for block in b]


def test_render_records_respects_max_blocks_per_file():
    cfg = _base_config("ignored", "ignored", records_per_block=1, max_blocks_per_file=2)

    blocks = render_records_to_dolma_blocks(SAMPLE_RECORDS, cfg)

    assert len(blocks) == 2


def test_render_file_to_dolma_blocks_reads_jsonl(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_file = input_dir / "conn.jsonl"
    with input_file.open("w", encoding="utf-8") as handle:
        for record in SAMPLE_RECORDS:
            handle.write(json.dumps(record))
            handle.write("\n")

    cfg = _base_config(str(input_dir), str(tmp_path / "output"))

    blocks = render_file_to_dolma_blocks(str(input_file), cfg)

    assert len(blocks) == 2
    assert blocks[0]["render"] == ZEEK_RENDER_TAG
    # Ids embed the block index and a short hash for reproducibility.
    assert blocks[0]["id"].startswith("conn-000000-")
    assert blocks[1]["id"].startswith("conn-000001-")


def test_render_file_to_dolma_blocks_reads_gzipped_jsonl(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_file = input_dir / "conn.jsonl.gz"
    with gzip.open(input_file, "wt", encoding="utf-8") as handle:
        for record in SAMPLE_RECORDS:
            handle.write(json.dumps(record))
            handle.write("\n")

    cfg = _base_config(str(input_dir), str(tmp_path / "output"))

    blocks = render_file_to_dolma_blocks(str(input_file), cfg)

    assert len(blocks) == 2
