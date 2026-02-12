# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax.random as jrandom
import pytest

from levanter.inference.utils import INVALID
from levanter.main.sample_lm_multihost import (
    SampleLmMultihostConfig,
    _build_requests_for_rounds,
    _decode_host_rows_payload,
    _encode_host_rows_payload,
    _host_output_path,
    _merge_gathered_host_rows,
    _merged_host_output_path,
    _write_host_generations_jsonl,
)


class _FakeTokenizer:
    pad_token_id = 0

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token) for token in tokens)


def test_host_output_path_is_deterministic():
    path = _host_output_path("out", process_index=3, process_count=16)
    assert path.as_posix() == "out/host_0003_of_0016.jsonl"


def test_merged_host_output_path_is_deterministic():
    path = _merged_host_output_path("out", process_count=16)
    assert path.as_posix() == "out/all_hosts_merged_of_0016.jsonl"


def test_write_host_generations_jsonl_writes_rows_with_global_prompt_identity(tmp_path):
    config = SampleLmMultihostConfig(max_new_tokens=16, n_generations=1)
    prompt_ids = [[11, 12], [21, 22, 23]]
    local_prompts = ["prompt-8", "prompt-9"]
    requests, request_meta = _build_requests_for_rounds(
        prompt_ids=prompt_ids,
        stop_tokens=None,
        config=config,
        base_key=jrandom.PRNGKey(0),
        rounds=1,
        prompt_id_offset=8,
        total_num_prompts=128,
    )
    result_tokens = [
        [0, 1001, INVALID],
        [1002, 1003, 0],
    ]

    output_path = tmp_path / "host_0001_of_0016.jsonl"
    rows_written = _write_host_generations_jsonl(
        output_path=output_path,
        requests=requests,
        request_meta=request_meta,
        result_tokens=result_tokens,
        local_prompts=local_prompts,
        shard_start=8,
        shard_end=10,
        process_index=1,
        process_count=16,
        n_generations=1,
        tokenizer=_FakeTokenizer(),
    )

    assert rows_written == 2
    parsed_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(parsed_rows) == 2

    assert parsed_rows[0]["global_prompt_index"] == 8
    assert parsed_rows[0]["local_prompt_index"] == 0
    assert parsed_rows[0]["request_id"] == 8
    assert parsed_rows[0]["prompt"] == "prompt-8"
    assert parsed_rows[0]["generated_tokens"] == [1001]
    assert parsed_rows[0]["generated_text"] == "1001"

    assert parsed_rows[1]["global_prompt_index"] == 9
    assert parsed_rows[1]["local_prompt_index"] == 1
    assert parsed_rows[1]["request_id"] == 9
    assert parsed_rows[1]["prompt"] == "prompt-9"
    assert parsed_rows[1]["generated_tokens"] == [1002, 1003]
    assert parsed_rows[1]["generated_text"] == "1002 1003"


def test_write_host_generations_jsonl_rejects_sequence_count_mismatch(tmp_path):
    config = SampleLmMultihostConfig(max_new_tokens=16, n_generations=1)
    requests, request_meta = _build_requests_for_rounds(
        prompt_ids=[[1], [2]],
        stop_tokens=None,
        config=config,
        base_key=jrandom.PRNGKey(0),
        rounds=1,
        prompt_id_offset=4,
        total_num_prompts=32,
    )

    with pytest.raises(RuntimeError, match="Expected 2 generated sequences"):
        _write_host_generations_jsonl(
            output_path=tmp_path / "rows.jsonl",
            requests=requests,
            request_meta=request_meta,
            result_tokens=[],
            local_prompts=["p4", "p5"],
            shard_start=4,
            shard_end=6,
            process_index=0,
            process_count=2,
            n_generations=1,
            tokenizer=_FakeTokenizer(),
        )


def test_host_rows_payload_roundtrip():
    rows = [
        {"process_index": 0, "round_index": 0, "global_prompt_index": 9, "generation_index": 0},
        {"process_index": 0, "round_index": 1, "global_prompt_index": 2, "generation_index": 0},
    ]
    payload = _encode_host_rows_payload(rows)
    decoded = _decode_host_rows_payload(payload)
    assert decoded == rows


def test_merge_gathered_host_rows_orders_globally_and_validates_host_identity():
    gathered_rows = [
        [
            {"process_index": 0, "round_index": 1, "global_prompt_index": 2, "generation_index": 0},
            {"process_index": 0, "round_index": 0, "global_prompt_index": 7, "generation_index": 0},
        ],
        [
            {"process_index": 1, "round_index": 0, "global_prompt_index": 3, "generation_index": 0},
            {"process_index": 1, "round_index": 1, "global_prompt_index": 1, "generation_index": 0},
        ],
    ]

    merged = _merge_gathered_host_rows(gathered_rows, process_count=2)
    assert [(row["round_index"], row["global_prompt_index"]) for row in merged] == [
        (0, 3),
        (0, 7),
        (1, 1),
        (1, 2),
    ]

    bad_rows = [[{"process_index": 1, "round_index": 0, "global_prompt_index": 0, "generation_index": 0}], []]
    with pytest.raises(RuntimeError, match="source mismatch"):
        _merge_gathered_host_rows(bad_rows, process_count=2)
