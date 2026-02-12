# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax.random as jrandom
import pytest

from levanter.inference.engine import InferenceEngineConfig
from levanter.main.sample_lm_multihost import (
    SampleLmMultihostConfig,
    _build_requests_for_rounds,
    _derive_host_local_engine_config,
    _prompt_shard_bounds,
    _shard_prompts_for_host,
)


def test_prompt_shard_bounds_partition_all_prompts_without_gaps_or_overlap():
    num_prompts = 128
    process_count = 16

    prev_end = 0
    total_local = 0
    for process_index in range(process_count):
        start, end = _prompt_shard_bounds(num_prompts, process_index, process_count)
        assert start == prev_end
        assert 0 <= start <= end <= num_prompts
        prev_end = end
        total_local += end - start

    assert prev_end == num_prompts
    assert total_local == num_prompts


@pytest.mark.parametrize(
    "num_prompts,process_count",
    [
        (0, 1),
        (1, 1),
        (10, 3),
        (57, 2),
        (124, 8),
    ],
)
def test_prompt_shard_bounds_cover_range_for_varied_shapes(num_prompts: int, process_count: int):
    starts_ends = [_prompt_shard_bounds(num_prompts, i, process_count) for i in range(process_count)]

    assert starts_ends[0][0] == 0
    assert starts_ends[-1][1] == num_prompts

    for i in range(1, process_count):
        assert starts_ends[i - 1][1] == starts_ends[i][0]


def test_shard_prompts_for_host_returns_expected_contiguous_slice():
    prompts = [f"prompt-{i}" for i in range(10)]
    prompt_ids = [[i] for i in range(10)]

    local_prompts, local_prompt_ids, start, end = _shard_prompts_for_host(
        prompts,
        prompt_ids,
        process_index=2,
        process_count=3,
    )

    assert (start, end) == (6, 10)
    assert local_prompts == prompts[6:10]
    assert local_prompt_ids == prompt_ids[6:10]


def test_shard_prompts_for_host_rejects_prompt_length_mismatch():
    prompts = ["a", "b"]
    prompt_ids = [[1]]

    with pytest.raises(ValueError, match="length mismatch"):
        _shard_prompts_for_host(prompts, prompt_ids, process_index=0, process_count=1)


def test_build_requests_uses_global_prompt_ids_for_sharded_inputs():
    config = SampleLmMultihostConfig(max_new_tokens=32)
    prompt_ids = [[10, 11], [20, 21, 22]]

    requests, request_meta = _build_requests_for_rounds(
        prompt_ids=prompt_ids,
        stop_tokens=None,
        config=config,
        base_key=jrandom.PRNGKey(0),
        rounds=2,
        round_offset=0,
        prompt_id_offset=3,
        total_num_prompts=10,
    )

    assert [request.request_id for request in requests] == [3, 4, 13, 14]
    assert request_meta == [(0, 3), (0, 4), (1, 3), (1, 4)]


def test_build_requests_default_id_behavior_unchanged_without_offsets():
    config = SampleLmMultihostConfig(max_new_tokens=32)
    prompt_ids = [[1], [2], [3]]

    requests, request_meta = _build_requests_for_rounds(
        prompt_ids=prompt_ids,
        stop_tokens=None,
        config=config,
        base_key=jrandom.PRNGKey(0),
        rounds=1,
    )

    assert [request.request_id for request in requests] == [0, 1, 2]
    assert request_meta == [(0, 0), (0, 1), (0, 2)]


def test_derive_host_local_engine_config_scales_sequence_limits_and_preserves_tunable_budgets():
    base_engine = InferenceEngineConfig(
        max_seq_len=2560,
        max_pages=2304,
        page_size=128,
        max_seqs=128,
        max_seqs_in_prefill=128,
        max_prefill_size=2048,
        max_queued_tokens=128,
        max_tokens_per_round=128,
        max_rounds=64,
        max_stop_seqs=1,
        max_stop_tokens=8,
    )
    config = SampleLmMultihostConfig(
        max_new_tokens=2048,
        n_generations=1,
        engine=base_engine,
    )
    local_prompt_ids = [[101 + i] * 10 for i in range(64)]
    local_devices = ["d0", "d1", "d2", "d3"]

    local_engine = _derive_host_local_engine_config(
        config=config,
        local_prompt_ids=local_prompt_ids,
        devices=local_devices,
    )

    assert local_engine.max_seqs == 64
    assert local_engine.max_seqs_in_prefill == 64
    assert local_engine.max_tokens_per_round == 128
    assert local_engine.max_queued_tokens == 128
    assert local_engine.max_prefill_size == 640
    assert local_engine.max_pages == 2304
    assert list(local_engine.devices) == local_devices
