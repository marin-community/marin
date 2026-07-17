# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np

from tests.cluster.vllm.representative_eval import (
    pad_prompt_batch,
    parse_prompt_fixture,
    read_representative_goldens,
)


def _prompt_payload() -> tuple[dict, tuple]:
    goldens = read_representative_goldens()
    lengths = [1] * 16 + [257] * 16 + [1025] * 16 + [4097] * 8 + [16385] * 8
    cases = [
        {"id": golden.id, "prompt_token_ids": [token_index % 128 for token_index in range(length)]}
        for golden, length in zip(goldens, lengths, strict=True)
    ]
    return {
        "tokenizer": "test-tokenizer",
        "tokenizer_revision": "test-revision",
        "cases": cases,
    }, tuple(reversed(goldens))


def test_representative_fixture_joins_goldens_into_exclusive_full_batches() -> None:
    payload, goldens = _prompt_payload()

    fixture = parse_prompt_fixture(json.dumps(payload).encode(), goldens)

    assert len(fixture.cases) == 64
    assert [batch.max_tokens for batch in fixture.batches] == [256, 256, 1024, 1024, 4096, 4096, 16384, 32768]
    assert all(len(batch.cases) == 8 for batch in fixture.batches)
    batched_case_ids = [case.id for batch in fixture.batches for case in batch.cases]
    assert sorted(batched_case_ids) == sorted(case["id"] for case in payload["cases"])
    assert {case.id: case.top_logprobs for case in fixture.cases} == {
        golden.id: golden.top_logprobs for golden in goldens
    }
    first_batch = fixture.batches[0]
    token_ids, last_token_indices = pad_prompt_batch(first_batch, eos_token_id=999)
    for row, case in enumerate(first_batch.cases):
        assert token_ids[row, : len(case.prompt_token_ids)].tolist() == list(case.prompt_token_ids)
        assert np.all(token_ids[row, len(case.prompt_token_ids) :] == 999)
        assert last_token_indices[row] == len(case.prompt_token_ids) - 1
