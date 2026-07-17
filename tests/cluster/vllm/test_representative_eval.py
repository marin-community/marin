# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from tests.cluster.vllm.representative_eval import (
    EXPECTED_TOKENIZER,
    EXPECTED_TOKENIZER_REVISION,
    INFERENCE_GOLDEN_PATH,
    PROMPT_BUCKET_MAX_TOKENS,
    RepresentativeCase,
    TokenScore,
    pad_prompt_batch,
    parse_prompt_fixture,
    parse_representative_goldens,
    prompt_batches,
    read_representative_goldens,
)


def _prompt_payload() -> tuple[dict, tuple]:
    goldens = read_representative_goldens()
    lengths = [1] * 16 + [257] * 16 + [1025] * 16 + [4097] * 8 + [16385] * 7 + [29364]
    cases = [
        {"id": golden.id, "prompt_token_ids": [token_index % 128 for token_index in range(length)]}
        for golden, length in zip(goldens, lengths, strict=True)
    ]
    return {
        "tokenizer": EXPECTED_TOKENIZER,
        "tokenizer_revision": EXPECTED_TOKENIZER_REVISION,
        "cases": cases,
    }, goldens


def test_representative_fixture_joins_goldens_into_exclusive_full_batches() -> None:
    payload, goldens = _prompt_payload()

    fixture = parse_prompt_fixture(json.dumps(payload).encode(), goldens)
    batches = prompt_batches(fixture.cases)

    assert len(fixture.cases) == 64
    assert [batch.max_tokens for batch in batches] == [256, 256, 1024, 1024, 4096, 4096, 16384, 32768]
    assert all(len(batch.cases) == 8 for batch in batches)
    assert {case.id: case.top_logprobs for case in fixture.cases} == {
        golden.id: golden.top_logprobs for golden in goldens
    }
    token_ids, last_token_indices = pad_prompt_batch(batches[0], eos_token_id=999)
    for row, case in enumerate(batches[0].cases):
        assert token_ids[row, : len(case.prompt_token_ids)].tolist() == list(case.prompt_token_ids)
        assert np.all(token_ids[row, len(case.prompt_token_ids) :] == 999)
        assert last_token_indices[row] == len(case.prompt_token_ids) - 1


def test_representative_fixture_rejects_duplicate_prompt_case_ids() -> None:
    payload, goldens = _prompt_payload()
    payload["cases"][1]["id"] = payload["cases"][0]["id"]

    with pytest.raises(ValueError):
        parse_prompt_fixture(json.dumps(payload).encode(), goldens)


def test_representative_goldens_reject_duplicate_case_ids() -> None:
    payload = json.loads(INFERENCE_GOLDEN_PATH.read_bytes())
    payload["cases"][1]["id"] = payload["cases"][0]["id"]

    with pytest.raises(ValueError):
        parse_representative_goldens(json.dumps(payload).encode())


def test_prompt_batches_rejects_overlong_and_partial_buckets() -> None:
    score = (TokenScore(logprob=0.0, token_id=0),)
    overlong = RepresentativeCase(
        id="overlong",
        prompt_token_ids=(0,) * (PROMPT_BUCKET_MAX_TOKENS[-1] + 1),
        top_logprobs=score,
    )
    partial = RepresentativeCase(id="partial", prompt_token_ids=(0,), top_logprobs=score)

    with pytest.raises(ValueError):
        prompt_batches((overlong,))
    with pytest.raises(ValueError):
        prompt_batches((partial,))
