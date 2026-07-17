# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Representative evaluation prompts and frozen next-token scores for June 67B."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rigging.filesystem import StoragePath

PROMPT_BUCKET_MAX_TOKENS = (256, 1024, 4096, 16384, 32768)
BATCH_SIZE = 8
TOP_K = 25
EXPECTED_CASE_COUNT = 64
EXPECTED_BUCKET_POPULATIONS = (16, 16, 16, 8, 8)
INFERENCE_GOLDEN_PATH = (
    Path(__file__).parent / "resources" / "june_tpu_67b_a2b_step_42150_representative_eval_golden.json"
)
PROMPT_FIXTURE_SHA256 = "47863868cbfe336739c8097535f113f4d2dae4954f772eb91511c911433596e8"
PROMPT_FIXTURE_URL = (
    "https://storage.googleapis.com/marin-public/test-data/vllm/e2e/representative-eval-prompts/"
    f"{PROMPT_FIXTURE_SHA256}.json"
)


@dataclass(frozen=True)
class TokenScore:
    logprob: float
    token_id: int


@dataclass(frozen=True)
class RepresentativeGolden:
    id: str
    top_logprobs: tuple[TokenScore, ...]


@dataclass(frozen=True)
class RepresentativeCase:
    id: str
    prompt_token_ids: tuple[int, ...]
    top_logprobs: tuple[TokenScore, ...]


@dataclass(frozen=True)
class PromptBatch:
    max_tokens: int
    cases: tuple[RepresentativeCase, ...]


@dataclass(frozen=True)
class RepresentativePromptFixture:
    tokenizer: str
    tokenizer_revision: str
    cases: tuple[RepresentativeCase, ...]
    batches: tuple[PromptBatch, ...]


def read_representative_goldens() -> tuple[RepresentativeGolden, ...]:
    """Read the committed 64-case, top-25 evaluation oracle."""
    payload = json.loads(INFERENCE_GOLDEN_PATH.read_bytes())
    cases = tuple(
        RepresentativeGolden(
            id=raw_case["id"],
            top_logprobs=tuple(
                TokenScore(logprob=float(score["logprob"]), token_id=score["token_id"])
                for score in raw_case["top_logprobs"]
            ),
        )
        for raw_case in payload["cases"]
    )

    assert len(cases) == EXPECTED_CASE_COUNT
    assert len({case.id for case in cases}) == EXPECTED_CASE_COUNT
    for case in cases:
        assert len(case.top_logprobs) == TOP_K, case.id
        assert len({score.token_id for score in case.top_logprobs}) == TOP_K, case.id
        assert all(math.isfinite(score.logprob) for score in case.top_logprobs), case.id
    return cases


def parse_prompt_fixture(
    raw: bytes,
    expected_cases: tuple[RepresentativeGolden, ...],
) -> RepresentativePromptFixture:
    """Join prompt IDs to goldens and form the production-shaped batches."""
    expected_by_id = {case.id: case for case in expected_cases}
    assert len(expected_by_id) == len(expected_cases)

    payload = json.loads(raw)
    prompt_case_ids = [raw_case["id"] for raw_case in payload["cases"]]
    assert len(prompt_case_ids) == len(set(prompt_case_ids))
    assert set(prompt_case_ids) == expected_by_id.keys()
    cases = tuple(
        RepresentativeCase(
            id=raw_case["id"],
            prompt_token_ids=tuple(raw_case["prompt_token_ids"]),
            top_logprobs=expected_by_id[raw_case["id"]].top_logprobs,
        )
        for raw_case in payload["cases"]
    )
    assert all(case.prompt_token_ids for case in cases)

    batches = prompt_batches(cases)
    bucket_populations = tuple(
        sum(len(batch.cases) for batch in batches if batch.max_tokens == bucket_max_tokens)
        for bucket_max_tokens in PROMPT_BUCKET_MAX_TOKENS
    )
    assert bucket_populations == EXPECTED_BUCKET_POPULATIONS
    return RepresentativePromptFixture(
        tokenizer=payload["tokenizer"],
        tokenizer_revision=payload["tokenizer_revision"],
        cases=cases,
        batches=batches,
    )


def read_prompt_fixture(
    expected_cases: tuple[RepresentativeGolden, ...],
) -> RepresentativePromptFixture:
    fixture_bytes = StoragePath(PROMPT_FIXTURE_URL).read_bytes()
    actual_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    if actual_sha256 != PROMPT_FIXTURE_SHA256:
        raise ValueError(f"Prompt fixture SHA-256 mismatch: expected {PROMPT_FIXTURE_SHA256}, got {actual_sha256}")
    return parse_prompt_fixture(fixture_bytes, expected_cases)


def prompt_batches(cases: tuple[RepresentativeCase, ...]) -> tuple[PromptBatch, ...]:
    """Return deterministic, exclusive, full batches for each padded prompt length."""
    batches = []
    remaining_cases = cases
    for bucket_max_tokens in PROMPT_BUCKET_MAX_TOKENS:
        bucket = tuple(
            sorted(
                (case for case in remaining_cases if len(case.prompt_token_ids) <= bucket_max_tokens),
                key=lambda case: case.id,
            )
        )
        remaining_cases = tuple(case for case in remaining_cases if len(case.prompt_token_ids) > bucket_max_tokens)
        if len(bucket) % BATCH_SIZE != 0:
            raise ValueError(
                f"Representative prompt bucket <= {bucket_max_tokens} has {len(bucket)} cases; "
                f"expected a multiple of {BATCH_SIZE}"
            )
        batches.extend(
            PromptBatch(max_tokens=bucket_max_tokens, cases=bucket[start : start + BATCH_SIZE])
            for start in range(0, len(bucket), BATCH_SIZE)
        )

    if remaining_cases:
        lengths = {case.id: len(case.prompt_token_ids) for case in remaining_cases}
        raise ValueError(f"Representative prompts exceed {PROMPT_BUCKET_MAX_TOKENS[-1]} tokens: {lengths}")
    return tuple(batches)


def pad_prompt_batch(batch: PromptBatch, eos_token_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad exact prompt IDs and return each row's last real position."""
    if len(batch.cases) != BATCH_SIZE:
        raise ValueError(f"Representative prompt batch has {len(batch.cases)} cases; expected {BATCH_SIZE}")
    token_ids = np.full((BATCH_SIZE, batch.max_tokens), eos_token_id, dtype=np.int32)
    last_token_indices = np.empty(BATCH_SIZE, dtype=np.int32)
    for row, case in enumerate(batch.cases):
        token_ids[row, : len(case.prompt_token_ids)] = case.prompt_token_ids
        last_token_indices[row] = len(case.prompt_token_ids) - 1
    return token_ids, last_token_indices
