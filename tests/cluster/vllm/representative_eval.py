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
EXPECTED_TOKENIZER = "marin-community/marin-tokenizer"
EXPECTED_TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
EXPECTED_BUCKET_POPULATIONS = (16, 16, 16, 8, 8)
EXPECTED_MAX_PROMPT_TOKENS = 29364
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
class RepresentativePromptFixture:
    tokenizer: str
    tokenizer_revision: str
    cases: tuple[RepresentativeCase, ...]


@dataclass(frozen=True)
class PromptBatch:
    max_tokens: int
    cases: tuple[RepresentativeCase, ...]


def parse_representative_goldens(raw: bytes) -> tuple[RepresentativeGolden, ...]:
    """Decode and validate the committed representative next-token scores."""
    payload = json.loads(raw)
    cases = []
    seen_case_ids = set()
    for raw_case in payload["cases"]:
        case_id = _case_id(raw_case["id"])
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate representative golden case ID: {case_id}")
        seen_case_ids.add(case_id)

        scores = tuple(_token_score(entry, case_id) for entry in raw_case["top_logprobs"])
        if len(scores) != TOP_K:
            raise ValueError(f"Representative golden {case_id} has {len(scores)} scores; expected {TOP_K}")
        token_ids = [score.token_id for score in scores]
        if len(token_ids) != len(set(token_ids)):
            raise ValueError(f"Representative golden {case_id} has duplicate token IDs")
        if any(left.logprob < right.logprob for left, right in zip(scores, scores[1:], strict=False)):
            raise ValueError(f"Representative golden {case_id} scores are not sorted by descending logprob")
        cases.append(RepresentativeGolden(id=case_id, top_logprobs=scores))

    if len(cases) != 64:
        raise ValueError(f"Representative golden has {len(cases)} cases; expected 64")
    return tuple(cases)


def read_representative_goldens() -> tuple[RepresentativeGolden, ...]:
    return parse_representative_goldens(INFERENCE_GOLDEN_PATH.read_bytes())


def parse_prompt_fixture(
    raw: bytes,
    expected_cases: tuple[RepresentativeGolden, ...],
) -> RepresentativePromptFixture:
    """Decode prompt IDs and join them to validated goldens by unique case ID."""
    expected_by_id = {case.id: case for case in expected_cases}
    if len(expected_by_id) != len(expected_cases):
        raise ValueError("Representative goldens contain duplicate case IDs")

    payload = json.loads(raw)
    tokenizer = payload["tokenizer"]
    tokenizer_revision = payload["tokenizer_revision"]
    if tokenizer != EXPECTED_TOKENIZER or tokenizer_revision != EXPECTED_TOKENIZER_REVISION:
        raise ValueError(
            "Representative prompt tokenizer metadata differs from the pinned tokenizer and revision: "
            f"{tokenizer}@{tokenizer_revision}"
        )

    cases = []
    seen_case_ids = set()
    for raw_case in payload["cases"]:
        case_id = _case_id(raw_case["id"])
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate representative prompt case ID: {case_id}")
        seen_case_ids.add(case_id)
        if case_id not in expected_by_id:
            raise ValueError(f"Representative prompt has no matching golden: {case_id}")

        prompt_token_ids = tuple(raw_case["prompt_token_ids"])
        if not prompt_token_ids:
            raise ValueError(f"Representative prompt {case_id} is empty")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in prompt_token_ids
        ):
            raise ValueError(f"Representative prompt {case_id} has an invalid token ID")
        cases.append(
            RepresentativeCase(
                id=case_id,
                prompt_token_ids=prompt_token_ids,
                top_logprobs=expected_by_id[case_id].top_logprobs,
            )
        )

    missing = expected_by_id.keys() - seen_case_ids
    if missing:
        raise ValueError(f"Representative goldens have no matching prompt: {sorted(missing)}")

    fixture = RepresentativePromptFixture(
        tokenizer=tokenizer,
        tokenizer_revision=tokenizer_revision,
        cases=tuple(cases),
    )
    prompt_batches(fixture.cases)
    return fixture


def read_prompt_fixture(
    expected_cases: tuple[RepresentativeGolden, ...],
) -> RepresentativePromptFixture:
    fixture_bytes = StoragePath(PROMPT_FIXTURE_URL).read_bytes()
    actual_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    if actual_sha256 != PROMPT_FIXTURE_SHA256:
        raise ValueError(f"Prompt fixture SHA-256 mismatch: expected {PROMPT_FIXTURE_SHA256}, got {actual_sha256}")
    fixture = parse_prompt_fixture(fixture_bytes, expected_cases)
    batches = prompt_batches(fixture.cases)
    bucket_populations = tuple(
        sum(len(batch.cases) for batch in batches if batch.max_tokens == bucket_max_tokens)
        for bucket_max_tokens in PROMPT_BUCKET_MAX_TOKENS
    )
    if bucket_populations != EXPECTED_BUCKET_POPULATIONS:
        raise ValueError(
            f"Representative prompt bucket populations are {bucket_populations}; "
            f"expected {EXPECTED_BUCKET_POPULATIONS}"
        )
    max_prompt_tokens = max(len(case.prompt_token_ids) for case in fixture.cases)
    if max_prompt_tokens != EXPECTED_MAX_PROMPT_TOKENS:
        raise ValueError(
            f"Longest representative prompt has {max_prompt_tokens} tokens; expected {EXPECTED_MAX_PROMPT_TOKENS}"
        )
    return fixture


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


def _case_id(raw: object) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"Representative case ID must be a non-empty string, got {raw!r}")
    return raw


def _token_score(raw: dict, case_id: str) -> TokenScore:
    logprob = raw["logprob"]
    token_id = raw["token_id"]
    if not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
        raise ValueError(f"Representative golden {case_id} has a non-finite logprob")
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise ValueError(f"Representative golden {case_id} has an invalid token ID")
    return TokenScore(logprob=float(logprob), token_id=token_id)
