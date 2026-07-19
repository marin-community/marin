# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball model identity and representative inference goldens."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rigging.filesystem import StoragePath

from tests.cluster.vllm.backend_parity import TokenScore

BATCH_SIZE = 8
TOP_K = 25
PROMPT_BUCKET_MAX_TOKENS = (256, 1024, 4096, 16384, 32768)
# Shared serving-path bound; repeated-run measurements are recorded in #7354.
MAX_PROBABILITY_ERROR = 0.075

_RESOURCES = Path(__file__).parent / "resources"
# Frozen artifacts retain the identifiers of the June 67B training lineage.
_REPRESENTATIVE_GOLDEN_PATH = _RESOURCES / "june_tpu_67b_a2b_step_42150_representative_eval_golden.json"
_PROMPT_FIXTURE_SHA256 = "47863868cbfe336739c8097535f113f4d2dae4954f772eb91511c911433596e8"
_PROMPT_FIXTURE_URL = (
    "https://storage.googleapis.com/marin-public/test-data/vllm/e2e/representative-eval-prompts/"
    f"{_PROMPT_FIXTURE_SHA256}.json"
)


@dataclass(frozen=True)
class ModelIdentity:
    """Checkpoint and export that form one model lineage."""

    run_root: str
    checkpoint_step: int
    export_sha256: str
    export_uri: str

    @property
    def executor_info_path(self) -> str:
        return f"{self.run_root}/.executor_info"

    @property
    def checkpoint_path(self) -> str:
        return f"{self.run_root}/checkpoints/step-{self.checkpoint_step}"

    @property
    def model_name(self) -> str:
        return f"snowball-step-{self.checkpoint_step}-bf16"


SNOWBALL = ModelIdentity(
    run_root=(
        "s3://marin-us-east-02a/marin/grug/"
        "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3"
    ),
    checkpoint_step=42150,
    export_sha256="781bc3291c81ce282be6762520280ebd5ef5b85e88ba65129c2d0162d48ee632",
    export_uri="s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/781bc3291c81ce28/",
)


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
    payload = json.loads(_REPRESENTATIVE_GOLDEN_PATH.read_bytes())
    return tuple(
        RepresentativeGolden(
            id=raw_case["id"],
            top_logprobs=tuple(
                TokenScore(logprob=float(score["logprob"]), token_id=score["token_id"])
                for score in raw_case["top_logprobs"]
            ),
        )
        for raw_case in payload["cases"]
    )


def read_prompt_fixture(
    expected_cases: tuple[RepresentativeGolden, ...],
) -> RepresentativePromptFixture:
    payload = json.loads(StoragePath(_PROMPT_FIXTURE_URL).read_bytes())
    expected_by_id = {case.id: case for case in expected_cases}
    assert {case["id"] for case in payload["cases"]} == expected_by_id.keys()
    cases = tuple(
        RepresentativeCase(
            id=raw_case["id"],
            prompt_token_ids=tuple(raw_case["prompt_token_ids"]),
            top_logprobs=expected_by_id[raw_case["id"]].top_logprobs,
        )
        for raw_case in payload["cases"]
    )
    return RepresentativePromptFixture(
        tokenizer=payload["tokenizer"],
        tokenizer_revision=payload["tokenizer_revision"],
        cases=cases,
        batches=_prompt_batches(cases),
    )


def _prompt_batches(cases: tuple[RepresentativeCase, ...]) -> tuple[PromptBatch, ...]:
    batches = []
    remaining_cases = cases
    for max_tokens in PROMPT_BUCKET_MAX_TOKENS:
        bucket = tuple(
            sorted(
                (case for case in remaining_cases if len(case.prompt_token_ids) <= max_tokens), key=lambda case: case.id
            )
        )
        remaining_cases = tuple(case for case in remaining_cases if len(case.prompt_token_ids) > max_tokens)
        if len(bucket) % BATCH_SIZE:
            raise ValueError(
                f"Prompt bucket <= {max_tokens} has {len(bucket)} cases; expected full batches of {BATCH_SIZE}"
            )
        batches.extend(
            PromptBatch(max_tokens=max_tokens, cases=bucket[start : start + BATCH_SIZE])
            for start in range(0, len(bucket), BATCH_SIZE)
        )

    if remaining_cases:
        raise ValueError(f"Prompts exceed {PROMPT_BUCKET_MAX_TOKENS[-1]} tokens")
    return tuple(batches)


def pad_prompt_batch(batch: PromptBatch, eos_token_id: int) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.full((BATCH_SIZE, batch.max_tokens), eos_token_id, dtype=np.int32)
    last_token_indices = np.empty(BATCH_SIZE, dtype=np.int32)
    for row, case in enumerate(batch.cases):
        token_ids[row, : len(case.prompt_token_ids)] = case.prompt_token_ids
        last_token_indices[row] = len(case.prompt_token_ids) - 1
    return token_ids, last_token_indices
