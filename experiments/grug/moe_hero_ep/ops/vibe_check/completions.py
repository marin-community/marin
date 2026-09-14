# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable requests and completed sample sets for hero checkpoints."""

import hashlib
import json
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.storage_path import StoragePath


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Prompt(Record):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    text: str = Field(min_length=1)
    seed: int = Field(ge=0)
    source_url: str


class SamplingSpec(Record):
    # Bump the release when sampler behavior changes. Unrelated commits do not trigger backfills.
    release: str
    prompts: tuple[Prompt, ...] = Field(min_length=1)
    batch_size: int = Field(gt=0)
    tokenizer: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model: dict
    temperature: float = Field(ge=0, allow_inf_nan=False)
    max_new_tokens: int = Field(gt=0)
    context_length: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_prompt_bank(self) -> Self:
        if len({prompt.id for prompt in self.prompts}) != len(self.prompts):
            raise ValueError("Prompt IDs must be unique")
        return self


class Checkpoint(Record):
    uri: str
    run_id: str
    step: int = Field(ge=0)
    timestamp: str
    metadata_digest: str


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SampleRequest(Record):
    checkpoint: Checkpoint
    spec: SamplingSpec
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_cluster: str

    @property
    def sample_id(self) -> str:
        return digest({"checkpoint": self.checkpoint.model_dump(), "spec": self.spec.model_dump()})


class StopReason(StrEnum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"
    CONTEXT_LIMIT = "context_limit"


class Completion(Record):
    prompt_id: str
    prompt_token_ids: tuple[int, ...] = Field(min_length=1)
    token_ids: tuple[int, ...]
    text: str
    stop_reason: StopReason


class SampleResult(Record):
    request: SampleRequest
    completions: tuple[Completion, ...]
    completed_at: str
    eos_token_id: int = Field(ge=0)

    @model_validator(mode="after")
    def complete_sample_set(self) -> Self:
        spec = self.request.spec
        if [row.prompt_id for row in self.completions] != [prompt.id for prompt in spec.prompts]:
            raise ValueError("Results must contain each requested prompt exactly once, in order")
        for row in self.completions:
            total = len(row.prompt_token_ids) + len(row.token_ids)
            if any(token < 0 for token in (*row.prompt_token_ids, *row.token_ids)):
                raise ValueError("Token IDs must be non-negative")
            ended_with_eos = bool(row.token_ids) and row.token_ids[-1] == self.eos_token_id
            if (row.stop_reason == StopReason.EOS) != ended_with_eos or self.eos_token_id in row.token_ids[:-1]:
                raise ValueError("EOS tokens and stop reason disagree")
            if total > spec.context_length or len(row.token_ids) > spec.max_new_tokens:
                raise ValueError(f"Generation exceeds the request limits: {row.prompt_id}")
            if row.stop_reason == StopReason.CONTEXT_LIMIT and total != spec.context_length:
                raise ValueError("Context-limit result does not fill the context")
            if row.stop_reason == StopReason.MAX_NEW_TOKENS and len(row.token_ids) != spec.max_new_tokens:
                raise ValueError("Token-limit result does not reach the limit")
        return self


class SampleStore:
    """Store requests, completed results, and exhausted retry markers."""

    def __init__(self, root: str):
        self.root = StoragePath(root)

    def save_request(self, request: SampleRequest) -> None:
        target = conditional_object(str(self.root / f"requests/{request.sample_id}.json"))
        if target.version() is None:
            target.write(request.model_dump_json().encode(), expected_version=None)

    def requests(self) -> list[SampleRequest]:
        return [SampleRequest.model_validate_json(path.read_bytes()) for path in (self.root / "requests/*.json").glob()]

    def results(self) -> list[SampleResult]:
        return [SampleResult.model_validate_json(path.read_bytes()) for path in (self.root / "results/*.json").glob()]

    def failed(self, request: SampleRequest) -> bool:
        return conditional_object(str(self.root / f"failures/{request.sample_id}.txt")).version() is not None

    def save_failure(self, request: SampleRequest, error: str) -> None:
        conditional_object(str(self.root / f"failures/{request.sample_id}.txt")).write(
            error.encode(), expected_version=None
        )

    def result_uri(self, sample_id: str) -> str:
        return str(self.root / f"results/{sample_id}.json")

    def result(self, request: SampleRequest) -> SampleResult | None:
        value = conditional_object(self.result_uri(request.sample_id)).read()
        if value is None:
            return None
        result = SampleResult.model_validate_json(value.data)
        if result.request.sample_id != request.sample_id:
            raise ValueError(f"Result provenance does not match request {request.sample_id}")
        return result

    def save_result(self, result: SampleResult) -> None:
        target = conditional_object(self.result_uri(result.request.sample_id))
        try:
            target.write(result.model_dump_json().encode(), expected_version=None)
        except ConditionalWriteError:
            # A previous attempt may have committed its result before its acknowledgement was lost.
            if self.result(result.request) is None:
                raise ValueError("Committed result disappeared during recovery") from None
