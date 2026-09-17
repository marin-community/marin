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

PRIORITIES_KEY = "priorities.json"
TOP_TOKEN_COUNT = 5


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Prompt(Record):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    text: str = Field(min_length=1)
    seed: int = Field(ge=0)
    source_url: str
    # A reference continuation. Open-ended prompts can have other valid answers.
    expected: str | None = None


class SamplingSpec(Record):
    # Bump the release when sampler behavior changes. Unrelated commits do not trigger backfills.
    release: str
    prompts: tuple[Prompt, ...] = Field(min_length=1)
    batch_size: int = Field(gt=0)
    completions_per_prompt: int = Field(gt=0)
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


class TokenProbability(Record):
    token_id: int = Field(ge=0)
    text: str
    logprob: float = Field(le=0, allow_inf_nan=False)


class TokenScore(Record):
    token_id: int = Field(ge=0)
    # Context-aware text fragments preserve Unicode across byte-token boundaries.
    text: str
    logprob: float = Field(le=0, allow_inf_nan=False)
    top_tokens: tuple[TokenProbability, ...] = Field(min_length=1, max_length=TOP_TOKEN_COUNT)


class Completion(Record):
    sample_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    token_ids: tuple[int, ...]
    text: str
    stop_reason: StopReason
    token_scores: tuple[TokenScore, ...] | None = None


class PromptCompletions(Record):
    prompt_id: str
    prompt_token_ids: tuple[int, ...] = Field(min_length=1)
    samples: tuple[Completion, ...]
    expected_scores: tuple[TokenScore, ...] | None = None


class SampleResult(Record):
    request: SampleRequest
    completions: tuple[PromptCompletions, ...]
    completed_at: str
    eos_token_id: int = Field(ge=0)

    @model_validator(mode="after")
    def complete_sample_set(self) -> Self:
        spec = self.request.spec
        if [row.prompt_id for row in self.completions] != [prompt.id for prompt in spec.prompts]:
            raise ValueError("Results must contain each requested prompt exactly once, in order")
        for prompt, row in zip(spec.prompts, self.completions, strict=True):
            if [sample.sample_index for sample in row.samples] != list(range(spec.completions_per_prompt)):
                raise ValueError("Results must contain every sample index exactly once, in order")
            for sample in row.samples:
                if sample.seed != prompt.seed + sample.sample_index:
                    raise ValueError("Completion seed does not match the prompt and sample index")
                total = len(row.prompt_token_ids) + len(sample.token_ids)
                if any(token < 0 for token in (*row.prompt_token_ids, *sample.token_ids)):
                    raise ValueError("Token IDs must be non-negative")
                ended_with_eos = bool(sample.token_ids) and sample.token_ids[-1] == self.eos_token_id
                if (sample.stop_reason == StopReason.EOS) != ended_with_eos or self.eos_token_id in sample.token_ids[
                    :-1
                ]:
                    raise ValueError("EOS tokens and stop reason disagree")
                if total > spec.context_length or len(sample.token_ids) > spec.max_new_tokens:
                    raise ValueError(f"Generation exceeds the request limits: {row.prompt_id}")
                if sample.stop_reason == StopReason.CONTEXT_LIMIT and total != spec.context_length:
                    raise ValueError("Context-limit result does not fill the context")
                if sample.stop_reason == StopReason.MAX_NEW_TOKENS and len(sample.token_ids) != spec.max_new_tokens:
                    raise ValueError("Token-limit result does not reach the limit")
                if sample.token_scores is not None:
                    if tuple(token.token_id for token in sample.token_scores) != sample.token_ids:
                        raise ValueError("Token scores do not match the generated token IDs")
                    if "".join(token.text for token in sample.token_scores) != sample.text:
                        raise ValueError("Token text does not match the completion")
            if row.expected_scores is not None:
                if prompt.expected is None:
                    raise ValueError("Expected token scores have no reference completion")
                expected_ids = [token.token_id for token in row.expected_scores]
                if not expected_ids or expected_ids[-1] != self.eos_token_id or self.eos_token_id in expected_ids[:-1]:
                    raise ValueError("Expected token scores must end with exactly one EOS token")
                if "".join(token.text for token in row.expected_scores) != prompt.expected:
                    raise ValueError("Expected token text does not match the reference completion")
                if len(row.prompt_token_ids) + len(row.expected_scores) > spec.context_length:
                    raise ValueError("Expected completion exceeds the context limit")
        return self


class SampleStore:
    """Store requests, completed results, and exhausted retry markers."""

    def __init__(self, root: str):
        self.root = StoragePath(root)

    def save_request(self, request: SampleRequest) -> None:
        target = conditional_object(str(self.root / f"requests/{request.sample_id}.json"))
        if target.version() is None:
            target.write(request.model_dump_json().encode(), expected_version=None)

    def requests(self, spec: SamplingSpec) -> list[SampleRequest]:
        """Read only requests with the current specification, before schema validation."""
        current = spec.model_dump(mode="json")
        requests = []
        for path in (self.root / "requests/*.json").glob():
            data = json.loads(path.read_bytes())
            if data["spec"] != current:
                continue
            request = SampleRequest.model_validate(data)
            if request.sample_id != path.name.removesuffix(".json"):
                raise ValueError(f"Request provenance does not match filename {path.name}")
            requests.append(request)
        return requests

    def completed_ids(self) -> set[str]:
        return {path.name.removesuffix(".json") for path in (self.root / "results/*.json").glob()}

    def attempt_names(self) -> set[str]:
        return {path.name.removesuffix(".txt") for path in (self.root / "attempts/*.txt").glob()}

    def priorities(self) -> dict[str, int]:
        saved = conditional_object(str(self.root / PRIORITIES_KEY)).read()
        return json.loads(saved.data) if saved else {}

    def set_priorities(self, sample_ids: list[str], priority_band: int) -> None:
        target = conditional_object(str(self.root / PRIORITIES_KEY))
        saved = target.read()
        priorities = json.loads(saved.data) if saved else {}
        priorities.update(dict.fromkeys(sample_ids, priority_band))
        target.write(json.dumps(priorities).encode(), expected_version=saved.version if saved else None)

    def save_attempt(self, name: str) -> None:
        conditional_object(str(self.root / f"attempts/{name}.txt")).write(b"", expected_version=None)

    def retries_exhausted(self, request: SampleRequest) -> bool:
        return conditional_object(str(self.root / f"failures/{request.sample_id}.txt")).version() is not None

    def save_failure(self, request: SampleRequest, error: str) -> None:
        conditional_object(str(self.root / f"failures/{request.sample_id}.txt")).write(
            error.encode(), expected_version=None
        )

    def result_uri(self, sample_id: str) -> str:
        return str(self.root / f"results/{sample_id}.json")

    def result(self, sample_id: str) -> SampleResult:
        result = SampleResult.model_validate_json(StoragePath(self.result_uri(sample_id)).read_bytes())
        if result.request.sample_id != sample_id:
            raise ValueError(f"Result provenance does not match request {sample_id}")
        return result

    def save_result(self, result: SampleResult) -> None:
        target = conditional_object(self.result_uri(result.request.sample_id))
        try:
            target.write(result.model_dump_json().encode(), expected_version=None)
        except ConditionalWriteError:
            # A previous attempt may have committed its result before its acknowledgement was lost.
            self.result(result.request.sample_id)
