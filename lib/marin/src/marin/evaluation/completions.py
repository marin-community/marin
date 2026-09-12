# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent checkpoint sampling requests and a restart-safe, single-job queue."""

import hashlib
import json
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object


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
    def unique_prompts(self) -> Self:
        if len({prompt.id for prompt in self.prompts}) != len(self.prompts):
            raise ValueError("Prompt IDs must be unique")
        if len(self.prompts) > self.batch_size:
            raise ValueError("Prompt bank exceeds the sampling batch size")
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
            if row.stop_reason == StopReason.EOS and not row.token_ids:
                raise ValueError("EOS result has no generated token")
        return self


class Phase(StrEnum):
    QUEUED = "queued"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


class Entry(Record):
    request: SampleRequest
    phase: Phase = Phase.QUEUED
    attempt: int = 0
    failures: int = 0
    retry_after: datetime | None = None
    started_at: datetime | None = None
    error: str = ""

    @property
    def job_name(self) -> str:
        return f"hero-completions-{self.request.sample_id}-a{self.attempt}"


class Queue(Record):
    entries: dict[str, Entry] = Field(default_factory=dict)
    inventory_at: datetime | None = None
    published_date: str = ""
    report_url: str = ""


class JobStatus(StrEnum):
    MISSING = "missing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEFERRED = "deferred"


class Jobs(Protocol):
    def status(self, name: str) -> JobStatus: ...

    def submit(self, entry: Entry) -> None:
        """Submit the deterministic name with ERROR-on-exists, never replacement."""
        ...


class SampleStore:
    """Store queue state and immutable results through the shared conditional-object API."""

    def __init__(self, root: str):
        self.root = root.rstrip("/")
        self.state = conditional_object(f"{self.root}/queue.json")

    def read_queue(self) -> tuple[Queue, str | None]:
        value = self.state.read()
        if value is None:
            return Queue(), None
        return Queue.model_validate_json(value.data), value.version

    def save_queue(self, queue: Queue, version: str | None) -> str:
        return self.state.write(queue.model_dump_json().encode(), expected_version=version)

    def result_uri(self, sample_id: str) -> str:
        return f"{self.root}/results/{sample_id}.json"

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


def reconcile(store: SampleStore, jobs: Jobs, requests: list[SampleRequest], now: datetime) -> Queue:
    """Discover all requests, recover an active attempt, and start at most one job.

    Concurrent callers either address the same persisted attempt or fail a conditional write.
    A service error propagates. It is not evidence that a job has stopped.
    """
    queue, version = store.read_queue()
    entries = dict(queue.entries)
    for request in requests:
        entries.setdefault(request.sample_id, Entry(request=request))
    active = [(key, entry) for key, entry in entries.items() if entry.phase == Phase.ACTIVE]
    if len(active) > 1:
        raise ValueError("Queue contains more than one active allocation")
    for key, entry in active:
        result = store.result(entry.request)
        status = jobs.status(entry.job_name)
        if status == JobStatus.RUNNING:
            continue  # Wait for teardown even when process zero has written the result.
        if result is not None:
            entries[key] = entry.model_copy(update={"phase": Phase.COMPLETE, "request": result.request, "error": ""})
        elif status == JobStatus.MISSING:
            assert entry.started_at is not None
            if now - entry.started_at < timedelta(hours=48):
                continue  # Retry submission below, using the same name and pinned request.
            entries[key] = entry.model_copy(update={"phase": Phase.FAILED, "error": "Attempt absent after 48 hours"})
        elif status == JobStatus.DEFERRED:
            entries[key] = entry.model_copy(
                update={"phase": Phase.QUEUED, "retry_after": now + timedelta(hours=6), "error": "Waiting for capacity"}
            )
        else:
            failures = entry.failures + 1
            phase = Phase.QUEUED if failures < 3 else Phase.FAILED
            entries[key] = entry.model_copy(
                update={"phase": phase, "failures": failures, "error": f"Job {status} without a result"}
            )

    if not any(entry.phase == Phase.ACTIVE for entry in entries.values()):
        pending = [
            (key, entry)
            for key, entry in entries.items()
            if entry.phase == Phase.QUEUED and (entry.retry_after is None or entry.retry_after <= now)
        ]
        for key, entry in sorted(pending, key=lambda pair: (pair[1].request.checkpoint.step, pair[0]), reverse=True):
            result = store.result(entry.request)
            if result is not None:
                entries[key] = entry.model_copy(update={"phase": Phase.COMPLETE, "request": result.request, "error": ""})
                continue
            entries[key] = entry.model_copy(
                update={
                    "phase": Phase.ACTIVE,
                    "attempt": entry.attempt + 1,
                    "started_at": now,
                    "error": "",
                    "retry_after": None,
                }
            )
            break
    queue = queue.model_copy(update={"entries": entries, "inventory_at": now})
    store.save_queue(queue, version)  # The attempt is durable before any submission side effect.
    for entry in entries.values():
        if entry.phase == Phase.ACTIVE and jobs.status(entry.job_name) == JobStatus.MISSING:
            jobs.submit(entry)
    return queue
