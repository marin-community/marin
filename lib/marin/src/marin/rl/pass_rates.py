# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure how often a frozen policy solves each prompt row.

:func:`measure_pass_rates` serves the policy with vLLM and asks for K completions of every row in one
``n=K`` chat request, so each prompt is prefilled once. It grades every returned message and keeps
every sample. The output is unfiltered: ``samples/`` holds each graded sample, and
``pass_rates.parquet`` holds every input row plus ``passed``, ``total``, ``pass_rate``, and
``exclusion`` columns. Filtering on those columns is a separate artifact.

Rows are processed in fixed chunks. Each chunk is committed as one atomic Parquet file once all of
its samples are final. A rerun into the same output skips committed chunks and redraws uncommitted
ones whole. Nothing from an uncommitted chunk is ever read, so resuming cannot bias the rates.

A :class:`PassRateTask`, named by import path, says what a row means: its identity and order, its
chat request, and how to grade a returned assistant message.
"""

import importlib
import json
import logging
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from functools import partial
from itertools import batched
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath
from verifyit.grade import Reward, Status
from zephyr.writers import write_parquet_file

from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.serving_config import inference_config_for_model
from marin.execution.fingerprint import canonical_json
from marin.inference.iris import remote_inference
from marin.inference.types import OpenAIEndpoint

logger = logging.getLogger(__name__)

type Row = dict[str, Any]

PASS_RATES_FILENAME = "pass_rates.parquet"

REQUEST_WORKERS = 256
"""Chat requests in flight at once, each asking for K samples; keep K times this above the engines' slots."""
CHUNK_WORKERS = 4
"""Chunks sampled at once, so the engines stay busy while one chunk's last requests finish."""
REQUEST_TIMEOUT = 1800
"""Seconds one chat request may take, long enough for K samples of a long prompt under load."""
MAX_ATTEMPTS = 3
"""Requests per row before its server errors exclude it."""


class SampleStatus(StrEnum):
    VERIFIED = "verified"
    CONTEXT_WINDOW = "context_window"
    """The prompt does not fit the served context; the row is excluded rather than failed."""
    ERROR = "error"
    """The server kept failing on every attempt, or the grader could not score the reply."""


@dataclass(frozen=True)
class Sample:
    """One graded completion. A failed request yields K samples that share its status."""

    row_id: str
    sample: int
    status: SampleStatus
    attempts: int
    score: float | None = None
    finish_reason: str | None = None
    message_json: str | None = None
    grade_json: str | None = None
    """The grader's diagnostic detail, such as why a reply scored 0."""
    prompt_tokens: int | None = None
    components: dict[str, float] = field(default_factory=dict)
    """The task's named criteria for this reply (each 0.0 or 1.0), written as ``c_<name>`` columns."""


SAMPLE_SCHEMA = pa.schema(
    [
        ("row_id", pa.string()),
        ("sample", pa.int32()),
        ("status", pa.string()),
        ("attempts", pa.int32()),
        ("score", pa.float64()),
        ("finish_reason", pa.string()),
        ("message_json", pa.string()),
        ("grade_json", pa.string()),
        ("prompt_tokens", pa.int64()),
    ]
)


@dataclass(frozen=True)
class PassCount:
    """The columns ``pass_rates.parquet`` adds to every input row."""

    passed: int
    total: int
    pass_rate: float | None
    exclusion: SampleStatus | None
    components: dict[str, int] = field(default_factory=dict)
    """Verified samples passing each named criterion, written as ``passed_<name>`` columns."""


PASS_COUNT_SCHEMA = pa.schema(
    [
        ("passed", pa.int32()),
        ("total", pa.int32()),
        ("pass_rate", pa.float64()),
        ("exclusion", pa.string()),
    ]
)
"""The fixed count columns; ``passed_<name>`` columns follow, one per task component."""


class PassRateTask(Protocol):
    """How pass-rate measurement reads, requests, and grades one kind of prompt row."""

    components: tuple[str, ...]
    """Names of the criteria every scored verdict reports in ``detail["components"]``."""

    def row_id(self, row: Row) -> str: ...

    def sort_key(self, row: Row) -> tuple:
        """Order rows so prompts that share a prefix are requested together."""
        ...

    def request(self, row: Row) -> dict[str, Any]:
        """The chat-completions body for this row, without model, n, or sampling settings."""
        ...

    def grade(self, row: Row, message: dict[str, Any]) -> Reward:
        """Score one returned assistant message; an unscored verdict excludes its row."""
        ...


@dataclass(frozen=True)
class PassRateSampling:
    samples_per_row: int
    max_tokens: int
    enable_thinking: bool
    """Passed to the chat template, so whether the policy is asked to think is always explicit."""
    temperature: float = 1.0
    top_p: float = 1.0
    chunk_size: int = 256
    seed: int | None = None
    """Sent with every request, so a row's K samples are reproducible; ``None`` leaves sampling unseeded."""
    skip_special_tokens: bool = True
    """False keeps special tokens in replies, such as reasoning delimiters the grader must see."""


def measure_pass_rates(
    *,
    rows_path: str,
    rows_filename: str,
    output_path: str,
    task: str,
    model: ModelConfig,
    accelerator: AcceleratorChoice,
    sampling: PassRateSampling,
) -> None:
    """Sample and grade every row of ``rows_path/rows_filename``; write unfiltered pass rates.

    Args:
        rows_path: Directory holding the rows.
        rows_filename: Parquet file of rows inside ``rows_path``.
        output_path: Artifact directory; a rerun into it resumes.
        task: Import path ``module:attribute`` of a :class:`PassRateTask`.
        model: Frozen policy to serve.
        accelerator: Serving slice.
        sampling: How samples are drawn; pinned for the life of ``output_path``.
    """
    output = StoragePath(output_path)
    _pin_config(output, {"rows_filename": rows_filename, "task": task, "model": model, "sampling": sampling})
    row_task = _load_task(task)
    table = _read_table(StoragePath(rows_path) / rows_filename)

    chunks = list(batched(sorted(table.to_pylist(), key=row_task.sort_key), sampling.chunk_size))
    pending = {index: chunk for index, chunk in enumerate(chunks) if not _chunk_path(output, index).exists()}
    logger.info("%d rows in %d chunks; %d chunks left to sample", table.num_rows, len(chunks), len(pending))
    if pending:
        _sample_chunks(pending, output, row_task, model, accelerator, sampling)

    samples = [sample for index in range(len(chunks)) for sample in _read_samples(_chunk_path(output, index))]
    _write_table(with_pass_counts(table, samples, row_task, sampling.samples_per_row), output / PASS_RATES_FILENAME)


def with_pass_counts(table: pa.Table, samples: list[Sample], task: PassRateTask, samples_per_row: int) -> pa.Table:
    """``table`` with the :class:`PassCount` columns appended; no row is dropped."""
    by_row: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_row[sample.row_id].append(sample)

    counts = [pass_count(by_row[task.row_id(row)], samples_per_row, task.components) for row in table.to_pylist()]
    schema = pa.schema([*PASS_COUNT_SCHEMA, *(pa.field(f"passed_{name}", pa.int32()) for name in task.components)])
    count_rows = [
        {
            **{key: value for key, value in asdict(count).items() if key != "components"},
            **{f"passed_{name}": value for name, value in count.components.items()},
        }
        for count in counts
    ]
    count_table = pa.Table.from_pylist(count_rows, schema=schema)
    return pa.Table.from_arrays(table.columns + count_table.columns, names=table.column_names + count_table.column_names)


def pass_count(samples: list[Sample], samples_per_row: int, components: tuple[str, ...]) -> PassCount:
    if len(samples) != samples_per_row:
        raise ValueError(f"expected {samples_per_row} samples, got {len(samples)}")

    statuses = {sample.status for sample in samples}
    excluded = dict.fromkeys(components, 0)
    if SampleStatus.CONTEXT_WINDOW in statuses:
        return PassCount(0, 0, None, SampleStatus.CONTEXT_WINDOW, excluded)
    if SampleStatus.ERROR in statuses:
        return PassCount(0, 0, None, SampleStatus.ERROR, excluded)

    passed = int(sum(sample.score for sample in samples))
    counts = {name: int(sum(sample.components[name] for sample in samples)) for name in components}
    return PassCount(passed, len(samples), passed / len(samples), None, counts)


def _sample_chunks(
    pending: dict[int, tuple[Row, ...]],
    output: StoragePath,
    task: PassRateTask,
    model: ModelConfig,
    accelerator: AcceleratorChoice,
    sampling: PassRateSampling,
) -> None:
    inference = inference_config_for_model(
        model,
        accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    with (
        remote_inference(inference) as session,
        ThreadPoolExecutor(max_workers=REQUEST_WORKERS) as request_pool,
        ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as chunk_pool,
    ):
        sample_row = partial(_sample_row, session.model.endpoint, task, sampling)
        commits = [
            chunk_pool.submit(_commit_chunk, _chunk_path(output, index), rows, sample_row, request_pool, task.components)
            for index, rows in pending.items()
        ]
        for commit in commits:
            commit.result()


def _commit_chunk(
    path: StoragePath,
    rows: tuple[Row, ...],
    sample_row: Callable[[Row], list[Sample]],
    request_pool: ThreadPoolExecutor,
    components: tuple[str, ...],
) -> None:
    """Write a chunk only once every row in it is final, so a partial chunk never exists."""
    samples = [sample for row_samples in request_pool.map(sample_row, rows) for sample in row_samples]
    _write_table(samples_table(samples, components), path)
    logger.info("committed %s", path)


def samples_table(samples: list[Sample], components: tuple[str, ...]) -> pa.Table:
    """Samples as Parquet columns, each component its own ``c_<name>`` column."""
    schema = pa.schema([*SAMPLE_SCHEMA, *(pa.field(f"c_{name}", pa.float64()) for name in components)])
    rows = [
        {
            **{key: value for key, value in asdict(sample).items() if key != "components"},
            **{f"c_{name}": sample.components.get(name) for name in components},
        }
        for sample in samples
    ]
    return pa.Table.from_pylist(rows, schema=schema)


def _sample_row(endpoint: OpenAIEndpoint, task: PassRateTask, sampling: PassRateSampling, row: Row) -> list[Sample]:
    """Draw and grade K samples of one row, retrying the whole request on server errors."""
    row_id = task.row_id(row)
    body = {
        **task.request(row),
        "model": endpoint.model,
        "n": sampling.samples_per_row,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
        "chat_template_kwargs": {"enable_thinking": sampling.enable_thinking},
    }
    if sampling.seed is not None:
        body["seed"] = sampling.seed
    if not sampling.skip_special_tokens:
        body["skip_special_tokens"] = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = _post_chat(endpoint, body)
        if _is_context_overflow(response):
            return _unsampled(row_id, SampleStatus.CONTEXT_WINDOW, attempt, sampling.samples_per_row)
        if response is not None and response.ok:
            return _graded(task, row, row_id, response.json(), attempt)
        logger.warning("request for %s failed on attempt %d", row_id, attempt)
    return _unsampled(row_id, SampleStatus.ERROR, MAX_ATTEMPTS, sampling.samples_per_row)


def _post_chat(endpoint: OpenAIEndpoint, body: dict[str, Any]) -> requests.Response | None:
    """POST a chat completion; ``None`` when the request never reached a response."""
    headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
    try:
        return requests.post(endpoint.url("chat/completions"), json=body, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException:
        logger.warning("chat request raised", exc_info=True)
        return None


def _is_context_overflow(response: requests.Response | None) -> bool:
    return response is not None and response.status_code == 400 and "context length" in response.text.lower()


def _graded(task: PassRateTask, row: Row, row_id: str, payload: dict[str, Any], attempt: int) -> list[Sample]:
    prompt_tokens = payload["usage"]["prompt_tokens"]
    return [
        _graded_sample(task.grade(row, choice["message"]), task.components, row_id, choice, attempt, prompt_tokens)
        for choice in payload["choices"]
    ]


def _graded_sample(
    reward: Reward,
    components: tuple[str, ...],
    row_id: str,
    choice: dict[str, Any],
    attempt: int,
    prompt_tokens: int,
) -> Sample:
    """A scored verdict is a verified sample; an invalid task or grader failure is an error, never a 0."""
    detail = dict(reward.detail)
    scores = detail.pop("components", {})
    is_scored = reward.status == Status.SCORED
    if is_scored and set(scores) != set(components):
        raise ValueError(f"grader reported components {sorted(scores)}, task declares {sorted(components)}")
    return Sample(
        row_id=row_id,
        sample=choice["index"],
        status=SampleStatus.VERIFIED if is_scored else SampleStatus.ERROR,
        attempts=attempt,
        score=reward.reward if is_scored else None,
        finish_reason=choice["finish_reason"],
        message_json=json.dumps(choice["message"]),
        grade_json=json.dumps({"status": reward.status, **detail}, default=str),
        prompt_tokens=prompt_tokens,
        components=scores if is_scored else {},
    )


def _unsampled(row_id: str, status: SampleStatus, attempts: int, samples_per_row: int) -> list[Sample]:
    return [Sample(row_id=row_id, sample=sample, status=status, attempts=attempts) for sample in range(samples_per_row)]


def _pin_config(output: StoragePath, identity: dict[str, Any]) -> None:
    """Record how samples are drawn so a resumed run never mixes samples from two configs."""
    path = output / "config.json"
    pinned = json.loads(canonical_json(identity))
    if not path.exists():
        output.mkdirs()
        path.write_text(json.dumps(pinned, indent=2, sort_keys=True) + "\n")
        return
    if json.loads(path.read_text()) != pinned:
        raise ValueError(f"{path} records a different sampling config; use a new version")


def _load_task(path: str) -> PassRateTask:
    module, attribute = path.split(":")
    return getattr(importlib.import_module(module), attribute)


def _chunk_path(output: StoragePath, index: int) -> StoragePath:
    return output / "samples" / f"chunk-{index:06d}.parquet"


def _read_samples(path: StoragePath) -> list[Sample]:
    return [_sample_from_row(row) for row in _read_table(path).to_pylist()]


def _sample_from_row(row: Row) -> Sample:
    """Inverse of :func:`samples_table`: ``c_<name>`` columns become ``components``."""
    fields = {key: value for key, value in row.items() if not key.startswith("c_")}
    components = {key.removeprefix("c_"): value for key, value in row.items() if key.startswith("c_")}
    return Sample(**{**fields, "status": SampleStatus(fields["status"])}, components=components)


def _read_table(path: StoragePath) -> pa.Table:
    with path.open("rb") as source:
        return pq.read_table(source)


def _write_table(table: pa.Table, path: StoragePath) -> None:
    """Atomically write ``table``; readers see the whole file or none of it."""
    write_parquet_file(table.to_batches(), str(path), schema=table.schema)
