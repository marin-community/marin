# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic `log`-shaped data and the operator query corpus it is measured with.

The generator reproduces the properties of the deployed `log` namespace that
decide query cost, measured from its production segments:

- a few thousand distinct `key` values per ~10M-row segment, so the column costs
  a few MiB on disk for a 15 GiB namespace but ~100 bytes per row once decoded
- keys that carry the job and task *inside* the value rather than as a prefix,
  which is why operators search them with `LIKE '%<job>%'` instead of a range
- one job holding a ~0.01% share of the rows, so a job-scoped query is a needle
- log bodies wide enough that a segment's row groups are sized by `data`

The batches carry the columns of the `log` schema but the generator never
registers one: Finelog auto-registers `log` at boot, and a client-side
`RegisterTable` would pin the index policy to whatever this file declares,
hiding a server-side schema change from the measurement.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum

import pyarrow as pa

from finelog.schema import LOG_REGISTERED_SCHEMA, schema_to_arrow

LOG_NAMESPACE = "log"
# A log row encodes to roughly 160 bytes, so this keeps a batch well under the
# server's 16 MiB WriteRows body limit.
DEFAULT_BATCH_ROWS = 50_000
DEFAULT_SEGMENTS = 8
DEFAULT_WARMUP_ITERATIONS = 1
DEFAULT_ITERATIONS = 5
START_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z

USERS = ("power", "benjaminfeuer", "held", "ryan", "nfliu", "abhinav")
JOB_PREFIXES = (
    "grug-train",
    "eval",
    "harrier-dedup",
    "glm52-datagen",
    "tei-index",
    "marin-serve",
)
SOURCES = ("stdout", "stderr")

# Distinct keys per generated segment. Production L3 segments hold 2.1k-3.6k
# distinct keys across ~10M rows; matching that band keeps the per-span distinct
# key count a trigram bloom has to encode realistic.
KEYS_PER_SEGMENT = 2_560
TASKS_PER_JOB = 16

# One row in this many is given to the target job, in every segment, so a
# job-scoped query is a needle spread across the whole namespace rather than one
# file. Production jobs behave this way: a run outlives many compactions.
TARGET_JOB_ROW_STRIDE = 4_096

# The job every job-scoped workload searches for. It is named like every other
# generated job; `TARGET_JOB_INDEX` is a multiple of both tuple lengths so the
# naming rule below places it under `TARGET_USER` with the `grug-train` prefix.
TARGET_JOB_INDEX = 408
TARGET_USER = USERS[TARGET_JOB_INDEX % len(USERS)]
TARGET_JOB = f"{JOB_PREFIXES[TARGET_JOB_INDEX % len(JOB_PREFIXES)]}-{TARGET_JOB_INDEX:04d}"

LEVEL_DEBUG = 10
LEVEL_INFO = 20
LEVEL_WARNING = 30
LEVEL_ERROR = 40

# One row in this many is an error line, which is what `JOB_FIRST_ERROR` and the
# body searches look for.
ERROR_ROW_STRIDE = 3_001

# Bodies are templated but carry per-row identifiers and floats, so a segment
# compresses like real log text (~13x) instead of collapsing to a dictionary.
_BODY_TEMPLATES = (
    "step {step} loss {loss:.5f} lr {lr:.3e} tokens {tokens} throughput {rate:.2f} tok/s",
    "iris.worker heartbeat rank={rank} host=worker-{host:05d} elapsed={elapsed:.3f}s queue={queue}",
    "levanter checkpoint step={step} shard={rank} bytes={tokens} elapsed={elapsed:.3f}s",
    "GET /v1/completions status=200 latency={elapsed:.3f}s prompt_tokens={queue} model=marin-{host:05d}",
    "loading dataset shard {rank}/{queue} path=gs://marin-us-central2/tokenized/{step:09d}.parquet",
)
_ERROR_TEMPLATES = (
    'Traceback (most recent call last): File "/opt/marin/train.py", line {step}, in <module>',
    "RuntimeError: CUDA_ERROR_OUT_OF_MEMORY allocating {tokens} bytes on device {rank}",
)


class WorkloadName(StrEnum):
    JOB_LAST_TIMESTAMP = "job_last_timestamp"
    JOB_LINE_COUNT = "job_line_count"
    JOB_TASK_KEYS = "job_task_keys"
    JOB_FIRST_ERROR = "job_first_error"
    JOB_TAIL = "job_tail"
    JOB_TEXT_SEARCH = "job_text_search"
    JOB_SOURCE_SPLIT = "job_source_split"
    JOB_RECENT_WINDOW = "job_recent_window"
    USER_PREFIX_TAIL = "user_prefix_tail"
    TASK_TAIL = "task_tail"
    GLOBAL_TEXT_SEARCH = "global_text_search"
    SOURCE_ROLLUP = "source_rollup"


@dataclass(frozen=True)
class LogDatasetSpec:
    """Dimensions of a generated `log` corpus."""

    rows: int
    segments: int = DEFAULT_SEGMENTS
    batch_rows: int = DEFAULT_BATCH_ROWS

    @property
    def rows_per_segment(self) -> int:
        return max(self.rows // self.segments, 1)


@dataclass(frozen=True)
class Workload:
    """One corpus query, and whether it reads the whole namespace unpruned.

    `prunes_on_key_substring` marks the shapes whose only selective predicate is
    a `key` substring: they decode every row of `key` in the namespace unless a
    segment index answers them.
    """

    name: WorkloadName
    sql: str
    prunes_on_key_substring: bool


def job_name(index: int) -> str:
    return f"{JOB_PREFIXES[index % len(JOB_PREFIXES)]}-{index:04d}"


def key_for_slot(slot: int) -> str:
    """The `key` for key slot `slot`: `/user/job-coord/job/task:attempt`."""
    job_index, task = divmod(slot, TASKS_PER_JOB)
    job = job_name(job_index)
    return f"/{USERS[job_index % len(USERS)]}/{job}-coord/{job}/{task}:0"


def target_key(task: int = 0) -> str:
    """The exact key a task-scoped workload reads, inside [`TARGET_JOB`]."""
    return key_for_slot(TARGET_JOB_INDEX * TASKS_PER_JOB + task)


def dataset_facts(spec: LogDatasetSpec) -> dict[str, object]:
    """The corpus dimensions a result file needs to be reproducible.

    Only the requested dimensions and the generator's constants. Achieved
    per-segment rows, distinct values, and matched rows come from the server and
    the query results, so nothing here restates generator arithmetic.
    """
    return {
        **asdict(spec),
        "rows_per_segment": spec.rows_per_segment,
        "keys_per_segment": KEYS_PER_SEGMENT,
        "tasks_per_job": TASKS_PER_JOB,
        "target_job": TARGET_JOB,
        "target_key": target_key(),
        "target_job_row_stride": TARGET_JOB_ROW_STRIDE,
        "error_row_stride": ERROR_ROW_STRIDE,
    }


def generate_batch(spec: LogDatasetSpec, start: int, stop: int) -> pa.RecordBatch:
    """Build one deterministic log-shaped Arrow batch for row range ``[start, stop)``."""
    arrow_schema = schema_to_arrow(LOG_REGISTERED_SCHEMA)
    keys_per_segment = min(KEYS_PER_SEGMENT, max(spec.rows_per_segment, 1))
    target_slot = TARGET_JOB_INDEX * TASKS_PER_JOB

    keys: list[str] = []
    sources: list[str] = []
    bodies: list[str] = []
    epochs: list[int] = []
    levels: list[int] = []
    for index in range(start, stop):
        segment = min(index // max(spec.rows_per_segment, 1), spec.segments - 1)
        offset = index - segment * spec.rows_per_segment
        # Rows walk that segment's own slice of key slots, so each segment holds
        # a distinct key band and one job's rows cluster physically once
        # compaction sorts by `(key, seq)`.
        slot = segment * keys_per_segment + (offset % keys_per_segment)
        if offset % TARGET_JOB_ROW_STRIDE == 0:
            slot = target_slot + (offset // TARGET_JOB_ROW_STRIDE) % TASKS_PER_JOB
        is_error = index % ERROR_ROW_STRIDE == 0
        templates = _ERROR_TEMPLATES if is_error else _BODY_TEMPLATES
        keys.append(key_for_slot(slot))
        sources.append(SOURCES[index % len(SOURCES)])
        levels.append(LEVEL_ERROR if is_error else (LEVEL_INFO, LEVEL_DEBUG, LEVEL_WARNING)[index % 3])
        bodies.append(
            templates[index % len(templates)].format(
                step=index,
                loss=3.5 - (index % 100_000) * 1e-5,
                lr=1e-3 / (1 + index % 997),
                tokens=index * 4_096,
                rate=1_000.0 + (index % 9_973) * 0.25,
                rank=slot % TASKS_PER_JOB,
                host=slot % 65_536,
                elapsed=(index % 60_013) * 0.001,
                queue=index % 512,
            )
        )
        epochs.append(START_MS + index)

    return pa.RecordBatch.from_arrays(
        [
            pa.array(keys, type=pa.string()),
            pa.array(sources, type=pa.string()),
            pa.array(bodies, type=pa.string()),
            pa.array(epochs, type=pa.int64()),
            pa.array(levels, type=pa.int32()),
        ],
        schema=arrow_schema,
    )


def segment_row_ranges(spec: LogDatasetSpec) -> Iterator[tuple[int, int]]:
    """Half-open row ranges, one per segment the corpus is written as.

    The loader compacts after each range so the store ends up with one segment
    per key band, the way a live namespace accumulates them. The last range
    absorbs the remainder.
    """
    for index in range(spec.segments):
        start = index * spec.rows_per_segment
        stop = spec.rows if index == spec.segments - 1 else start + spec.rows_per_segment
        if start < stop:
            yield start, stop


def generate_batches(spec: LogDatasetSpec, start: int = 0, stop: int | None = None) -> Iterator[pa.RecordBatch]:
    """Yield deterministic log-shaped Arrow batches over ``[start, stop)``."""
    stop = spec.rows if stop is None else stop
    for batch_start in range(start, stop, spec.batch_rows):
        yield generate_batch(spec, batch_start, min(batch_start + spec.batch_rows, stop))


def build_workloads(spec: LogDatasetSpec) -> tuple[Workload, ...]:
    """The operator query corpus for the `log` namespace.

    Every shape is one operators issue against a live cluster: scoping to a job
    by substring, tailing a task, finding the first error in a run, and searching
    bodies. The `prunes_on_key_substring` shapes are the ones that read the whole
    namespace when no segment index answers the `key` predicate.
    """
    table = f'"{LOG_NAMESPACE}"'
    job = TARGET_JOB
    recent_from = START_MS + max(spec.rows - 3_600_000, 0)

    return (
        Workload(
            WorkloadName.JOB_LAST_TIMESTAMP,
            f"SELECT max(epoch_ms) AS last_ms FROM {table} WHERE key LIKE '%{job}%'",
            prunes_on_key_substring=True,
        ),
        Workload(
            WorkloadName.JOB_LINE_COUNT,
            f"SELECT count(*) AS lines FROM {table} WHERE key LIKE '%{job}%'",
            prunes_on_key_substring=True,
        ),
        Workload(
            WorkloadName.JOB_TASK_KEYS,
            f"SELECT DISTINCT key FROM {table} WHERE key LIKE '%{job}%' ORDER BY key",
            prunes_on_key_substring=True,
        ),
        Workload(
            WorkloadName.JOB_FIRST_ERROR,
            f"""
SELECT epoch_ms, key, data
FROM {table}
WHERE key LIKE '%{job}%' AND level >= {LEVEL_ERROR}
ORDER BY seq
LIMIT 5
""".strip(),
            prunes_on_key_substring=True,
        ),
        Workload(
            WorkloadName.JOB_TAIL,
            f"""
SELECT epoch_ms, key, source, data
FROM {table}
WHERE key LIKE '%{job}%'
ORDER BY seq DESC
LIMIT 200
""".strip(),
            prunes_on_key_substring=True,
        ),
        Workload(
            WorkloadName.JOB_TEXT_SEARCH,
            f"""
SELECT epoch_ms, key, data
FROM {table}
WHERE key LIKE '%{job}%' AND contains(data, 'CUDA_ERROR')
ORDER BY seq DESC
LIMIT 100
""".strip(),
            prunes_on_key_substring=False,
        ),
        Workload(
            WorkloadName.JOB_SOURCE_SPLIT,
            f"""
SELECT source, count(*) AS lines
FROM {table}
WHERE key LIKE '%{job}%'
GROUP BY source
ORDER BY lines DESC
""".strip(),
            prunes_on_key_substring=True,
        ),
        Workload(
            WorkloadName.JOB_RECENT_WINDOW,
            f"""
SELECT epoch_ms, key, data
FROM {table}
WHERE key LIKE '%{job}%' AND epoch_ms >= {recent_from}
ORDER BY seq DESC
LIMIT 200
""".strip(),
            prunes_on_key_substring=True,
        ),
        Workload(
            WorkloadName.USER_PREFIX_TAIL,
            f"""
SELECT epoch_ms, key, data
FROM {table}
WHERE key LIKE '/{TARGET_USER}/%'
ORDER BY seq DESC
LIMIT 100
""".strip(),
            prunes_on_key_substring=False,
        ),
        Workload(
            WorkloadName.TASK_TAIL,
            f"""
SELECT epoch_ms, source, data
FROM {table}
WHERE key = '{target_key()}'
ORDER BY seq DESC
LIMIT 200
""".strip(),
            prunes_on_key_substring=False,
        ),
        Workload(
            WorkloadName.GLOBAL_TEXT_SEARCH,
            f"""
SELECT epoch_ms, key, data
FROM {table}
WHERE contains(data, 'CUDA_ERROR')
ORDER BY seq DESC
LIMIT 100
""".strip(),
            prunes_on_key_substring=False,
        ),
        Workload(
            WorkloadName.SOURCE_ROLLUP,
            f"SELECT source, count(*) AS lines FROM {table} GROUP BY source ORDER BY lines DESC",
            prunes_on_key_substring=False,
        ),
    )
