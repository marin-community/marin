# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The normalized evaluation contract and its typed FineStore writer and reader.

Every eval mechanism normalizes its native per-question output into :class:`EvalSample` when the
mechanism produces it, and every consumer (the dashboard's sample browser, ad-hoc parquet analysis)
reads that contract back through this module. The producer and consumer share one schema definition,
so a format change is a change to this file and its round-trip test, never a guessing game in a
viewer. Each evaluator owns the conversion from its native output into this contract.

A sample is either ``multiple_choice`` (the model scored a fixed choice list by loglikelihood;
``choices`` carries the per-choice scores with the model's pick and the gold index resolved at
export time) or ``generation`` (the model produced free text; ``output`` is the raw completion and
``extracted`` the post-filter answer). Prompts are either raw text (``prompt_text``, the completions
API) or a chat message list (``prompt_messages``); exactly one is set.

A run's durable storage is one finestore archive rooted at its results directory (see the archive
writer below). The archive pins each table's column types from the pydantic model (via
``finestore.arrow_schema``), so the writer is ``model_dump`` and the reader is ``model_validate``,
with nested fields stored as parquet structs/lists and the dynamic-keyed ``metrics`` as a
``map<string,double>`` that cannot drift its type across shards. ``schema_version`` is a model field,
so it rides in every row for future evolution. The legacy per-(sub)task
``samples_<task>_<timestamp>.parquet`` layout is still read by the dashboard's migration fallback.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping
from enum import StrEnum

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from rigging.filesystem.storage_path import prefix_join

from finestore.schema import arrow_schema
from finestore.store import DataStore

# 3: metrics moved from an inferred struct to a pinned ``map<string,double>`` and the unused
# ``exchange_uri`` was dropped (finestore now pins both eval tables' schemas from these models).
# 4: the lm-eval extraction filter joined the samples primary key. A task that scores one document
# under several filters (gsm8k under strict-match and flexible-extract) emits one sample per filter,
# and each is a distinct row rather than one overwriting the other.
SCHEMA_VERSION = 4
ROLLOUT_SCHEMA_VERSION = 1

SAMPLES_PREFIX = "samples_"
SAMPLES_SUFFIX = ".parquet"


class SampleKind(StrEnum):
    """How the model was queried for this sample."""

    MULTIPLE_CHOICE = "multiple_choice"
    GENERATION = "generation"
    AGENTIC = "agentic"


class ConversationType(StrEnum):
    """The interaction protocol represented by a normalized rollout."""

    COMPLETION = "completion"
    CHAT = "chat"
    AGENTIC = "agentic"


class ParticipantType(StrEnum):
    """A provider-neutral participant category."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ENVIRONMENT = "environment"
    OTHER = "other"


class RolloutContentType(StrEnum):
    """The semantic type of one ordered rollout part."""

    MESSAGE = "message"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class Message(BaseModel):
    """One chat turn of a messages-based prompt."""

    role: str
    content: str


class Choice(BaseModel):
    """One scored option of a multiple-choice sample."""

    label: str
    text: str
    loglikelihood: float | None = None
    is_greedy: bool | None = None


class Grading(BaseModel):
    """How one prediction was scored, made explicit so the UI can show *why* a sample is (in)correct.

    ``method`` names the grader: ``lm-eval:<metric>`` for a harness metric, ``harbor:<verifier>`` for a
    Harbor trial's verifier, ``judge:<model>`` for an LLM judge. ``metric`` is the full headline key
    (with lm-eval's ``,<filter>`` suffix), ``filter`` the extraction filter that produced it, ``score``
    its value, and ``passed`` whether it cleared the pass threshold. ``detail`` carries the grader's raw
    output verbatim (the verifier/judge JSON) as the escape hatch for anything the fields do not.
    """

    method: str
    metric: str | None = None
    filter: str | None = None
    score: float | None = None
    passed: bool | None = None
    detail: str = "{}"


class EvalSample(BaseModel):
    """One evaluated question: the prompt, the model's answer, the gold answer, and its scores.

    ``prompt_text`` and ``prompt_messages`` are mutually exclusive; ``choices``/``model_choice``/
    ``target_choice`` are set for ``multiple_choice`` samples, ``output``/``extracted`` for
    ``generation`` samples, and ``trajectory_uri`` for ``agentic`` (Harbor) samples. ``grading`` makes
    the scoring decision explicit for the UI. ``doc`` is the source dataset row as a JSON string, kept
    verbatim as the escape hatch for anything the normalized fields do not carry.

    Rows stay bounded: the one unbounded payload, an agentic trajectory, is stored as a sibling blob
    and referenced here by ``trajectory_uri``, so the columnar reader never has to materialize it to
    page the light columns.
    """

    schema_version: int = SCHEMA_VERSION
    task: str
    doc_id: str
    kind: SampleKind
    prompt_text: str | None = None
    prompt_messages: list[Message] | None = None
    choices: list[Choice] | None = None
    model_choice: int | None = None
    target_choice: int | None = None
    output: str | None = None
    extracted: str | None = None
    target_text: str | None = None
    trajectory_uri: str | None = None
    grading: Grading | None = None
    metrics: dict[str, float] = {}
    correct: bool | None = None
    doc: str = "{}"


class RolloutRecord(BaseModel):
    """One ordered part of an evaluation rollout.

    A rollout is addressed by ``task``/``doc_id``/``trial_id``. ``turn_id`` orders conversational
    turns and ``part_id`` orders message, reasoning, tool-call, and tool-result content within a
    turn. ``participant_type`` is the provider-neutral role; ``participant_id`` retains the concrete
    source name, such as a model or an evaluator-native role.

    Structured content is encoded as JSON in ``content`` and identified by ``content_type``. The
    raw evaluator output remains in the archive's source artifacts and trajectory blobs.
    """

    schema_version: int = ROLLOUT_SCHEMA_VERSION
    task: str
    doc_id: str
    trial_id: str = ""
    turn_id: int
    part_id: int
    conversation_type: ConversationType
    participant_type: ParticipantType
    participant_id: str
    content_type: RolloutContentType
    content: str
    metadata_json: str = "{}"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None


# --------------------------------------------------------------------------------------------------
# Writer: the parquet schema is the pydantic model, so ``EvalSample.model_validate(row)`` on any
# ``to_pylist`` row is the reader.
# --------------------------------------------------------------------------------------------------


def write_sample_parquet(fs, dest: str, samples: Iterable[EvalSample]) -> None:
    """Write samples as one parquet at ``dest`` on ``fs``."""
    rows = [sample.model_dump() for sample in samples]
    # Parquet cannot store a zero-field struct; the model default restores {} on read.
    if all(not row["metrics"] for row in rows):
        for row in rows:
            del row["metrics"]
    table = pa.Table.from_pylist(rows)
    with fs.open(dest, "wb") as handle:
        pq.write_table(table, handle)


# --------------------------------------------------------------------------------------------------
# FineStore archive writer: an eval run's durable output is one FineStore archive rooted at its
# results directory, with a ``samples`` table (one row per evaluated question), a ``steps`` table
# (Harbor trajectories flattened for compatibility), a versioned provider-neutral rollout
# conversation table, and finestore's reserved ``blobs`` table for raw artifacts.
# --------------------------------------------------------------------------------------------------

ARCHIVE_SAMPLES_TABLE = "samples"
ARCHIVE_STEPS_TABLE = "steps"
ARCHIVE_ROLLOUTS_TABLE = f"rollouts_v{ROLLOUT_SCHEMA_VERSION}"

# Blob-name prefix for preserved evaluator-native inputs. A blob under this prefix is the verbatim
# bytes of a file the export read, keyed by its path relative to the run's results root, so a rebuild
# can recover the same inputs from the archive alone.
SOURCES_PREFIX = "sources"

# The archive's own keys on a samples row, added at write time (not EvalSample fields): a sample is
# unique within a run by its task, its document, the trial that produced it (for multi-attempt Harbor
# runs), and the extraction filter that scored it (for lm-eval tasks that apply more than one).
# evalchemy leaves ``trial_id`` empty (one attempt per document). An lm-eval row names its filter
# even when the task applies only one ("none"), so the column holds that name; a Harbor trial, or a
# benchmark that reports no filter at all, leaves it empty.
TRIAL_ID_COLUMN = "trial_id"
FILTER_COLUMN = "filter"
SAMPLES_MERGE_KEY = ("task", "doc_id", TRIAL_ID_COLUMN, FILTER_COLUMN)
STEPS_MERGE_KEY = ("task", "doc_id", TRIAL_ID_COLUMN, "step_id")
ROLLOUTS_MERGE_KEY = ("task", "doc_id", TRIAL_ID_COLUMN, "turn_id", "part_id")


def samples_schema() -> pa.Schema:
    """The pinned ``samples`` table schema: the :class:`EvalSample` columns plus the archive's
    ``trial_id`` and ``filter`` keys. Pinning fixes each column's type up front, so ``metrics`` stays
    a ``map<string,double>`` and no per-flush inference can drift a column's type across shards."""
    return pa.schema(
        [
            *arrow_schema(EvalSample),
            pa.field(TRIAL_ID_COLUMN, pa.string()),
            pa.field(FILTER_COLUMN, pa.string()),
        ]
    )


def steps_schema() -> pa.Schema:
    """The pinned ``steps`` table schema, derived from :class:`StepRecord` (its own columns carry the
    primary key), so token-id and logprob columns keep their list types across shards."""
    return arrow_schema(StepRecord)


def rollouts_schema() -> pa.Schema:
    """The pinned schema for provider-neutral rollout conversation parts."""
    return arrow_schema(RolloutRecord)


def sample_to_archive_row(sample: EvalSample, *, trial_id: str = "") -> dict:
    """One archive ``samples`` row: the sample's JSON-mode dump plus its ``trial_id`` and ``filter``
    archive keys. The filter comes from the sample's own grading, so a caller sets it by grading the
    sample, not by passing it here."""
    row = sample.model_dump(mode="json")
    row[TRIAL_ID_COLUMN] = trial_id
    row[FILTER_COLUMN] = (sample.grading.filter or "") if sample.grading else ""
    return row


def sample_from_archive_row(row: dict) -> EvalSample:
    """Reconstruct an :class:`EvalSample` from an archive ``samples`` row.

    ``metrics`` is a pinned ``map<string,double>``, so a batch that wrote no metrics reads back a null
    map and a partly-populated map can carry null values; normalize null, absent, and null-valued
    metrics to a plain dict (a null value means the metric is absent, not zero) before validation.
    Arrow map-pair iterables are accepted directly. Archive-only keys (``trial_id`` and finestore's
    ``_seq``/``_writer``/``_gen`` columns) are ignored by the model.
    """
    metrics = dict(row.get("metrics") or {})
    return EvalSample.model_validate(
        {**row, "metrics": {name: value for name, value in metrics.items() if value is not None}}
    )


@dataclasses.dataclass(frozen=True)
class StepRecord:
    """One normalized agentic step: the row schema of the archive ``steps`` table.

    Token-id and logprob arrays stay as native list columns (the RL signal a reader projects); nested
    tool calls and observations are kept as JSON strings.
    """

    task: str
    doc_id: str
    trial_id: str
    step_id: int | None
    source: str | None
    model_name: str | None
    message: str | None
    reasoning_content: str | None
    tool_calls_json: str | None
    observation_json: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    prompt_token_ids: list[int] | None
    completion_token_ids: list[int] | None
    logprobs: list[float] | None


class EvaluationStore:
    """One eval run's FineStore archive.

    ``samples`` holds evaluation and grading data, ``steps`` retains the earlier flattened Harbor
    contract, and ``rollouts_vN`` holds provider-neutral conversation parts under its schema version.
    Raw evaluator files and trajectories remain blobs. Reads go through ``ReadView`` over the same
    root.
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store
        self._samples = store.table(
            ARCHIVE_SAMPLES_TABLE, schema=samples_schema(), primary_key=SAMPLES_MERGE_KEY, schema_version=SCHEMA_VERSION
        )
        self._steps = store.table(
            ARCHIVE_STEPS_TABLE, schema=steps_schema(), primary_key=STEPS_MERGE_KEY, schema_version=SCHEMA_VERSION
        )
        self._rollouts = store.table(
            ARCHIVE_ROLLOUTS_TABLE,
            schema=rollouts_schema(),
            primary_key=ROLLOUTS_MERGE_KEY,
            schema_version=ROLLOUT_SCHEMA_VERSION,
        )

    @classmethod
    def open(cls, root: str, *, writer_id: str) -> EvaluationStore:
        """Open the archive for the run rooted at ``root``, written by ``writer_id``."""
        return cls(DataStore.open(root, writer_id=writer_id))

    def add_sample(self, sample: EvalSample, *, trial_id: str = "") -> None:
        """Append one evaluated question to the ``samples`` table."""
        self._samples.append(sample_to_archive_row(sample, trial_id=trial_id))

    def add_steps(self, steps: Iterable[StepRecord]) -> None:
        """Append normalized agentic steps to the ``steps`` table."""
        self._steps.extend(dataclasses.asdict(step) for step in steps)

    def add_rollouts(self, records: Iterable[RolloutRecord]) -> None:
        """Append provider-neutral conversation parts to the ``rollouts`` table."""
        self._rollouts.extend(record.model_dump(mode="json") for record in records)

    def add_artifact(self, name: str, raw: bytes, *, metadata: Mapping[str, object] | None = None) -> str:
        """Store an opaque artifact and return its ``finestore://`` URI."""
        return self._store.write_object(name, raw, metadata)

    def add_source_artifact(self, name: str, raw: bytes, *, content_type: str) -> str:
        """Preserve one evaluator-native source file inside the archive; return its blob URI.

        ``name`` is the evaluator-owned path below ``sources/``. A rebuild can re-derive the tables
        from the archive alone, without an external results tree still being intact.
        """
        return self.add_artifact(
            prefix_join(SOURCES_PREFIX, name),
            raw,
            metadata={"content_type": content_type, "sha256": hashlib.sha256(raw).hexdigest()},
        )

    def flush(self) -> None:
        """Write buffered rows to shards now, bounding how much a large payload holds in memory."""
        self._store.flush()

    def seal(self) -> None:
        """Flush every table and mark the run complete."""
        self._store.seal()

    def close(self) -> None:
        """Stop the writer and flush any remaining rows."""
        self._store.close()

    def __enter__(self) -> EvaluationStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
