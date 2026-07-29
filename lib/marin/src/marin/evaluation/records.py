# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The canonical eval-run record and its object-store layout.

One eval launch writes one ``record.json`` under ``{prefix}/{run_id}/record.json``: the durable,
self-describing account of what model was evaluated on what hardware, whether it succeeded, and the
per-task metrics it produced. The record is the source of truth; evaldash builds its query index from
these object-store records.

This module is import-light on purpose -- stdlib plus fsspec and Pydantic only, no marin/levanter/iris
imports -- so it can be vendored verbatim into a standalone dashboard image that only reads records back.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

import fsspec
from fsspec.core import url_to_fs
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

DEFAULT_RECORDS_PREFIX = "gs://marin-eval-metadata/runs"
# CoreWeave runs write records to the CW-local object store: their workers hold CW S3
# credentials but no GCP ones. The dashboard's ingest scans both prefixes. Access from outside
# the cluster needs `rigging.filesystem.s3_compat.configure_coreweave_s3()` first.
CW_RECORDS_PREFIX = "s3://marin-us-east-02a/marin/eval-metadata/runs"
RECORD_FILE = "record.json"


class RunStatus(StrEnum):
    """Terminal outcome of an eval run.

    ``INFRA_FAILED`` (endpoint never came up, job submission died) is distinct from ``FAILED`` (the
    eval itself ran and reported a bad result) so the dashboard can separate flaky infrastructure from
    genuine model regressions.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INFRA_FAILED = "infra_failed"


class ModelRef(BaseModel):
    """The evaluated model's identity: registry name, weight location, and serving backend."""

    model_config = ConfigDict(frozen=True)

    name: str
    location: str
    backend: str


class EvalTaskRef(BaseModel):
    """One lm-eval task and its shot count."""

    model_config = ConfigDict(frozen=True)

    name: str
    num_fewshot: int


class HarborRef(BaseModel):
    """The Harbor dataset, policy identity, agent, and sandbox environment used by a run."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    version: str
    agent: str
    env: str
    task_limit: int | None = Field(
        default=None,
        description="Marin runtime task cap; source-policy n_tasks remains part of config_digest",
        exclude_if=lambda value: value is None,
    )
    config_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )


class EvalRef(BaseModel):
    """The eval that was run: its name, mechanism, and mechanism-specific detail.

    ``tasks`` carries the lm-eval task list for the ``evalchemy`` mechanism; ``harbor`` carries the
    dataset descriptor for the ``harbor`` mechanism. Exactly one is populated per record.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    mechanism: str
    tasks: tuple[EvalTaskRef, ...] = ()
    harbor: HarborRef | None = None


class HardwareRef(BaseModel):
    """The slice the model was served on. ``region_or_cluster`` is the GCP region or CW cluster name."""

    model_config = ConfigDict(frozen=True)

    platform: str
    accelerator: str
    region_or_cluster: str | None


class Provenance(BaseModel):
    """Where the run came from: launch-time git SHA, eval runtime, and launch host.

    ``eval_runtime`` is Evalchemy's commit-pinned package requirement or Harbor's pinned package
    requirements.
    """

    model_config = ConfigDict(frozen=True)

    git_sha: str
    eval_runtime: str
    launch_host: str


class ServingParams(BaseModel):
    """The model-serving and generation settings a run evaluated under, when the launcher captured them.

    The typed fields are the settings that change results or throughput (parallelism, context length,
    generation budget); ``extra`` carries the long tail -- backend-specific engine flags and extra
    generation kwargs -- as strings so the record stays backend-agnostic. The whole field is optional:
    runs whose launcher did not record it (every run written so far) omit it, and the dashboard shows
    no serving section for them.
    """

    model_config = ConfigDict(frozen=True)

    tensor_parallel_size: int | None = None
    data_parallel_size: int | None = None
    max_model_len: int | None = None
    max_gen_tokens: int | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class RunTiming(BaseModel):
    """The eval's wall-clock window, when the orchestrator captured it.

    ``started_at`` is when the eval began executing (after the served model was healthy);
    ``finished_at`` is when the run reached its terminal state. Both are ISO 8601 strings. The whole
    field is optional on a record: runs whose orchestrator did not record timing, and every record
    written before timing existed, simply omit it, and the dashboard shows no duration for them.
    """

    model_config = ConfigDict(frozen=True)

    started_at: str
    finished_at: str | None = None


class EvalRunRecord(BaseModel):
    """The full account of one eval run, serialized to ``record.json``.

    ``metrics`` is ``{task: {metric: value}}`` as produced by
    :meth:`~marin.evaluation.evalchemy.result.EvalchemyResult.task_metrics`; it is empty when the run did
    not reach the metric-reading stage (an infra failure). The ``evaluation`` field serializes as
    ``eval`` (a reserved-looking but unambiguous JSON key); use ``model_dump(mode="json",
    by_alias=True)`` or ``model_dump_json(by_alias=True)`` to produce it.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    run_id: str
    group_id: str
    """The serve group this run belongs to: one orchestrator serves a model once and evaluates N
    evals against it, writing N records that share a ``group_id``. Standalone runs use their own
    ``run_id`` as the group."""
    created_at: str
    user: str
    version: str | None = None
    """A human version label for the launch (``--version``), e.g. ``2026.07.20`` or ``rl-fix-sweep``.
    Every record in a group shares it. The dashboard groups a model's runs by version so the headline
    matrix shows the latest labelled cohort rather than mixing evals across model states; ``None`` for
    an unlabelled launch."""
    description: str | None = None
    """A free-text note on why the launch was run (``--description``), e.g. ``Trying out a new sweep
    after fixing RL``. Shared by every record in a group and surfaced on the launch in the dashboard."""
    model: ModelRef
    evaluation: EvalRef = Field(alias="eval")
    hardware: HardwareRef
    status: RunStatus
    error: str | None
    results_path: str
    metrics: dict[str, dict[str, float]]
    jobs: dict[str, str]
    """Pipeline role (``orchestrator``/``inference``/``eval``) to Iris job path, for every job the run
    submitted before finishing; a failure before a role's submission simply omits that role."""
    log_tails: dict[str, tuple[str, ...]]
    """For failed runs, the last log lines of the child job(s) behind the failure, keyed like
    ``jobs`` -- enough to diagnose most failures without cluster access. Empty on success."""
    provenance: Provenance
    timing: RunTiming | None = None
    """The eval's wall-clock window when captured; ``None`` on records without recorded timing."""
    serving: ServingParams | None = None
    """The model-serving and generation settings the run evaluated under; ``None`` when not captured."""


def record_path(prefix: str, run_id: str) -> str:
    """The ``record.json`` object path for ``run_id`` under ``prefix``."""
    return f"{prefix.rstrip('/')}/{run_id}/{RECORD_FILE}"


def write_record(record: EvalRunRecord, prefix: str) -> str:
    """Write ``record.json`` under ``{prefix}/{run_id}/`` and return its full path."""
    path = record_path(prefix, record.run_id)
    with fsspec.open(path, "w") as handle:
        handle.write(record.model_dump_json(indent=2, by_alias=True))
    return path


def read_record(path: str) -> EvalRunRecord:
    """Read one ``record.json`` back into an :class:`EvalRunRecord`."""
    with fsspec.open(path, "r") as handle:
        return EvalRunRecord.model_validate_json(handle.read())


@dataclass(frozen=True)
class RecordParseFailure:
    """One ``record.json`` that failed to parse during a listing: its path and the error message.

    Surfaced so the dashboard's Debug view can show records dropped from the snapshot, rather than the
    failure being only logged. A schema drift (a new required field on an old record) shows up here.
    """

    path: str
    error: str


def read_records(prefix: str) -> tuple[list[EvalRunRecord], list[RecordParseFailure]]:
    """Read every ``{prefix}/*/record.json``, returning the parsed records and, separately, the paths
    that failed to parse with their error. Parse failures are logged and collected rather than raised,
    so one malformed record never hides the rest."""
    fs, root = url_to_fs(prefix)
    pattern = f"{root.rstrip('/')}/*/{RECORD_FILE}"
    protocol = f"{prefix.split('://', 1)[0]}://" if "://" in prefix else ""
    records: list[EvalRunRecord] = []
    failures: list[RecordParseFailure] = []
    for match in sorted(fs.glob(pattern)):
        url = f"{protocol}{match}"
        try:
            records.append(read_record(url))
        except Exception as exc:
            logger.warning("skipping unparseable eval record at %s", url, exc_info=True)
            failures.append(RecordParseFailure(path=url, error=f"{type(exc).__name__}: {exc}"))
    return records, failures


def list_records(prefix: str) -> list[EvalRunRecord]:
    """Read every ``{prefix}/*/record.json``, skipping (with a warning) any that fail to parse."""
    return read_records(prefix)[0]
