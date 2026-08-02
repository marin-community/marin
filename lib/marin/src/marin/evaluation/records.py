# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The canonical eval-run record and its object-store layouts.

One eval launch writes one ``record.json`` under ``{prefix}/{run_id}/record.json``: the durable,
self-describing account of what model was evaluated on what hardware, whether it succeeded, and the
per-task metrics it produced. The record is the source of truth; evaldash builds its query index from
these object-store records. New runs use an ``evals`` prefix. Evaldash also reads the former
``eval-metadata/runs`` prefixes and the bounded artifact layout produced before pipeline evals adopted
the shared output root.

This module is import-light on purpose -- stdlib plus fsspec and Pydantic only, no marin/levanter/iris
imports -- so it can be vendored verbatim into a standalone dashboard image that only reads records back.
"""

import logging
import posixpath
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum

import fsspec
from fsspec.core import url_to_fs
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

DEFAULT_RECORDS_PREFIX = "gs://marin-eval-metadata/evals"
# CoreWeave runs write records to the CW-local object store: their workers hold CW S3
# credentials but no GCP ones. The dashboard's ingest scans both prefixes. Access from outside
# the cluster needs `rigging.filesystem.s3_compat.configure_coreweave_s3()` first.
CW_RECORDS_PREFIX = "s3://marin-us-east-02a/marin/evals"
LEGACY_DEFAULT_RECORDS_PREFIX = "gs://marin-eval-metadata/runs"
LEGACY_CW_RECORDS_PREFIX = "s3://marin-us-east-02a/marin/eval-metadata/runs"
RECORD_FILE = "record.json"
_MAX_RECORD_READERS = 16


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
    requirements. Records written before external evaluators used ``eval_image`` or
    ``evalchemy_image`` for the same value.
    """

    model_config = ConfigDict(frozen=True)

    git_sha: str
    eval_runtime: str = Field(validation_alias=AliasChoices("eval_runtime", "eval_image", "evalchemy_image"))
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


@dataclass(frozen=True)
class _RecordRead:
    record: EvalRunRecord | None = None
    failure: RecordParseFailure | None = None
    missing: bool = False


@dataclass(frozen=True)
class RecordScan:
    """One prefix scan, including its path-keyed cache for the next pass."""

    records: tuple[EvalRunRecord, ...]
    failures: tuple[RecordParseFailure, ...]
    records_by_path: dict[str, EvalRunRecord]


def _directory_children(fs, path: str) -> list[str]:
    """Immediate child directories of ``path``; an absent object-store prefix is empty."""
    try:
        children = fs.ls(path, detail=True)
    except FileNotFoundError:
        return []
    return sorted(child["name"] for child in children if child.get("type") == "directory")


def _read_candidates(urls: list[str], cached: Mapping[str, EvalRunRecord]) -> list[_RecordRead]:
    def parse(url: str) -> _RecordRead:
        if url in cached:
            return _RecordRead(record=cached[url])
        try:
            return _RecordRead(record=read_record(url))
        except FileNotFoundError:
            return _RecordRead(missing=True)
        except Exception as exc:
            logger.warning("skipping unparseable eval record at %s", url, exc_info=True)
            return _RecordRead(failure=RecordParseFailure(path=url, error=f"{type(exc).__name__}: {exc}"))

    with ThreadPoolExecutor(max_workers=min(_MAX_RECORD_READERS, max(1, len(urls)))) as executor:
        return list(executor.map(parse, urls))


def scan_records(prefix: str, cached: Mapping[str, EvalRunRecord] | None = None) -> RecordScan:
    """Return current records under ``prefix`` and a cache for the next scan.

    Every root supports the flat ``{prefix}/{run_id}/record.json`` layout. Roots named ``evals`` also
    include ``{prefix}/{model}/{evals}/{version}/{run_id}/record.json``.

    Valid paths in ``cached`` are reused. New records and prior parse failures are read, and deleted
    paths are absent from the returned cache.
    """
    cached = cached or {}
    fs, root = url_to_fs(prefix)
    records: list[EvalRunRecord] = []
    failures: list[RecordParseFailure] = []
    records_by_path: dict[str, EvalRunRecord] = {}

    def collect(urls: list[str], results: list[_RecordRead]) -> None:
        for url, result in zip(urls, results, strict=True):
            if result.record is not None:
                records.append(result.record)
                records_by_path[url] = result.record
            if result.failure is not None:
                failures.append(result.failure)

    # Object-store globs recurse into result payloads. Walk only the two supported record layouts.
    top_level = _directory_children(fs, root)
    flat_urls = [fs.unstrip_protocol(posixpath.join(directory, RECORD_FILE)) for directory in top_level]
    flat_reads = _read_candidates(flat_urls, cached)
    collect(flat_urls, flat_reads)

    if prefix.rstrip("/").endswith("/evals"):
        artifact_roots = [directory for directory, result in zip(top_level, flat_reads, strict=True) if result.missing]
        artifact_runs = artifact_roots
        for _ in range(3):
            artifact_runs = [child for parent in artifact_runs for child in _directory_children(fs, parent)]
        artifact_urls = [fs.unstrip_protocol(posixpath.join(directory, RECORD_FILE)) for directory in artifact_runs]
        collect(artifact_urls, _read_candidates(artifact_urls, cached))
    return RecordScan(records=tuple(records), failures=tuple(failures), records_by_path=records_by_path)


def read_records(prefix: str) -> tuple[list[EvalRunRecord], list[RecordParseFailure]]:
    """Read every eval record under ``prefix`` and return records plus parse failures."""
    scan = scan_records(prefix)
    return list(scan.records), list(scan.failures)


def list_records(prefix: str) -> list[EvalRunRecord]:
    """Read every ``{prefix}/*/record.json``, skipping (with a warning) any that fail to parse."""
    return read_records(prefix)[0]
