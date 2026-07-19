# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The canonical eval-run record and its object-store layout.

One eval launch writes one ``record.json`` under ``{prefix}/{run_id}/record.json``: the durable,
self-describing account of what model was evaluated on what hardware, whether it succeeded, and the
per-task metrics it produced. The record is the source of truth; the Postgres mirror
(:mod:`marin.evaluation.results_db`) is a queryable index built from it.

This module is import-light on purpose -- stdlib plus fsspec only, no marin/levanter/iris imports --
so it can be vendored verbatim into a standalone dashboard image that only reads records back.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum

import fsspec
from fsspec.core import url_to_fs

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


@dataclass(frozen=True)
class ModelRef:
    """The evaluated model's identity: registry name, weight location, and serving backend."""

    name: str
    location: str
    backend: str


@dataclass(frozen=True)
class EvalTaskRef:
    """One lm-eval task and its shot count."""

    name: str
    num_fewshot: int


@dataclass(frozen=True)
class EvalRef:
    """The eval suite that was run: its name, mechanism, and constituent tasks."""

    name: str
    mechanism: str
    tasks: tuple[EvalTaskRef, ...]


@dataclass(frozen=True)
class HardwareRef:
    """The slice the model was served on. ``region_or_cluster`` is the GCP region or CW cluster name."""

    platform: str
    accelerator: str
    region_or_cluster: str | None


@dataclass(frozen=True)
class Provenance:
    """Where the run came from: launch-time git SHA, eval container digest, and launch host."""

    git_sha: str
    evalchemy_image: str
    launch_host: str


@dataclass(frozen=True)
class EvalRunRecord:
    """The full account of one eval run, serialized to ``record.json``.

    ``metrics`` is ``{task: {metric: value}}`` as produced by
    :meth:`~marin.evaluation.eval_result.EvalchemyResult.task_metrics`; it is empty when the run did
    not reach the metric-reading stage (an infra failure).
    """

    run_id: str
    group_id: str
    """The serve group this run belongs to: one orchestrator serves a model once and evaluates N
    evals against it, writing N records that share a ``group_id``. Standalone runs use their own
    ``run_id`` as the group."""
    created_at: str
    user: str
    model: ModelRef
    evaluation: EvalRef
    hardware: HardwareRef
    status: RunStatus
    error: str | None
    results_path: str
    metrics: dict[str, dict[str, float]]
    provenance: Provenance
    jobs: dict[str, str]
    """Pipeline role (``orchestrator``/``serve``/``eval``) to iris job path, for every job the run
    submitted before finishing; a failure before a role's submission simply omits that role."""
    log_tails: dict[str, tuple[str, ...]]
    """For failed runs, the last log lines of the child job(s) behind the failure, keyed like
    ``jobs`` -- enough to diagnose most failures without cluster access. Empty on success."""

    def to_json(self) -> dict:
        """The canonical ``record.json`` structure as a JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "group_id": self.group_id,
            "created_at": self.created_at,
            "user": self.user,
            "model": {
                "name": self.model.name,
                "location": self.model.location,
                "backend": self.model.backend,
            },
            "eval": {
                "name": self.evaluation.name,
                "mechanism": self.evaluation.mechanism,
                "tasks": [{"name": t.name, "num_fewshot": t.num_fewshot} for t in self.evaluation.tasks],
            },
            "hardware": {
                "platform": self.hardware.platform,
                "accelerator": self.hardware.accelerator,
                "region_or_cluster": self.hardware.region_or_cluster,
            },
            "status": self.status.value,
            "error": self.error,
            "results_path": self.results_path,
            "metrics": {task: dict(metrics) for task, metrics in self.metrics.items()},
            "jobs": dict(self.jobs),
            "log_tails": {role: list(lines) for role, lines in self.log_tails.items()},
            "provenance": {
                "git_sha": self.provenance.git_sha,
                "evalchemy_image": self.provenance.evalchemy_image,
                "launch_host": self.provenance.launch_host,
            },
        }

    @classmethod
    def from_json(cls, data: dict) -> EvalRunRecord:
        """Reconstruct a record from the ``record.json`` structure produced by :meth:`to_json`."""
        return cls(
            run_id=data["run_id"],
            group_id=data["group_id"],
            created_at=data["created_at"],
            user=data["user"],
            model=ModelRef(
                name=data["model"]["name"],
                location=data["model"]["location"],
                backend=data["model"]["backend"],
            ),
            evaluation=EvalRef(
                name=data["eval"]["name"],
                mechanism=data["eval"]["mechanism"],
                tasks=tuple(EvalTaskRef(name=t["name"], num_fewshot=t["num_fewshot"]) for t in data["eval"]["tasks"]),
            ),
            hardware=HardwareRef(
                platform=data["hardware"]["platform"],
                accelerator=data["hardware"]["accelerator"],
                region_or_cluster=data["hardware"]["region_or_cluster"],
            ),
            status=RunStatus(data["status"]),
            error=data["error"],
            results_path=data["results_path"],
            metrics={
                task: {metric: float(value) for metric, value in metrics.items()}
                for task, metrics in data["metrics"].items()
            },
            jobs=dict(data["jobs"]),
            log_tails={role: tuple(lines) for role, lines in data["log_tails"].items()},
            provenance=Provenance(
                git_sha=data["provenance"]["git_sha"],
                evalchemy_image=data["provenance"]["evalchemy_image"],
                launch_host=data["provenance"]["launch_host"],
            ),
        )


def record_path(prefix: str, run_id: str) -> str:
    """The ``record.json`` object path for ``run_id`` under ``prefix``."""
    return f"{prefix.rstrip('/')}/{run_id}/{RECORD_FILE}"


def write_record(record: EvalRunRecord, prefix: str) -> str:
    """Write ``record.json`` under ``{prefix}/{run_id}/`` and return its full path."""
    path = record_path(prefix, record.run_id)
    with fsspec.open(path, "w") as handle:
        handle.write(json.dumps(record.to_json(), indent=2))
    return path


def read_record(path: str) -> EvalRunRecord:
    """Read one ``record.json`` back into an :class:`EvalRunRecord`."""
    with fsspec.open(path, "r") as handle:
        return EvalRunRecord.from_json(json.load(handle))


def list_records(prefix: str) -> list[EvalRunRecord]:
    """Read every ``{prefix}/*/record.json``, skipping (with a warning) any that fail to parse."""
    fs, root = url_to_fs(prefix)
    pattern = f"{root.rstrip('/')}/*/{RECORD_FILE}"
    protocol = f"{prefix.split('://', 1)[0]}://" if "://" in prefix else ""
    records: list[EvalRunRecord] = []
    for match in sorted(fs.glob(pattern)):
        url = f"{protocol}{match}"
        try:
            records.append(read_record(url))
        except Exception:
            logger.warning("skipping unparseable eval record at %s", url, exc_info=True)
    return records
