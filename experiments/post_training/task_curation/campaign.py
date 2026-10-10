# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run independent source artifacts through one retained Zephyr worker pool."""

import contextvars
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock

from fray.current_client import current_client, set_current_client
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.models import SourceStatus
from zephyr.context import ZephyrContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """An execution outcome and the stages that actually ran."""

    status: SourceStatus
    outputs: dict[str, str]
    evidence: dict[str, str]
    stages: tuple[str, ...]


class CampaignStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OutcomeStatus(StrEnum):
    """A source's campaign state before or instead of its pipeline's ``SourceStatus``."""

    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"


@dataclass
class CampaignRuntime:
    """Bind source callbacks to the live pool without entering artifact identity."""

    _context: ZephyrContext | None = field(default=None, init=False, repr=False)

    @property
    def context(self) -> ZephyrContext:
        if self._context is None:
            raise RuntimeError("Source execution requires an active campaign runtime")
        return self._context

    @contextmanager
    def activate(self, context: ZephyrContext) -> Iterator[None]:
        if self._context is not None:
            raise RuntimeError("Campaign runtime is already active")
        self._context = context
        try:
            yield
        finally:
            self._context = None


@dataclass(frozen=True)
class CampaignPool:
    max_workers: int
    coordinator_resources: ResourceConfig = field(kw_only=True)
    concurrent_sources: int = 10
    worker_resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(cpu=2, ram="8g"))
    chunk_storage_prefix: str = "s3://marin-us-east-02a/marin/tmp/rl-data-campaign"


@dataclass(frozen=True)
class SourceOutcome:
    name: str
    path: str
    status: str
    error: str | None = None
    result: PipelineResult | None = None


class CampaignFailed(RuntimeError):
    """The campaign retained completed sources but at least one source failed."""


class CampaignArtifact(Artifact):
    """A source artifact that records the terminal status the campaign reports for it."""

    status: str
    result: PipelineResult | None = None


def campaign_plan(steps: Sequence[ArtifactStep[Artifact]], pool: CampaignPool) -> dict[str, object]:
    """Describe a campaign without resolving storage, creating clients, or starting workers."""
    return {
        "sources": [
            {
                "name": step.name,
                "version": step.version,
                "fingerprint": step.fingerprint(),
                "dependencies": [dep.name for dep in step.deps],
            }
            for step in steps
        ],
        "concurrent_sources": pool.concurrent_sources,
        "max_workers": pool.max_workers,
        "worker_resources": asdict(pool.worker_resources),
        "coordinator_resources": asdict(pool.coordinator_resources),
    }


def _build_source(
    step: ArtifactStep[CampaignArtifact], started: Callable[[ArtifactStep[CampaignArtifact]], None]
) -> SourceOutcome:
    started(step)
    result = run(step, max_concurrent=1)[0]
    return SourceOutcome(step.name, result.path, result.status, result=result.result)


def error_chain(error: BaseException) -> str:
    """The error and each explicit cause, outermost first."""
    causes = []
    cause: BaseException | None = error
    while cause is not None:
        causes.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return " <- ".join(causes)


def campaign_report(status: CampaignStatus, *, mode: str, outcomes: Sequence[SourceOutcome]) -> dict[str, object]:
    """The campaign's status with each source outcome and their counts."""
    return {
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "counts": dict(Counter(outcome.status for outcome in outcomes)),
        "sources": [asdict(outcome) for outcome in outcomes],
    }


def write_campaign_report(
    report_path: str, status: CampaignStatus, *, mode: str, outcomes: Sequence[SourceOutcome]
) -> None:
    """Persist the current source outcomes for local or reviewed runs."""
    report = campaign_report(status, mode=mode, outcomes=outcomes)
    StoragePath(report_path).write_text(json.dumps(report, indent=2))


def run_campaign(
    steps: Sequence[ArtifactStep[CampaignArtifact]],
    *,
    pool: CampaignPool,
    runtime: CampaignRuntime,
    report_path: str,
    mode: str = "sample",
) -> tuple[SourceOutcome, ...]:
    """Queue source builds, preserve peer outputs on failure, and write the campaign report.

    Every source occupies one admission slot for its entire graph. Consequently
    even a catalog larger than the coordinator limit queues instead of being rejected.
    """
    if pool.concurrent_sources < 1 or pool.max_workers < 1:
        raise ValueError("Campaign source concurrency and worker count must be positive")
    if len({step.name for step in steps}) != len(steps):
        raise ValueError("Campaign source artifact names must be unique")
    outcomes = {step.name: SourceOutcome(step.name, step.path(), OutcomeStatus.QUEUED) for step in steps}
    report_lock = Lock()

    def write_report(status: CampaignStatus) -> None:
        write_campaign_report(report_path, status, mode=mode, outcomes=list(outcomes.values()))

    def started(step: ArtifactStep[CampaignArtifact]) -> None:
        with report_lock:
            outcomes[step.name] = SourceOutcome(step.name, step.path(), OutcomeStatus.RUNNING)
            write_report(CampaignStatus.RUNNING)

    write_report(CampaignStatus.RUNNING)
    client = current_client()
    try:
        with (
            set_current_client(client),
            ZephyrContext(
                client=client,
                max_workers=pool.max_workers,
                resources=pool.worker_resources,
                coordinator_resources=pool.coordinator_resources,
                chunk_storage_prefix=pool.chunk_storage_prefix,
                max_concurrent_pipelines=pool.concurrent_sources,
                name="rl-data-campaign",
            ) as context,
            runtime.activate(context),
        ):
            with ThreadPoolExecutor(
                max_workers=pool.concurrent_sources, thread_name_prefix="rl-data-source"
            ) as executor:
                futures = {
                    executor.submit(contextvars.copy_context().run, _build_source, step, started): step for step in steps
                }
                for future in as_completed(futures):
                    step = futures[future]
                    try:
                        outcome = future.result()
                    except Exception as error:
                        logger.exception("Source failed: %s", step.name)
                        outcome = SourceOutcome(step.name, step.path(), OutcomeStatus.FAILED, error_chain(error))
                    with report_lock:
                        outcomes[step.name] = outcome
                        write_report(CampaignStatus.RUNNING)
    except Exception:
        write_report(CampaignStatus.FAILED)
        raise
    ordered = tuple(outcomes[step.name] for step in steps)
    failed = any(outcome.status == OutcomeStatus.FAILED for outcome in ordered)
    write_report(CampaignStatus.FAILED if failed else CampaignStatus.COMPLETED)
    if failed:
        raise CampaignFailed(f"Campaign failed; retained source outcomes at {report_path}")
    return ordered
