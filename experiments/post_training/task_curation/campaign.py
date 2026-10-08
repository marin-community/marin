# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run independent source artifacts through one retained Zephyr worker pool."""

import contextvars
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from threading import Lock

from fray.current_client import current_client, set_current_client
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, run
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext

logger = logging.getLogger(__name__)


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


class CampaignFailed(RuntimeError):
    """The campaign retained completed sources but at least one source failed."""


class CampaignArtifact(Artifact):
    """A source artifact that records the terminal status the campaign reports for it."""

    status: str


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


def campaign_identity(steps: Sequence[ArtifactStep[Artifact]], worker_image: str) -> str:
    """Seal sample source definitions and the worker runtime for full admission."""
    identity = {
        "sources": sorted((step.name, step.version, step.fingerprint()) for step in steps),
        "worker_image": worker_image,
    }
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()


def require_matching_sample(
    report: dict, expected_identity: str, sample_steps: Sequence[ArtifactStep[Artifact]]
) -> tuple[SourceOutcome, ...]:
    """Validate the whole terminal sample before selecting sources for full processing."""
    if (
        report.get("mode") != "sample"
        or report.get("status") not in {"completed", "failed"}
        or report.get("sample_identity") != expected_identity
    ):
        raise ValueError(
            "Full execution requires a terminal sample with matching source, input, model, sampling "
            "and runtime identities"
        )
    expected = {step.name: step.path() for step in sample_steps}
    sources = tuple(SourceOutcome(**source) for source in report.get("sources", []))
    if len(expected) != len(sample_steps) or len(sources) != len(expected) or {s.name for s in sources} != set(expected):
        raise ValueError("Sample outcomes must cover every canonical source exactly once")
    if any(
        source.path != expected[source.name]
        or source.status not in {"sampled", "completed", "gated", "unsupported", "failed", "incomplete"}
        for source in sources
    ):
        raise ValueError("Sample outcomes must have matching paths and terminal source states")
    counts = dict(Counter(source.status for source in sources))
    if report.get("counts") != counts or (report["status"] == "failed") != bool(counts.get("failed")):
        raise ValueError("Sample summary does not match its source outcomes")
    by_name = {source.name: source for source in sources}
    return tuple(by_name[step.name] for step in sample_steps)


def _build_source(
    step: ArtifactStep[CampaignArtifact], started: Callable[[ArtifactStep[CampaignArtifact]], None]
) -> SourceOutcome:
    started(step)
    result = run(step, max_concurrent=1)[0]
    return SourceOutcome(step.name, result.path, result.status)


def error_chain(error: BaseException) -> str:
    """The error and each explicit cause, outermost first."""
    causes = []
    cause: BaseException | None = error
    while cause is not None:
        causes.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__
    return " <- ".join(causes)


def campaign_report(
    status: str,
    *,
    mode: str,
    sample_identity: str | None,
    outcomes: Sequence[SourceOutcome],
    sample_outcomes: Mapping[str, SourceOutcome] | None,
) -> dict[str, object]:
    """The campaign's status with each source outcome, their counts, and the sample outcomes that admitted them."""
    return {
        "status": status,
        "updated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "sample_identity": sample_identity,
        "counts": dict(Counter(outcome.status for outcome in outcomes)),
        "sources": [asdict(outcome) for outcome in outcomes],
        "sample_outcomes": (
            {name: asdict(outcome) for name, outcome in sample_outcomes.items()} if sample_outcomes is not None else None
        ),
    }


def run_campaign(
    steps: Sequence[ArtifactStep[CampaignArtifact]],
    *,
    pool: CampaignPool,
    runtime: CampaignRuntime,
    report_path: str,
    sample_identity: str | None = None,
    mode: str = "sample",
    sample_outcomes: Mapping[str, SourceOutcome] | None = None,
) -> tuple[SourceOutcome, ...]:
    """Queue source builds, preserve peer outputs on failure, and write the campaign report.

    Every source occupies one admission slot for its entire graph. Consequently
    even a catalog larger than the coordinator limit queues instead of being rejected.
    """
    if pool.concurrent_sources < 1 or pool.max_workers < 1:
        raise ValueError("Campaign source concurrency and worker count must be positive")
    if len({step.name for step in steps}) != len(steps):
        raise ValueError("Campaign source artifact names must be unique")
    if mode == "full" and (sample_outcomes is None or set(sample_outcomes) != {step.name for step in steps}):
        raise ValueError("Full processing requires a sample outcome for every source")
    if mode != "full" and sample_outcomes is not None:
        raise ValueError("Sample admission applies only to full processing")
    outcomes = {step.name: SourceOutcome(step.name, step.path(), "queued") for step in steps}
    admitted = []
    for step in steps:
        sample = sample_outcomes[step.name] if sample_outcomes is not None else None
        if sample is not None and sample.status not in {"sampled", "completed"}:
            outcomes[step.name] = SourceOutcome(
                step.name, step.path(), "not_admitted", f"Sample status: {sample.status}"
            )
        else:
            admitted.append(step)
    report_lock = Lock()

    def write_report(status: str) -> None:
        report = campaign_report(
            status,
            mode=mode,
            sample_identity=sample_identity,
            outcomes=[outcomes[step.name] for step in steps],
            sample_outcomes=sample_outcomes,
        )
        StoragePath(report_path).write_text(json.dumps(report, indent=2))

    def started(step: ArtifactStep[CampaignArtifact]) -> None:
        with report_lock:
            outcomes[step.name] = SourceOutcome(step.name, step.path(), "running")
            write_report("running")

    write_report("running")
    if not admitted:
        write_report("completed")
        return tuple(outcomes[step.name] for step in steps)
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
                    executor.submit(contextvars.copy_context().run, _build_source, step, started): step
                    for step in admitted
                }
                for future in as_completed(futures):
                    step = futures[future]
                    try:
                        outcome = future.result()
                    except Exception as error:
                        logger.exception("Source failed: %s", step.name)
                        outcome = SourceOutcome(step.name, step.path(), "failed", error_chain(error))
                    with report_lock:
                        outcomes[step.name] = outcome
                        write_report("running")
    except Exception:
        write_report("failed")
        raise
    ordered = tuple(outcomes[step.name] for step in steps)
    failed = any(outcome.status == "failed" for outcome in ordered)
    write_report("failed" if failed else "completed")
    if failed:
        raise CampaignFailed(f"Campaign failed; retained source outcomes at {report_path}")
    return ordered
