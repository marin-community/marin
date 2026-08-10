# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Region guard for the paths a step graph reads and writes.

A :class:`~marin.execution.step_runner.StepRunner` running as an Iris task runs wherever
Iris placed that task, and every child job it submits inherits that placement. When
``MARIN_PREFIX`` — or a step's output or dependency path — is a regional marin bucket
somewhere else, the pipeline streams bulk parquet across regions at egress cost with
nothing in the run to say so. The guard resolves the runtime region once and refuses to
launch such a graph.

Only buckets the active cluster config declares under ``region_buckets`` are compared:
their placement is a deliberate choice and they hold the bulk data. Reads of any other
GCS bucket stay governed by the cumulative transfer budget in
:mod:`rigging.filesystem.cross_region`, which charges bytes as they are read.

That declared placement is why this does not reuse
:func:`rigging.filesystem.check_gcs_paths_same_region`, which a training job runs over
its own config: that one asks GCS where every ``gs://`` path actually lives, one live
bucket lookup at a time. A step graph names thousands of paths and includes shared
buckets like ``marin-data`` whose location is not a placement decision, so the guard
compares only what a config places and does no I/O.

A deliberate cross-region run sets ``MARIN_I_WILL_PAY_FOR_ALL_FEES`` (or passes
``allow_cross_region=True`` to ``StepRunner.run``); each offending path is then logged
and emitted as a telemetry event instead of failing the run.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

from iris.cluster.client.job_info import get_job_info
from rigging import telemetry
from rigging.filesystem import (
    MARIN_CROSS_REGION_OVERRIDE_ENV,
    StoreType,
    data_config,
    declared_bucket_region,
    regions_match,
)
from rigging.telemetry.serialization import EventBody

logger = logging.getLogger(__name__)

CROSS_REGION_OVERRIDE_EVENT = "marin.execution.cross_region_override"


@dataclass(frozen=True)
class CrossRegionPath:
    """A path whose bucket region differs from the region the run executes in."""

    label: str
    path: str
    region: str

    def __str__(self) -> str:
        return f"{self.label}={self.path} (region {self.region})"


class CrossRegionAccessError(RuntimeError):
    """A step graph would read or write a regional marin bucket outside the run's region."""

    def __init__(self, context: str, runtime_region: str, paths: Sequence[CrossRegionPath]):
        self.context = context
        self.runtime_region = runtime_region
        self.paths = tuple(paths)
        super().__init__(context, runtime_region, self.paths)

    def __str__(self) -> str:
        offenders = ", ".join(str(path) for path in self.paths)
        return (
            f"{self.context} would move data across regions: this run executes in "
            f"{self.runtime_region}, but {offenders}. Launch the job in the data's region "
            f"(iris job run --region) or point MARIN_PREFIX at the region-local bucket. To pay "
            f"the egress deliberately, set {MARIN_CROSS_REGION_OVERRIDE_ENV}=1."
        )


@dataclass(frozen=True)
class RegionGuard:
    """The region a run executes in, and whether cross-region access is allowed anyway.

    Resolved once per run by :func:`region_guard`. ``runtime_region`` is ``None`` when the
    run does not pin its children to a GCS region, which makes every check inert.
    """

    runtime_region: str | None
    allow_cross_region: bool

    def check(self, context: str, paths: Sequence[tuple[str, str]]) -> None:
        """Compare ``(label, path)`` pairs against the runtime region.

        The label names the path in the error message (``MARIN_PREFIX``, ``output_path``,
        a dependency's step name). Raises :class:`CrossRegionAccessError` on a mismatch
        unless the run opted in, in which case each offending path is logged and emitted
        as a telemetry event.
        """
        if self.runtime_region is None:
            return
        offenders = _cross_region_paths(self.runtime_region, paths)
        if not offenders:
            return
        if not self.allow_cross_region:
            raise CrossRegionAccessError(context, self.runtime_region, offenders)
        logger.warning(
            "%s runs in %s against %s; %s is set, so this run pays cross-region egress.",
            context,
            self.runtime_region,
            ", ".join(str(offender) for offender in offenders),
            MARIN_CROSS_REGION_OVERRIDE_ENV,
        )
        for offender in offenders:
            telemetry.event(
                CROSS_REGION_OVERRIDE_EVENT,
                EventBody(
                    {
                        "context": context,
                        "runtime_region": self.runtime_region,
                        "label": offender.label,
                        "path": offender.path,
                        "path_region": offender.region,
                    }
                ),
            )


def region_guard(*, allow_cross_region: bool) -> RegionGuard:
    """Build the guard for one run, resolving the runtime region and the override."""
    return RegionGuard(
        runtime_region=runtime_region(),
        allow_cross_region=allow_cross_region or bool(os.environ.get(MARIN_CROSS_REGION_OVERRIDE_ENV)),
    )


def runtime_region() -> str | None:
    """The region of the Iris worker running this task, or ``None``.

    The worker region is the one that matters: Iris pins every child job the task
    submits to it, so it decides where the pipeline's compute — and therefore its
    reads and writes — will happen.

    ``None`` outside Iris. A driver on a laptop or a dev VM submits children with no
    region constraint, so its own placement does not decide where data moves; the
    read-time transfer budget in :mod:`rigging.filesystem.cross_region` covers what
    such a driver reads itself. Also ``None`` for a worker region no cluster config
    serves from GCS: a CoreWeave region (``us-east-02a``) names another cloud and
    says nothing about GCS placement.
    """
    job_info = get_job_info()
    worker_region = job_info.worker_region if job_info is not None else None
    if worker_region is not None and _is_declared_gcs_region(worker_region):
        return worker_region
    return None


def _is_declared_gcs_region(region: str) -> bool:
    """True when the active cluster config serves *region* from a GCS bucket.

    A GCP region the config does not declare is not comparable either: without a
    declared bucket there, nothing in a step graph can name a path in it.
    """
    spec = data_config().region_buckets.get(region)
    return spec is not None and spec.store is StoreType.GCS


def _cross_region_paths(runtime_region: str, paths: Sequence[tuple[str, str]]) -> list[CrossRegionPath]:
    """The ``(label, path)`` pairs in a declared regional bucket outside *runtime_region*."""
    offenders: list[CrossRegionPath] = []
    for label, path in paths:
        if not path.startswith("gs://"):
            continue
        region = declared_bucket_region(path)
        if region is None or regions_match(runtime_region, region):
            continue
        offenders.append(CrossRegionPath(label=label, path=path, region=region))
    return offenders
