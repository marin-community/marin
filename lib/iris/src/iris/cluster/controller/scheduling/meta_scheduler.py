# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task->backend routing: pick a backend for each unpinned job.

The controller holds a collection of task backends. Before per-backend
scheduling, the meta-scheduler routes each *unpinned* job (a job whose
``backend_id`` is still empty) to exactly one backend by matching the job's
constraints against the backends' advertised attributes. The chosen backend is
stamped on the job and its tasks (pinning), so later ticks skip routing and the
per-backend scheduler only ever sees its own slice.

Backends are entities in a :class:`~iris.cluster.constraints.ConstraintIndex`.
Set-valued backend attributes (``device-variant: "v5e-4,v5p-8"``) expand into
posting lists — the backend lands in both the ``device-variant=v5e-4`` and
``=v5p-8`` buckets — so the existing EQ/IN/EXISTS posting-list path matches them
without any change to ``evaluate_constraint``.

A backend with no advertised attributes matches every job (it is a catch-all):
only constraint keys that *some* backend advertises participate in routing;
all other constraints are left to the per-backend scheduler. This makes the
implicit single backend an identity router. A ``device-type=cpu`` constraint
never participates either — CPU is fungible across every backend, so like the
federation and scaling-group routers this one drops it before matching.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintIndex,
    backend_directive,
    is_cpu_device_type_constraint,
)
from iris.cluster.types import JobName

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendRouting:
    """A backend's routing metadata, as the backend advertises it.

    The meta-scheduler routes against this rather than reading backend config:
    ``advertised`` is the backend's attribute sets (already comma-expanded).
    """

    advertised: dict[str, set[str]]


@dataclass(frozen=True)
class RoutableJob:
    """An unpinned job the meta-scheduler must route to a backend."""

    job_id: JobName
    constraints: list[Constraint]


@dataclass(frozen=True)
class RoutingResult:
    """Outcome of one routing pass.

    ``pins`` maps each routed job to its chosen backend id. ``unschedulable``
    maps each unroutable job to a human-readable reason (no backend matches, or
    an explicit ``--backend`` directive named a backend that does not exist).
    """

    pins: dict[JobName, str] = field(default_factory=dict)
    unschedulable: dict[JobName, str] = field(default_factory=dict)


def build_backend_index(routing: dict[str, BackendRouting]) -> ConstraintIndex:
    """Build a constraint index over backends, expanding set-valued attributes.

    Each backend becomes an entity keyed by its backend id; every value of every
    advertised attribute adds the backend to that value's posting list.
    """
    discrete_lists: dict[str, dict[str | int | float, set[str]]] = {}
    entity_attributes: dict[str, dict[str, AttributeValue]] = {}
    for backend_id, route in routing.items():
        attrs: dict[str, AttributeValue] = {}
        for key, values in route.advertised.items():
            for raw in values:
                av = AttributeValue(raw)
                discrete_lists.setdefault(key, {}).setdefault(av.value, set()).add(backend_id)
                attrs[key] = av
        entity_attributes[backend_id] = attrs
    return ConstraintIndex(
        _all_ids=frozenset(routing),
        _discrete_lists=discrete_lists,
        _entity_attributes=entity_attributes,
    )


def route_jobs_to_backends(
    jobs: Sequence[RoutableJob],
    routing: dict[str, BackendRouting],
    index: ConstraintIndex,
    *,
    pick: Callable[[set[str]], str] = lambda matched: min(matched),
) -> RoutingResult:
    """Route each unpinned job to a backend (or mark it unschedulable).

    For each job: honor an explicit ``--backend`` directive if present (iff it
    names an existing backend); otherwise match the job's routing constraints
    against the index. A single match pins it; multiple matches break ties
    deterministically via ``pick`` (default: lexicographic backend id). No
    static match finalizes the job UNSCHEDULABLE.

    Only constraint keys some backend advertises participate in matching; the
    rest are left to the per-backend scheduler, and a ``device-type=cpu``
    constraint is dropped because CPU is fungible across all backends.
    """
    routing_keys = {key for route in routing.values() for key in route.advertised}
    result = RoutingResult()
    for job in jobs:
        directive = backend_directive(job.constraints)
        if directive is not None:
            if directive in routing:
                result.pins[job.job_id] = directive
            else:
                result.unschedulable[job.job_id] = f"backend '{directive}' does not exist"
            continue

        routing_constraints = [
            c for c in job.constraints if c.key in routing_keys and not is_cpu_device_type_constraint(c)
        ]
        matched = index.matching_entities(routing_constraints)
        if not matched:
            result.unschedulable[job.job_id] = "no backend matches the job's constraints"
            continue
        result.pins[job.job_id] = pick(matched)
    return result
