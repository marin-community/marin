# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the task->backend meta-scheduler (routing + pinning).

Pure routing logic: given a set of backend configs and a batch of unpinned jobs,
decide which backend each job pins to (or why it cannot be placed). No DB, no
scheduler, no controller.
"""

from iris.cluster.config import AllowPolicy, BackendConfig
from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.controller.scheduling.meta_scheduler import (
    RoutableJob,
    build_backend_index,
    route_jobs_to_backends,
)
from iris.cluster.types import JobName


def _eq(key: str, value: str) -> Constraint:
    return Constraint.create(key=key, op=ConstraintOp.EQ, value=value)


def _backend(kind: str = "worker_daemon", **attributes: str) -> BackendConfig:
    return BackendConfig(kind=kind, attributes=dict(attributes))


def _job(name: str, *constraints: Constraint, user: str = "alice") -> RoutableJob:
    return RoutableJob(job_id=JobName.root(user, name), user=user, constraints=list(constraints))


def _route(configs: dict[str, BackendConfig], *jobs: RoutableJob):
    return route_jobs_to_backends(list(jobs), configs, build_backend_index(configs))


def test_constraint_routes_to_the_backend_advertising_it():
    configs = {
        "gcp": _backend(**{"device-variant": "v5p-8"}),
        "cw": _backend(**{"device-variant": "h100"}),
    }
    job = _job("j", _eq("device-variant", "h100"))

    result = _route(configs, job)

    assert result.pins == {job.job_id: "cw"}
    assert result.unschedulable == {}


def test_set_valued_attribute_matches_any_member():
    # One backend advertises two variants via a comma-split attribute.
    configs = {
        "gcp": _backend(**{"device-variant": "v5e-4,v5p-8"}),
        "cw": _backend(**{"device-variant": "h100"}),
    }

    to_v5e = _job("a", _eq("device-variant", "v5e-4"))
    to_v5p = _job("b", _eq("device-variant", "v5p-8"))

    result = _route(configs, to_v5e, to_v5p)

    assert result.pins == {to_v5e.job_id: "gcp", to_v5p.job_id: "gcp"}


def test_unadvertised_constraint_keys_do_not_block_routing():
    # The job constrains on a key no backend advertises (a worker attribute the
    # per-backend scheduler handles), plus one routing key. Routing must ignore
    # the worker-level key entirely.
    configs = {
        "gcp": _backend(**{"device-variant": "v5p-8"}),
        "cw": _backend(**{"device-variant": "h100"}),
    }
    job = _job("j", _eq("device-variant", "v5p-8"), _eq("zone", "us-central1-a"))

    result = _route(configs, job)

    assert result.pins == {job.job_id: "gcp"}


def test_attributeless_backend_is_a_catch_all():
    # The single-backend identity case: a backend with no advertised attributes
    # routes every job regardless of its constraints.
    configs = {"default": _backend()}
    constrained = _job("j", _eq("device-variant", "anything"), _eq("zone", "z"))

    result = _route(configs, constrained)

    assert result.pins == {constrained.job_id: "default"}


def test_explicit_backend_directive_overrides_attribute_match():
    configs = {
        "gcp": _backend(**{"device-variant": "v5p-8"}),
        "cw": _backend(**{"device-variant": "v5p-8"}),
    }
    # Constraints match BOTH backends, but the directive forces cw.
    job = _job("j", _eq("device-variant", "v5p-8"), _eq("backend", "cw"))

    result = _route(configs, job)

    assert result.pins == {job.job_id: "cw"}


def test_directive_to_missing_backend_is_unschedulable():
    configs = {"gcp": _backend(), "cw": _backend()}
    job = _job("j", _eq("backend", "nope"))

    result = _route(configs, job)

    assert job.job_id not in result.pins
    assert "does not exist" in result.unschedulable[job.job_id]


def test_no_matching_backend_is_unschedulable_without_naming_a_backend():
    configs = {
        "gcp": _backend(**{"device-variant": "v5p-8"}),
        "cw": _backend(**{"device-variant": "h100"}),
    }
    job = _job("j", _eq("device-variant", "v6e-256"))

    result = _route(configs, job)

    reason = result.unschedulable[job.job_id]
    assert "gcp" not in reason and "cw" not in reason


def test_allow_policy_hides_forbidden_backend_from_routing_and_reason():
    # bob's job matches only the restricted backend; he is not on its allow list,
    # so it is unschedulable and the reason must not leak the backend name.
    configs = {
        "public": _backend(**{"device-variant": "v5p-8"}),
        "secret": BackendConfig(
            kind="worker_daemon",
            attributes={"device-variant": "h100"},
            allow_policy=AllowPolicy(users=["alice"]),
        ),
    }
    job = _job("j", _eq("device-variant", "h100"), user="bob")

    result = _route(configs, job)

    assert job.job_id not in result.pins
    assert "secret" not in result.unschedulable[job.job_id]


def test_multiple_matches_break_ties_deterministically():
    configs = {
        "b-second": _backend(**{"device-variant": "v5p-8"}),
        "a-first": _backend(**{"device-variant": "v5p-8"}),
    }
    job = _job("j", _eq("device-variant", "v5p-8"))

    result = _route(configs, job)

    # Default tie-break is the lexicographically smallest backend id.
    assert result.pins == {job.job_id: "a-first"}
