# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the task->backend meta-scheduler (routing + pinning).

Pure routing logic: given a set of backend configs and a batch of unpinned jobs,
decide which backend each job pins to (or why it cannot be placed). No DB, no
scheduler, no controller.
"""

from iris.cluster.config import (
    BackendConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    backend_attribute_sets,
)
from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.controller.scheduling.meta_scheduler import (
    BackendRouting,
    RoutableJob,
    build_backend_index,
    route_jobs_to_backends,
)
from iris.cluster.types import AcceleratorType, JobName


def _eq(key: str, value: str) -> Constraint:
    return Constraint.create(key=key, op=ConstraintOp.EQ, value=value)


def _backend(kind: str = "worker_daemon", **attributes: str) -> BackendConfig:
    return BackendConfig(kind=kind, attributes=dict(attributes))


def _accel_backend(*groups: tuple[str, str]) -> BackendConfig:
    """A backend in the composer-synthesized shape: device attributes come from
    scale_groups (as ``resolve_backends`` builds them), not literal ``attributes``.
    Each group is a ``(device_type, device_variant)`` pair.
    """
    scale_groups = {
        f"sg{i}": ScaleGroupConfig(
            name=f"sg{i}",
            num_vms=1,
            resources=ScaleGroupResources(device_type=AcceleratorType(dt), device_variant=dv),
        )
        for i, (dt, dv) in enumerate(groups)
    }
    return BackendConfig(kind="worker_daemon", scale_groups=scale_groups)


def _job(name: str, *constraints: Constraint) -> RoutableJob:
    return RoutableJob(job_id=JobName.root("alice", name), constraints=list(constraints))


def _routing(configs: dict[str, BackendConfig]) -> dict[str, BackendRouting]:
    """Mirror how the composer/controller derive each backend's routing metadata."""
    return {backend_id: BackendRouting(advertised=backend_attribute_sets(cfg)) for backend_id, cfg in configs.items()}


def _route(configs: dict[str, BackendConfig], *jobs: RoutableJob):
    routing = _routing(configs)
    return route_jobs_to_backends(list(jobs), routing, build_backend_index(routing))


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


def test_multiple_matches_break_ties_deterministically():
    configs = {
        "b-second": _backend(**{"device-variant": "v5p-8"}),
        "a-first": _backend(**{"device-variant": "v5p-8"}),
    }
    job = _job("j", _eq("device-variant", "v5p-8"))

    result = _route(configs, job)

    # Default tie-break is the lexicographically smallest backend id.
    assert result.pins == {job.job_id: "a-first"}


# --- Attributes auto-derived from scale_groups (no literal `attributes`) --------
# The same backend_attribute_sets map that these exercise also feeds the
# federation router, so routing here mirrors what a peer advertises live.


def test_scale_group_derived_attrs_route_a_matching_job():
    # A backend synthesized from a GPU scale group advertises device-type and
    # device-variant with no literal attributes; a job requesting them routes to it.
    configs = {"cw": _accel_backend(("gpu", "H100"))}
    job = _job("j", _eq("device-type", "gpu"), _eq("device-variant", "h100"))

    result = _route(configs, job)

    assert result.pins == {job.job_id: "cw"}


def test_multi_scale_group_backend_routes_either_variant():
    # Two GPU scale groups on one backend advertise the union of their variants.
    configs = {"cw": _accel_backend(("gpu", "H100"), ("gpu", "A100"))}
    to_h100 = _job("h", _eq("device-variant", "h100"))
    to_a100 = _job("a", _eq("device-variant", "a100"))

    result = _route(configs, to_h100, to_a100)

    assert result.pins == {to_h100.job_id: "cw", to_a100.job_id: "cw"}


def test_cpu_only_backend_derives_nothing_and_stays_catch_all():
    # An existing CPU-only single-backend cluster derives no device attributes, so
    # it keeps matching every job regardless of constraints (unchanged behavior).
    configs = {"default": _accel_backend(("cpu", ""))}
    job = _job("j", _eq("device-variant", "anything"), _eq("zone", "z"))

    result = _route(configs, job)

    assert result.pins == {job.job_id: "default"}


def test_mixed_cpu_gpu_backend_routes_both_cpu_and_gpu_jobs():
    # The CoreWeave shape: a CPU group plus a GPU group. The backend advertises
    # only the GPU attributes; a GPU job matches them and a CPU job (no device
    # constraints) still routes because those keys never appear in its constraints.
    configs = {"default": _accel_backend(("cpu", ""), ("gpu", "H100"))}
    gpu_job = _job("g", _eq("device-type", "gpu"), _eq("device-variant", "h100"))
    cpu_job = _job("c")

    result = _route(configs, gpu_job, cpu_job)

    assert result.pins == {gpu_job.job_id: "default", cpu_job.job_id: "default"}


def test_single_backend_rejects_a_variant_it_does_not_provide():
    # Behavior change (fail fast): a single GPU backend advertises its variant, so a
    # job requesting a different variant is unschedulable at meta-scheduling rather
    # than dispatched and then failed by the per-backend scheduler.
    configs = {"default": _accel_backend(("gpu", "H100"))}
    job = _job("j", _eq("device-type", "gpu"), _eq("device-variant", "a100"))

    result = _route(configs, job)

    assert job.job_id not in result.pins
    assert "no backend matches" in result.unschedulable[job.job_id]


def test_explicit_cpu_device_type_constraint_stays_routable():
    # A GPU backend advertises device-type, making it a routing key. A CPU-pinned
    # job (explicit device-type=cpu) must not be rejected: CPU is fungible, so the
    # constraint is dropped before matching — as the federation and scaling-group
    # routers already do — and the job routes to a backend.
    configs = {
        "cpu": _accel_backend(("cpu", "")),
        "gpu": _accel_backend(("gpu", "H100")),
    }
    job = _job("j", _eq("device-type", "cpu"))

    result = _route(configs, job)

    assert job.job_id in result.pins
    assert result.unschedulable == {}


def test_cpu_constraint_alongside_gpu_variant_still_routes_by_variant():
    # Dropping device-type=cpu must not drop the other routing constraints: a job
    # carrying both device-type=cpu and device-variant=h100 still routes by variant.
    configs = {
        "cw": _accel_backend(("gpu", "H100")),
        "gcp": _accel_backend(("gpu", "A100")),
    }
    job = _job("j", _eq("device-type", "cpu"), _eq("device-variant", "h100"))

    result = _route(configs, job)

    assert result.pins == {job.job_id: "cw"}
