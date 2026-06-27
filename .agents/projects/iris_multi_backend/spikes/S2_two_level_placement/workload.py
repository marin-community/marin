# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic-but-realistic multi-backend fleet + workload generators for SPIKE S2.

Fleet: two equivalent TPU backends (cross-backend routing/rebalancing), one GPU
k8s backend (capability routing), two CPU pools (fungible-CPU bin-packing across
backends). Workload: a seeded mix of explicit --backend pins, capability
constraints (device-variant), gangs/coscheduled jobs, region pins, and varied
priorities/users.
"""

from __future__ import annotations

import random

from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute
from iris.cluster.types import JobName

from harness import BATCH, GIB, INTER, PROD, BackendDef, JobSpec, WorkerSpec

USERS = ["alice", "bob", "carol", "dave"]


def _eq(key: str, value: str) -> Constraint:
    return Constraint.create(key=key, op=ConstraintOp.EQ, value=value)


def _tpu_backend(backend_id: str, region: str, n_slices: int, slice_size: int) -> BackendDef:
    workers: list[WorkerSpec] = []
    for s in range(n_slices):
        group = f"{backend_id}-s{s}"
        for i in range(slice_size):
            workers.append(
                WorkerSpec(
                    worker_id=f"{backend_id}-s{s}-{i}",
                    backend=backend_id,
                    variant="v5e-4",
                    cpu_millicores=8000,
                    memory_bytes=32 * GIB,
                    gpu_count=0,
                    tpu_count=4,
                    group=group,
                    region=region,
                )
            )
    return BackendDef(
        backend_id=backend_id,
        kind="worker_daemon",
        static={
            WellKnownAttribute.DEVICE_TYPE: {"tpu"},
            WellKnownAttribute.DEVICE_VARIANT: {"v5e-4"},
            WellKnownAttribute.REGION: {region},
            "backend": {backend_id},
            "provider": {"gcp"},
        },
        workers=workers,
    )


def _gpu_backend(backend_id: str, region: str, n_hosts: int, gpus_per_host: int) -> BackendDef:
    workers: list[WorkerSpec] = []
    for h in range(n_hosts):
        group = f"{backend_id}-h{h}"
        for i in range(gpus_per_host):
            workers.append(
                WorkerSpec(
                    worker_id=f"{backend_id}-h{h}-{i}",
                    backend=backend_id,
                    variant="h100",
                    cpu_millicores=8000,
                    memory_bytes=64 * GIB,
                    gpu_count=1,
                    tpu_count=0,
                    group=group,
                    region=region,
                )
            )
    return BackendDef(
        backend_id=backend_id,
        kind="k8s",
        static={
            WellKnownAttribute.DEVICE_TYPE: {"gpu"},
            WellKnownAttribute.DEVICE_VARIANT: {"h100"},
            WellKnownAttribute.REGION: {region},
            "backend": {backend_id},
            "provider": {"coreweave"},
        },
        workers=workers,
    )


def _cpu_backend(backend_id: str, region: str, n_workers: int) -> BackendDef:
    workers = [
        WorkerSpec(
            worker_id=f"{backend_id}-{i}",
            backend=backend_id,
            variant="cpu",
            cpu_millicores=64000,
            memory_bytes=256 * GIB,
            gpu_count=0,
            tpu_count=0,
            group=None,
            region=region,
        )
        for i in range(n_workers)
    ]
    return BackendDef(
        backend_id=backend_id,
        kind="worker_daemon",
        static={
            WellKnownAttribute.DEVICE_TYPE: {"cpu"},
            WellKnownAttribute.REGION: {region},
            "backend": {backend_id},
            "provider": {"gcp"},
        },
        workers=workers,
    )


def standard_fleet() -> list[BackendDef]:
    return [
        _tpu_backend("gcp-tpu-west", "us-west4", n_slices=6, slice_size=4),
        _tpu_backend("gcp-tpu-central", "us-central1", n_slices=6, slice_size=4),
        _gpu_backend("cw-east-h100", "us-east", n_hosts=4, gpus_per_host=8),
        _cpu_backend("cpu-pool-west", "us-west4", n_workers=8),
        _cpu_backend("cpu-pool-central", "us-central1", n_workers=8),
    ]


def _band(rng: random.Random) -> int:
    return rng.choices([PROD, INTER, BATCH], weights=[0.2, 0.5, 0.3])[0]


def standard_workload(rng: random.Random, horizon: int, jobs_per_tick: float) -> list[JobSpec]:
    """A mixed multi-backend stream. ~jobs_per_tick arrivals/tick over [0, horizon-15)."""
    jobs: list[JobSpec] = []
    counter = 0
    kinds = ["v5e_solo", "v5e_gang", "h100_solo", "h100_gang", "cpu", "v5e_pinned"]
    weights = [0.30, 0.12, 0.18, 0.06, 0.28, 0.06]

    for tick in range(horizon - 15):
        n = rng.poisson(jobs_per_tick) if hasattr(rng, "poisson") else _poisson(rng, jobs_per_tick)
        for _ in range(n):
            kind = rng.choices(kinds, weights=weights)[0]
            user = rng.choice(USERS)
            band = _band(rng)
            counter += 1
            jid = JobName.root(user, f"j{counter}")
            if kind == "v5e_solo":
                jobs.append(
                    JobSpec(
                        job_id=jid, user=user, band=band, arrival=tick,
                        duration=rng.randint(3, 8), num_tasks=1, coscheduled=False, group_by=None,
                        variant="v5e-4", cpu_millicores=8000, memory_bytes=16 * GIB, gpu_count=0, tpu_count=4,
                        constraints=(_eq(WellKnownAttribute.DEVICE_TYPE, "tpu"), _eq(WellKnownAttribute.DEVICE_VARIANT, "v5e-4")),
                        pinned_backend=None,
                    )
                )
            elif kind == "v5e_pinned":
                jobs.append(
                    JobSpec(
                        job_id=jid, user=user, band=band, arrival=tick,
                        duration=rng.randint(3, 8), num_tasks=1, coscheduled=False, group_by=None,
                        variant="v5e-4", cpu_millicores=8000, memory_bytes=16 * GIB, gpu_count=0, tpu_count=4,
                        constraints=(
                            _eq(WellKnownAttribute.DEVICE_TYPE, "tpu"),
                            _eq(WellKnownAttribute.DEVICE_VARIANT, "v5e-4"),
                            _eq("backend", "gcp-tpu-west"),
                        ),
                        pinned_backend="gcp-tpu-west",
                    )
                )
            elif kind == "v5e_gang":
                jobs.append(
                    JobSpec(
                        job_id=jid, user=user, band=band, arrival=tick,
                        duration=rng.randint(4, 10), num_tasks=4, coscheduled=True, group_by="slice-group",
                        variant="v5e-4", cpu_millicores=8000, memory_bytes=16 * GIB, gpu_count=0, tpu_count=4,
                        constraints=(_eq(WellKnownAttribute.DEVICE_TYPE, "tpu"), _eq(WellKnownAttribute.DEVICE_VARIANT, "v5e-4")),
                        pinned_backend=None,
                    )
                )
            elif kind == "h100_solo":
                jobs.append(
                    JobSpec(
                        job_id=jid, user=user, band=band, arrival=tick,
                        duration=rng.randint(3, 9), num_tasks=1, coscheduled=False, group_by=None,
                        variant="h100", cpu_millicores=8000, memory_bytes=32 * GIB, gpu_count=1, tpu_count=0,
                        constraints=(_eq(WellKnownAttribute.DEVICE_TYPE, "gpu"), _eq(WellKnownAttribute.DEVICE_VARIANT, "h100")),
                        pinned_backend=None,
                    )
                )
            elif kind == "h100_gang":
                jobs.append(
                    JobSpec(
                        job_id=jid, user=user, band=band, arrival=tick,
                        duration=rng.randint(4, 10), num_tasks=8, coscheduled=True, group_by="slice-group",
                        variant="h100", cpu_millicores=8000, memory_bytes=32 * GIB, gpu_count=1, tpu_count=0,
                        constraints=(_eq(WellKnownAttribute.DEVICE_TYPE, "gpu"), _eq(WellKnownAttribute.DEVICE_VARIANT, "h100")),
                        pinned_backend=None,
                    )
                )
            else:  # cpu (fungible; some region-pinned)
                cores = rng.choice([16000, 32000, 48000])
                region_pin = rng.random() < 0.4
                cons: tuple[Constraint, ...] = ()
                if region_pin:
                    region = rng.choice(["us-west4", "us-central1"])
                    cons = (_eq(WellKnownAttribute.REGION, region),)
                jobs.append(
                    JobSpec(
                        job_id=jid, user=user, band=band, arrival=tick,
                        duration=rng.randint(2, 7), num_tasks=1, coscheduled=False, group_by=None,
                        variant="cpu", cpu_millicores=cores, memory_bytes=rng.choice([16, 32, 64]) * GIB,
                        gpu_count=0, tpu_count=0, constraints=cons, pinned_backend=None,
                    )
                )
    return jobs


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth Poisson sampler (random.Random has no .poisson)."""
    import math

    el = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= el:
            return k - 1
