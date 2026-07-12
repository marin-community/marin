# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-tick availability-attribute injection (policy.py).

Pure helpers — no DB, no scheduler — that fold a zone's accelerator-capability
markers onto workers and pick out which variants a tick actually needs to inject
so the per-worker copy is confined to workers in a zone that provisions a
demanded accelerator.
"""

from iris.cluster.constraints import (
    AttributeValue,
    WellKnownAttribute,
    availability_constraint,
    availability_key,
    region_constraint,
)
from iris.cluster.controller.codec import constraints_to_json
from iris.cluster.controller.scheduling.policy import (
    demanded_availability_variants,
    enrich_workers_with_availability,
)
from iris.cluster.controller.scheduling.scheduler import WorkerSnapshot
from iris.cluster.types import JobName, PendingTask, WorkerId
from rigging.timing import Timestamp


def _worker(worker_id: str, zone: str | None) -> WorkerSnapshot:
    attrs: dict[str, AttributeValue] = {"pool": AttributeValue("default")}
    if zone is not None:
        attrs[WellKnownAttribute.ZONE] = AttributeValue(zone)
    return WorkerSnapshot(
        worker_id=WorkerId(worker_id),
        total_cpu_millicores=64_000,
        total_memory_bytes=1024,
        total_gpu_count=0,
        total_tpu_count=0,
        committed_cpu_millicores=0,
        committed_memory_bytes=0,
        committed_gpu_count=0,
        committed_tpu_count=0,
        attributes=attrs,
    )


def _pending(constraints_json: str | None) -> PendingTask:
    """A PendingTask carrying only the field ``demanded_availability_variants`` reads."""
    job_id = JobName.root("u", "j")
    return PendingTask(
        task_id=job_id.task(0),
        job_id=job_id,
        backend_id="default",
        state=0,
        current_attempt_id=0,
        max_retries_failure=0,
        max_retries_preemption=0,
        submitted_at_ms=Timestamp.from_ms(0),
        priority_band=0,
        priority_neg_depth=0,
        priority_root_submitted_ms=0,
        priority_insertion=0,
        job_state=0,
        scheduling_deadline_epoch_ms=None,
        scheduling_timeout_ms=None,
        has_coscheduling=False,
        coscheduling_group_by=None,
        constraints_json=constraints_json,
        res_cpu_millicores=0,
        res_memory_bytes=0,
        res_disk_bytes=0,
        res_device_json=None,
    )


def test_enrich_adds_zone_availability_markers():
    workers = [_worker("w1", "us-central1-a"), _worker("w2", "us-east5-b")]
    enriched = enrich_workers_with_availability(workers, {"us-central1-a": frozenset({"v5p-8"})})
    by_id = {str(w.worker_id): w for w in enriched}
    assert by_id["w1"].attributes[availability_key("v5p-8")] == AttributeValue("true")
    # A worker whose zone is not in the (pruned) capability map is left untouched.
    assert availability_key("v5p-8") not in by_id["w2"].attributes


def test_enrich_passes_through_workers_without_zone_without_copying():
    workers = [_worker("w1", None)]
    enriched = enrich_workers_with_availability(workers, {"us-central1-a": frozenset({"v5p-8"})})
    # No zone -> no markers -> the original object is reused (no per-worker copy).
    assert enriched[0] is workers[0]


def test_enrich_empty_capabilities_returns_input_unchanged():
    workers = [_worker("w1", "us-central1-a")]
    # The demand-pruned map can be empty when nothing is reserved this tick; the
    # whole pass must then be a no-op that avoids rebuilding the worker list.
    assert enrich_workers_with_availability(workers, {}) is workers


def test_demanded_variants_empty_without_availability_constraints():
    rows = [
        _pending(None),
        _pending("[]"),
        _pending(constraints_to_json([region_constraint(["us-central1"]).to_proto()])),
    ]
    assert demanded_availability_variants(rows) == set()


def test_demanded_variants_collects_lowercased_keys_across_tasks():
    with_tpu = constraints_to_json(
        [
            availability_constraint("v5p-8").to_proto(),
            region_constraint(["us-central1"]).to_proto(),
        ]
    )
    with_gpu = constraints_to_json([availability_constraint("H100").to_proto()])
    # "H100" lowercases to "h100" to match the zone_capabilities map.
    assert demanded_availability_variants([_pending(with_tpu), _pending(with_gpu)]) == {"v5p-8", "h100"}
