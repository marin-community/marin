# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LoadtestGcpService: a wrapper around InMemoryGcpService with richer
failure injection.

The production :class:`iris.cluster.providers.gcp.fake.InMemoryGcpService`
only supports *one-shot* failure injection and does not model per-group
targeting, durations, or latency. The 2026-04-18 spike is characterised by a
whole zone's ``tpu_create`` hanging for ~120 s and then failing for an hour;
reproducing that needs all three.

Rather than thread those concerns into the production fake (which is used by
hundreds of unit tests), this wrapper lives in ``tests/loadtest/``. It
delegates every method call to the inner service and only intercepts
operations configured via :meth:`configure_failure`.

Matching is done on the ``request.name`` (for ``tpu_create``) or ``name``
(for ``tpu_delete``). Since the autoscaler derives slice names from the
group name (e.g. ``iris-marin-tpu_v6e-preemptible_8-europe-west4-a-...``),
a simple substring match on the scale-group fragment is sufficient to target
all slices belonging to a group.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal

from iris.cluster.providers.gcp.service import TpuCreateRequest
from iris.cluster.providers.types import InfraError, QuotaExhaustedError

from iris.loadtest.synthetic_worker import SyntheticWorkerPool

logger = logging.getLogger(__name__)

FailureMode = Literal["timeout", "internal_error", "quota"]


@dataclass
class FailureRule:
    """One active failure-injection rule.

    Matches on a regex against the TPU/slice name. Applied to every call to
    the configured operation until ``expires_at`` (monotonic seconds) passes.
    ``latency_seconds`` is slept before the failure fires — this is what
    actually burns threads in the autoscaler's scale-up pool.
    """

    operation: str
    name_pattern: re.Pattern[str]
    failure_mode: FailureMode
    expires_at: float
    latency_seconds: float


# Maximum allowed latency-sleep. The Autoscaler thread container's stop() uses
# a short timeout; sleeps longer than this leak a thread past teardown. Tests
# should use <=30 s where possible.
MAX_LATENCY_SECONDS = 120.0


class LoadtestGcpService:
    """Wrapping GcpService that delegates to an inner ``InMemoryGcpService``.

    Exposes every method used by :class:`GcpWorkerProvider` on the scale-up
    path by forwarding to the inner instance, plus :meth:`configure_failure`
    / :meth:`clear_failures` for test control.
    """

    def __init__(self, inner: Any, *, tpu_create_timeout_seconds: float | None = None) -> None:
        """Wrap *inner* with optional failure injection.

        Args:
            inner: A real or fake ``GcpService`` instance to delegate to.
            tpu_create_timeout_seconds: If set, ``tpu_create`` enforces this
                deadline against the injected latency; an injected sleep that
                exceeds the deadline is cut short and an InfraError("timed out")
                is raised.
        """
        self._inner = inner
        self._rules: list[FailureRule] = []
        self._rules_lock = threading.Lock()
        self._tpu_create_timeout_seconds = tpu_create_timeout_seconds
        # Observable counters for assertions in stimuli tests.
        self.counts_create_attempts: int = 0
        self.counts_create_failures: int = 0
        # Optional synthetic-worker pool. When set (via attach_worker_pool),
        # each successful tpu_create spawns a SyntheticWorker; each tpu_delete
        # stops the corresponding worker.
        self._worker_pool: SyntheticWorkerPool | None = None

    # -- configuration -------------------------------------------------------

    def configure_failure(
        self,
        *,
        operation: str,
        name_regex: str,
        failure_mode: FailureMode,
        duration_seconds: float,
        latency_seconds: float = 0.0,
    ) -> None:
        """Install a failure rule. See :class:`FailureRule`."""
        if latency_seconds < 0:
            raise ValueError(f"latency_seconds must be >= 0, got {latency_seconds}")
        latency_seconds = min(latency_seconds, MAX_LATENCY_SECONDS)

        rule = FailureRule(
            operation=operation,
            name_pattern=re.compile(name_regex),
            failure_mode=failure_mode,
            expires_at=time.monotonic() + duration_seconds,
            latency_seconds=latency_seconds,
        )
        with self._rules_lock:
            self._rules.append(rule)

    def clear_failures(self) -> None:
        with self._rules_lock:
            self._rules.clear()

    def active_rules(self) -> list[FailureRule]:
        with self._rules_lock:
            now = time.monotonic()
            return [r for r in self._rules if r.expires_at > now]

    # -- interception --------------------------------------------------------

    def _maybe_inject(self, operation: str, name: str, deadline_seconds: float | None = None) -> None:
        with self._rules_lock:
            now = time.monotonic()
            self._rules = [r for r in self._rules if r.expires_at > now]
            rules = [r for r in self._rules if r.operation == operation and r.name_pattern.search(name)]
        if not rules:
            return
        rule = rules[0]
        if rule.latency_seconds > 0:
            # Blocking sleep. This is the whole point — we want autoscaler
            # scale-up threads to pile up on slow TPU API calls.
            sleep_for = rule.latency_seconds
            timed_out = False
            if deadline_seconds is not None and sleep_for > deadline_seconds:
                sleep_for = deadline_seconds
                timed_out = True
            time.sleep(sleep_for)
            if timed_out:
                raise InfraError(
                    f"LoadtestGcpService: tpu_create timed out after {deadline_seconds:.1f}s "
                    f"(injected latency={rule.latency_seconds:.1f}s on {operation}({name!r}))"
                )
        if rule.failure_mode == "quota":
            raise QuotaExhaustedError(f"LoadtestGcpService: injected quota on {operation}({name!r})")
        if rule.failure_mode == "timeout":
            raise InfraError(f"LoadtestGcpService: injected timeout on {operation}({name!r})")
        if rule.failure_mode == "internal_error":
            raise InfraError(f"LoadtestGcpService: injected internal_error on {operation}({name!r})")
        raise ValueError(f"Unknown failure_mode: {rule.failure_mode}")

    # -- operations ----------------------------------------------------------

    def attach_worker_pool(self, pool: SyntheticWorkerPool, *, scale_group_lookup) -> None:
        """Enable synthetic workers for every successful tpu_create.

        Args:
            pool: Worker pool that owns the SyntheticWorker threads.
            scale_group_lookup: Callable ``name -> (scale_group, device_variant)``
                that maps a slice/TPU name back to the scale group it was
                created under. The name-mangling is done by the GCP provider
                (``_build_gce_resource_name``), so we need an explicit lookup
                rather than parsing.
        """
        self._worker_pool = pool
        self._scale_group_lookup = scale_group_lookup

    def tpu_create(self, request: TpuCreateRequest):
        self.counts_create_attempts += 1
        try:
            self._maybe_inject("tpu_create", request.name, deadline_seconds=self._tpu_create_timeout_seconds)
            result = self._inner.tpu_create(request)
        except Exception:
            self.counts_create_failures += 1
            raise
        # InMemoryGcpService leaves newly-created TPUs in state="CREATING";
        # the autoscaler's worker_registry maps anything other than RUNNING to
        # VM_STATE_BOOTING, so without this the dashboard shows every synthetic
        # worker stuck in "booting" forever. Flip to READY so handle.status()
        # reports CloudWorkerState.RUNNING.
        self._inner.advance_tpu_state(request.name, request.zone, "READY")
        # InMemoryGcpService leaves TPUs in state="CREATING"; the autoscaler's
        # worker_registry maps anything other than RUNNING to VM_STATE_BOOTING,
        # so without this flip the dashboard shows every synthetic worker stuck
        # in "booting" and the scale group never reports as ready.
        self._inner.advance_tpu_state(request.name, request.zone, "READY")
        if self._worker_pool is not None:
            scale_group, device_variant = self._scale_group_lookup(request.name, request.accelerator_type)
            self._worker_pool.spawn_for_slice(
                slice_id=request.name,
                scale_group=scale_group,
                zone=request.zone,
                device_variant=device_variant,
            )
        return result

    def tpu_delete(self, name: str, zone: str) -> None:
        self._maybe_inject("tpu_delete", name)
        if self._worker_pool is not None:
            self._worker_pool.stop_for_slice(name)
        self._inner.tpu_delete(name, zone)

    # Methods without injection hooks just delegate. We keep them explicit so
    # callers and static tools see the surface, instead of a catch-all
    # __getattr__ that hides typos.

    def tpu_describe(self, name: str, zone: str):
        return self._inner.tpu_describe(name, zone)

    def tpu_list(self, zones, labels=None):
        return self._inner.tpu_list(zones, labels)

    def queued_resource_create(self, request: TpuCreateRequest) -> None:
        self._maybe_inject("queued_resource_create", request.name)
        self._inner.queued_resource_create(request)

    def queued_resource_describe(self, name: str, zone: str):
        return self._inner.queued_resource_describe(name, zone)

    def queued_resource_delete(self, name: str, zone: str) -> None:
        self._inner.queued_resource_delete(name, zone)

    def queued_resource_list(self, zones, labels=None):
        return self._inner.queued_resource_list(zones, labels)

    def vm_create(self, request):
        return self._inner.vm_create(request)

    def vm_delete(self, name: str, zone: str, *, wait: bool = False) -> None:
        self._inner.vm_delete(name, zone, wait=wait)

    def vm_reset(self, name: str, zone: str) -> None:
        self._inner.vm_reset(name, zone)

    def vm_describe(self, name: str, zone: str):
        return self._inner.vm_describe(name, zone)

    def vm_list(self, zones, labels=None):
        return self._inner.vm_list(zones, labels)

    def vm_update_labels(self, name: str, zone: str, labels) -> None:
        self._inner.vm_update_labels(name, zone, labels)

    def vm_set_metadata(self, name: str, zone: str, metadata) -> None:
        self._inner.vm_set_metadata(name, zone, metadata)

    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str:
        return self._inner.vm_get_serial_port_output(name, zone, start)

    def logging_read(self, filter_str: str, limit: int = 200):
        return self._inner.logging_read(filter_str, limit)

    def shutdown(self) -> None:
        self._inner.shutdown()

    @property
    def mode(self):
        return self._inner.mode

    @property
    def project_id(self) -> str:
        return self._inner.project_id

    # -- test-control passthroughs -------------------------------------------

    def delete_tpu(self, name: str, zone: str) -> None:
        """Forcibly remove an in-memory TPU without going through the wrapper
        (so failure injection does not fire). Used by the ``preempt_workers``
        stimulus to simulate "GCP yanked the TPU".
        """
        self._inner._tpus.pop((name, zone), None)
        if self._worker_pool is not None:
            self._worker_pool.stop_for_slice(name)

    def list_all_tpus(self):
        return list(self._inner._tpus.items())
