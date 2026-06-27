# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""S1 spike runner: wipe the agent worker-state cache, rebuild from the three
sources, and assert functional equivalence.

Run:  uv run python -m run_spike      (from this directory)
  or  .venv/bin/python .agents/projects/iris_multi_backend/spikes/S1_recoverability/run_spike.py

Exit 0 iff the invariant holds with the port fix and only health resets differ.
"""

from __future__ import annotations

import sys

from agent_cache import (
    AgentWorkerCache,
    DiscoveredAttempt,
    ListedSliceLite,
    RecoverySources,
    WorkerRegistration,
    rebuild_from_sources,
)
from port_recovery import adopt_today, adopt_with_recovery

from iris.cluster.backends.types import CloudSliceState
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerHealthEventKind
from iris.cluster.types import AttemptUid, WorkerId
from iris.rpc import job_pb2

NOW_MS = 1_900_000_000_000

# Workers: two slices, two scale groups.
_WORKERS = [
    # worker_id, slice_id, scale_group, tpu variant
    ("w0", "slice-A", "g1", "v5e-4"),
    ("w1", "slice-A", "g1", "v5e-4"),
    ("w2", "slice-B", "g2", "v5p-8"),
    ("w3", "slice-B", "g2", "v5p-8"),
]

# Desired + running attempts (steady state): uid -> (worker, num_ports).
_ATTEMPTS = [
    ("uid-1", "w0", 1),
    ("uid-2", "w1", 2),
    ("uid-3", "w2", 1),
]

_DESIRED = frozenset(AttemptUid(u) for u, _, _ in _ATTEMPTS)
_SCOPE = _DESIRED  # bindings/ports equivalence is scoped to the desired-set


def _metadata(variant: str, host: str) -> job_pb2.WorkerMetadata:
    md = job_pb2.WorkerMetadata(hostname=host, ip_address=f"10.0.0.{host[-1]}", cpu_count=8, memory_bytes=1 << 35)
    md.device.tpu.variant = variant
    return md


def _registrations() -> list[WorkerRegistration]:
    return [
        WorkerRegistration(
            worker_id=WorkerId(wid),
            address=f"10.0.0.{wid[-1]}:9000",
            slice_id=sid,
            scale_group=sg,
            metadata=_metadata(variant, wid),
        )
        for wid, sid, sg, variant in _WORKERS
    ]


def _listed_slices() -> list[ListedSliceLite]:
    return [
        ListedSliceLite("slice-A", "g1", "us-west4-a", CloudSliceState.READY, frozenset({WorkerId("w0"), WorkerId("w1")})),
        ListedSliceLite("slice-B", "g2", "us-west4-b", CloudSliceState.READY, frozenset({WorkerId("w2"), WorkerId("w3")})),
    ]


def build_original() -> tuple[AgentWorkerCache, dict[AttemptUid, dict[str, int]]]:
    """Populate a steady-state cache; return it plus the per-attempt stamped ports.

    The stamped ports are the exact host ports the original allocated -- the
    substrate footprint the agent must recover on rebuild.
    """
    cache = AgentWorkerCache()
    regs = _registrations()
    for reg in regs:
        cache.add_worker(reg, now_ms=NOW_MS)
    for listed in _listed_slices():
        cache.add_slice(listed)

    stamped: dict[AttemptUid, dict[str, int]] = {}
    for uid, wid, n in _ATTEMPTS:
        ports = cache.port_allocators[WorkerId(wid)].allocate(n)
        cache.bind_attempt(AttemptUid(uid), WorkerId(wid), ports)
        stamped[AttemptUid(uid)] = {f"p{i}": p for i, p in enumerate(ports)}

    # Make health non-trivial so a reset is observable as the only allowed diff.
    cache.health.apply(
        [
            WorkerHealthEvent(WorkerId("w0"), WorkerHealthEventKind.UNREACHABLE),
            WorkerHealthEvent(WorkerId("w0"), WorkerHealthEventKind.UNREACHABLE),
            WorkerHealthEvent(WorkerId("w1"), WorkerHealthEventKind.BUILD_FAILED),
        ],
        now_ms=NOW_MS + 1000,
    )
    return cache, stamped


def _sources(stamped: dict[AttemptUid, dict[str, int]], *, with_ports: bool) -> RecoverySources:
    """Build the three recovery sources, injecting a zombie discovery (uid-Z on
    w3, running but NOT desired) to exercise fence-by-absence."""
    discoveries = [
        DiscoveredAttempt(
            attempt_uid=AttemptUid(uid),
            worker_id=WorkerId(wid),
            ports=stamped[AttemptUid(uid)] if with_ports else {},
        )
        for uid, wid, _ in _ATTEMPTS
    ]
    discoveries.append(DiscoveredAttempt(attempt_uid=AttemptUid("uid-Z"), worker_id=WorkerId("w3"), ports={"p0": 39999}))
    return RecoverySources(
        registrations=_registrations(),
        discoveries=discoveries,
        slices=_listed_slices(),
        desired_uids=_DESIRED,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class _Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f"  -- {detail}"
        print(line)
        if not ok:
            self.failures.append(name)


def _diff_signatures(a: dict, b: dict) -> list[str]:
    diffs = []
    for key in a:
        if a[key] != b[key]:
            diffs.append(f"{key}: {a[key]} != {b[key]}")
    return diffs


def main() -> int:
    rep = _Report()
    print("=" * 78)
    print("S1 recoverability spike: agent worker-state cache wipe + rebuild")
    print("=" * 78)

    original, stamped = build_original()
    orig_sig = original.signature(_SCOPE)
    orig_health = original.health_snapshot()
    print("\nOriginal steady-state cache:")
    print(f"  roster        : {sorted(original.roster)}")
    print(f"  slices        : {sorted(original.slices)}")
    print(f"  bindings      : { {u: original.bindings[u] for u in sorted(original.bindings)} }")
    print(f"  attempt_ports : { {u: original.attempt_ports[u] for u in sorted(original.attempt_ports)} }")
    print(f"  reserved ports: { {w: sorted(a._allocated) for w, a in sorted(original.port_allocators.items())} }")
    print(f"  health (cf,bf): {orig_health}")

    # ---- WIPE ----
    print("\n--- WIPE: agent restarts, cache is gone ---")
    del original

    # ---- Rebuild with TODAY's adopt (no port footprint) ----
    print("\n[A] Rebuild with TODAY's semantics (DiscoveredContainer has no ports):")
    today = rebuild_from_sources(_sources(stamped, with_ports=False), now_ms=NOW_MS + 60_000, recover_ports=False)
    today_sig = today.signature(_SCOPE)
    rep.check("roster recovered", today_sig["roster"] == orig_sig["roster"])
    rep.check("slice inventory recovered", today_sig["slices"] == orig_sig["slices"])
    rep.check("attempt->worker bindings recovered", today_sig["bindings"] == orig_sig["bindings"])
    rep.check("zombie uid-Z fenced (absent from bindings)", AttemptUid("uid-Z") not in today.bindings)
    ports_lost = today_sig["attempt_ports"] != orig_sig["attempt_ports"]
    rep.check(
        "PORT HOLE: ports NOT recovered (expected with today's code)",
        ports_lost,
        f"rebuilt attempt_ports={today_sig['attempt_ports']} reserved={today_sig['reserved_ports']}",
    )

    # ---- Rebuild with the FIX (ports stamped + re-reserved) ----
    print("\n[B] Rebuild with the FIX (ports stamped on substrate, re-reserved on adopt):")
    fixed = rebuild_from_sources(_sources(stamped, with_ports=True), now_ms=NOW_MS + 60_000, recover_ports=True)
    fixed_sig = fixed.signature(_SCOPE)
    fixed_health = fixed.health_snapshot()
    diffs = _diff_signatures(orig_sig, fixed_sig)
    rep.check("full functional signature identical", not diffs, "; ".join(diffs))
    rep.check("attempt_ports recovered exactly", fixed_sig["attempt_ports"] == orig_sig["attempt_ports"])
    rep.check("reserved port sets recovered exactly", fixed_sig["reserved_ports"] == orig_sig["reserved_ports"])
    rep.check("zombie uid-Z fenced (absent from bindings)", AttemptUid("uid-Z") not in fixed.bindings)
    rep.check(
        "zombie ports NOT reserved on w3 (fence releases)",
        fixed.port_allocators[WorkerId("w3")]._allocated == set(),
    )

    # ---- Health is the ONLY allowed difference ----
    health_reset = all(cf == 0 and bf == 0 for cf, bf in fixed_health.values())
    rep.check(
        "health counters reset (the ONLY allowed diff)",
        health_reset and fixed_health != orig_health,
        f"rebuilt={fixed_health} vs original={orig_health}",
    )
    all_live = all(fixed.health.liveness(WorkerId(wid)).usability.name in {"HEALTHY", "DEGRADED"} for wid, *_ in _WORKERS)
    rep.check("all re-registered workers seeded live", all_live)

    # ---- Concrete port hole against the REAL adopt() ----
    print("\n[C] Port hole against the REAL TaskAttempt.adopt():")
    real_stamp = {"p0": 30000, "p1": 30001}
    today_adopt = adopt_today(real_stamp)
    fixed_adopt = adopt_with_recovery(real_stamp)
    print(f"  adopt_today()        -> attempt.ports={today_adopt.attempt_ports} reserved={today_adopt.allocator_reserved}")
    print(f"  adopt_with_recovery()-> attempt.ports={fixed_adopt.attempt_ports} reserved={fixed_adopt.allocator_reserved}")
    rep.check(
        "real adopt() loses ports (ports={} , allocator empty)",
        today_adopt.attempt_ports == {} and today_adopt.allocator_reserved == (),
    )
    rep.check(
        "fix re-reserves the stamped ports in PortAllocator",
        fixed_adopt.attempt_ports == real_stamp and fixed_adopt.allocator_reserved == (30000, 30001),
    )

    print("\n" + "=" * 78)
    if rep.failures:
        print(f"VERDICT: {len(rep.failures)} check(s) FAILED: {rep.failures}")
        return 1
    print("VERDICT: invariant HOLDS WITH the port fix; health reset is the only diff.")
    print("         WITHOUT the fix, port reservations are lost (latent double-alloc).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
