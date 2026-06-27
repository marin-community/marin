# Spike S1 — Recoverability of the agent's worker-state cache

## The question

The multi-backend design (`design.md`) moves the per-cluster agent's worker state
**out** of the root authoritative DB and into a volatile, in-memory cache, on the
strength of one governing invariant:

> A fact may live outside the root DB **iff** it is a pure function of
> `(worker re-registration ∪ list_all_slices() ∪ the root's Poll desired-set)`.

The cache covers: **worker roster, per-worker health, slice inventory,
attempt→worker binding, allocated host ports, local placement.**

This spike asks: is that actually true? Build a runnable harness that populates a
worker-state cache, **wipes** it, rebuilds it purely from those three sources, and
asserts functional equivalence — roster + active attempt→worker bindings +
allocated **ports** identical; health/failure counters may reset (the only allowed
diff). Surface any secretly-authoritative worker state that breaks it.

## What I built

Three new files, all under this directory; **no shared source was modified.**

| File | What it does |
|---|---|
| `agent_cache.py` | Models the agent's recoverable worker-state cache. Reuses the **real** `WorkerHealthTracker` (controller/worker_health.py) for health and the **real** `PortAllocator` (worker/port_allocator.py) for ports. Defines the three recovery sources and `rebuild_from_sources()` — the pure function under test. |
| `port_recovery.py` | Exercises the **real** `TaskAttempt.adopt()` + real `PortAllocator` + real `DiscoveredContainer` to prove the port hole concretely, then shows the fix direction (stamp + re-reserve). |
| `run_spike.py` | Populates a steady-state cache, wipes it, rebuilds it two ways (today's no-port-footprint semantics vs. the fix), and asserts. Prints a PASS/FAIL report; exit 0 iff the invariant holds with the fix and only health reset. |

Faithfulness choices: roster/slice/binding records are small dataclasses (in
production they are SQLAlchemy rows + the cloud `SliceHandle` protocol) — their
*content* is what the recoverability claim is about, not their storage engine.
Health and ports use the production classes unchanged. The one shared-source
change the fix needs — a `PortAllocator.reserve(ports)` method — is emulated in the
harness by reaching into the allocator's `_allocated` set, and flagged below.

### How the three sources map

- **(1) worker re-registration** → roster (`WorkerRegistration`: id, address,
  `slice_id`, `scale_group`, metadata) **and** the attempt→worker binding +
  allocated ports, recovered from the worker re-reporting its *running attempts*
  via substrate discovery (`DiscoveredAttempt`, the recoverability projection of
  `runtime.types.DiscoveredContainer`). Precision worth noting: the binding's real
  footprint is the **running container's labels**, surfaced through the worker's
  reconcile/discovery — **not** `RegisterRequest`, which carries identity only.
- **(2) `list_all_slices()`** → slice inventory (`ListedSliceLite`).
- **(3) the root's Poll desired-set** → the set of still-desired `attempt_uid`s;
  a discovered attempt absent from it is a **zombie** that fence-by-absence kills,
  so it is bound by neither the original nor the rebuilt cache.

## How to run

```bash
# from the repo root
.venv/bin/python .agents/projects/iris_multi_backend/spikes/S1_recoverability/run_spike.py

# or, from this directory
cd .agents/projects/iris_multi_backend/spikes/S1_recoverability && uv run python run_spike.py
```

No cluster, cloud, or network access; pure in-process. Exit code 0 = all checks pass.

## The result

**The invariant HOLDS — with caveats.** Verbatim from a run:

```
[A] Rebuild with TODAY's semantics (DiscoveredContainer has no ports):
  [PASS] roster recovered
  [PASS] slice inventory recovered
  [PASS] attempt->worker bindings recovered
  [PASS] zombie uid-Z fenced (absent from bindings)
  [PASS] PORT HOLE: ports NOT recovered (expected with today's code)
            rebuilt attempt_ports={'uid-1': (), 'uid-2': (), 'uid-3': ()} reserved={...all empty...}

[B] Rebuild with the FIX (ports stamped on substrate, re-reserved on adopt):
  [PASS] full functional signature identical
  [PASS] attempt_ports recovered exactly
  [PASS] reserved port sets recovered exactly
  [PASS] zombie uid-Z fenced (absent from bindings)
  [PASS] zombie ports NOT reserved on w3 (fence releases)
  [PASS] health counters reset (the ONLY allowed diff)  rebuilt all (0,0) vs original w0=(2,0) w1=(0,1)
  [PASS] all re-registered workers seeded live

[C] Port hole against the REAL TaskAttempt.adopt():
  adopt_today()        -> attempt.ports={} reserved=()
  adopt_with_recovery()-> attempt.ports={'p0': 30000, 'p1': 30001} reserved=(30000, 30001)
  [PASS] real adopt() loses ports (ports={} , allocator empty)
  [PASS] fix re-reserves the stamped ports in PortAllocator
```

- **Roster, slice inventory, and active attempt→worker bindings rebuild exactly**
  from the three sources, *today*, with no code change. Fence-by-absence correctly
  drops the running-but-undesired zombie.
- **Health counters reset to zero** on rebuild (original had `w0` cf=2, `w1`
  build_failures=1; rebuilt all zero) and every re-registered worker is seeded
  live — exactly the one allowed difference. So the rebuild loses no liveness
  *membership*, only the transient failure accounting (which then re-accrues from
  fresh observations). Sanctioned by the design.
- **Ports are the load-bearing caveat.** With today's code the rebuild silently
  loses every port reservation; with the fix it recovers them byte-for-byte.

## Secretly-authoritative worker state found

State that is **not** a pure function of the three sources (i.e. would silently
diverge or be lost on rebuild):

1. **Allocated host ports — the confirmed hole (must fix).** `DiscoveredContainer`
   (runtime/types.py:271) has **no `ports` field**, and `TaskAttempt.adopt()`
   (worker/task_attempt.py:303) rebuilds `ports={}` and never touches the
   `PortAllocator`. Demonstrated against the *real* classes in section [C]: a task
   holding `{30000, 30001}` comes back from `adopt()` with `ports={}` and an
   **empty** allocator — so the worker will re-hand those ports to the next task
   (double-allocation / bind clash). This is a latent bug **today**, independent of
   multi-backend (any worker restart with a port-using adopted task hits it). The
   stamp is dual-purpose: it serves both the **worker-restart** adopt path *and*
   the **agent-cache** rebuild — ports live in the worker's `PortAllocator`, not
   the controller, so without a substrate footprint neither layer can recover them.

2. **Service endpoints (`endpoints_table` / `EndpointsProjection`) — keep root-side,
   do NOT demote to agent cache.** Endpoints are registered once by a running task
   over RPC and stored in the **root DB** today (write-through cache). They are
   **not** a pure function of the three sources: a long-running task that keeps
   executing across an agent restart does **not** re-announce its endpoint, and it
   appears in neither `list_all_slices()` nor the Poll desired-set's substrate
   footprint. They are fine as long as they stay root-authoritative; they would
   break the invariant if moved into the agent's recoverable cache. The design's
   named cache set omits endpoints — this spike confirms that omission is load-bearing.

Allowed-to-reset (not violations, noted for completeness): per-worker health
counters and the autoscaler's per-slice consecutive-probe-failure counters
(same health class — they re-accrue from fresh observations); in-flight
*uncommitted* launches (a launch issued but whose container isn't yet
substrate-visible) are not in the recoverable set and rely on the launch-lease +
post-create re-check, not on cache rebuild.

## Implications for `spec.md`

1. **§5 substrate-stamping is mandatory, not optional, and needs a `PortAllocator`
   API.** The spec already says `adopt()` "must restore ports from the stamp into
   `PortAllocator` before scheduling new work." Make it concrete with the three
   code deltas this spike proved necessary and sufficient:
   - add a `ports` field to `runtime.types.DiscoveredContainer` (and the Docker /
     k8s discovery that populates it from the stamped label/annotation);
   - add `PortAllocator.reserve(ports: list[int])` — today only `allocate()` /
     `release()` exist, so there is **no way** to re-mark a known port set as taken;
   - have `TaskAttempt.adopt()` restore `attempt.ports` and call `reserve()`.
   Recommend filing this as a standalone bug independent of multi-backend, since it
   is a live double-allocation risk on every worker restart now.

2. **§5 "Agent-local cache" list should explicitly EXCLUDE service endpoints**, and
   the migration/spec should state endpoints remain root-authoritative (or are
   re-announced) — otherwise an agent restart silently drops live endpoints.

3. **Sharpen the §0/§5 wording on the binding's recovery footprint:** the
   attempt→worker binding recovers from the **substrate object's stamped labels
   surfaced via the worker's reconcile/discovery**, not from the registration
   handshake. This matters for the k8s path too (pod labels listed via the API)
   and reinforces why `attempt_uid` (+ ports) must be stamped on the substrate.

4. The recoverable set as named (roster, health, slices, binding, ports) is
   otherwise **correct**: roster/slices/binding rebuild exactly with zero new
   mechanism; only ports needed the stamp, and only health is allowed to reset.
