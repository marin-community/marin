# SPIKE S3 — Fence / Reuse Timing

**Question (design.md open question S3):** after a network partition, is safe
cross-attempt re-placement *seconds or minutes*? The skew-safe reuse invariant
(spec.md §1) requires `root_reuse_time > agent_self_fence_time`, where the root
waits `lease_duration + max_skew + transport_grace + kill_grace` before
re-placing a task on a new attempt. This spike pins **real magnitudes** for every
term.

**Bottom line:** with today's mechanisms, safe re-placement is **~10 minutes**
(dominated by the worker's 600 s lost-contact self-reset). "Seconds" (~20–30 s) is
achievable but is **not free** — it requires a *new* short dedicated lease plus a
faster worker self-fence, and on k8s a self-fence mechanism that **does not exist
today**. The irreducible floor is ~8–9 s (worker monitor poll + reconcile RPC
bound).

> Safety: per AGENTS.md, no live partition was injected. All numbers below come
> from driving **real iris code over in-memory fakes** (`offline_timing.py`) or
> are **read directly from source**. The two terms that genuinely need live
> infra (real worker self-fence over a real partition; k8s pod-create /
> pod-self-fence) are deferred to the gated `partition_harness.py` runs in §5.

---

## 1. The formula, every term defined

```
root_reuse_time  =  send_time
                 +  lease_duration      # the renewing window the runner obeys
                 +  max_skew            # clock skew between root clock and runner clock
                 +  transport_grace     # last renewing message in flight / reconcile round
                 +  kill_grace          # decide-to-kill  ->  old runner provably dead
```

The root starts its reuse clock at `send_time` (the moment it sends a Poll/
Reconcile response that *could* renew the lease — root-conservative; spec.md §1).
It may launch the next attempt only after `root_reuse_time`, by which point the
agent must have self-fenced the old runner.

| Term | What it is | Source of truth | Today's value |
|---|---|---|---|
| `lease_duration` | Window the runner keeps running without a renewing message; on expiry it self-fences. **No dedicated lease exists yet** — the closest live mechanism is the worker's lost-contact self-reset. | `worker.py:83 heartbeat_timeout` | **600.0 s** |
| `max_skew` | Skew between the root's reuse clock and the runner's self-fence clock. Leases are **monotonic-duration** (`timing.py:60`, `Deadline.from_seconds` → `time.monotonic`), so only clock-**rate** skew over the lease window matters, *not* absolute offset. | derived (NTP rate drift) | **< 30 ms** worst-case over 60 s; bound at **1.0 s** for margin |
| `transport_grace` | Time for the last renewing message to land / the reconcile round to finish. A partitioned worker costs exactly one such window per round. | `backends/rpc/backend.py:54 RECONCILE_RPC_TIMEOUT` | **3.0 s** |
| `kill_grace` | From the agent deciding to kill to the old runner being provably dead. | worker monitor poll + escalation (below) | **~5 s** (worker), **~30 s** (k8s graceful) |

---

## 2. Source-of-truth constants (read from code, not measured)

**Worker daemon** (`lib/iris/src/iris/cluster/worker/`):

| Constant | Value | Location | Role |
|---|---|---|---|
| `heartbeat_timeout` | **600.0 s** | `worker.py:83` | Worker self-resets (wipes containers) after this long with no controller contact. The only lost-contact self-fence today. |
| serve-loop poll | **1.0 s** | `worker.py:580` | Self-fence **detection granularity**. |
| `poll_interval` | **5.0 s** | `worker.py:82`, applied at `task_attempt.py:951` | Task monitor loop cadence — bounds how fast `should_stop` / container-exit is **observed**. |
| `kill` escalation | SIGTERM → wait `term_timeout_ms` → SIGKILL | `task_attempt.py:411` (`term_timeout_ms=5000`) | Graceful-kill window. |
| `_KILL_EXIT_WAIT_TIMEOUT` | **30.0 s** | `task_attempt.py:72` | Max wait for a wedged container to exit after SIGKILL. |

**Controller / RPC backend** (`lib/iris/src/iris/cluster/`):

| Constant | Value | Location | Role |
|---|---|---|---|
| `RECONCILE_RPC_TIMEOUT` | **3.0 s** | `backends/rpc/backend.py:54` | Per-worker reconcile RPC deadline (`_reconcile_one`, `:316`). |
| `RECONCILE_FANOUT_PARALLELISM` | 512 | `backends/rpc/backend.py:59` | Fleet reconciles in one wave → a partition costs **one** 3 s window per round, not N. |
| reconcile cadence | **1.0 s** | `controller.py:170` | Control-tick reconcile interval. |
| `worker_unreachable_grace` | **50.0 s** | `controller.py:178`, `worker_health.py:54` | Controller declares a worker dead after ≥50 s **and** ≥3 failures (`MIN_UNREACHABLE_FAILURES`, `worker_health.py:59`). |

**k8s backend** (`lib/iris/src/iris/cluster/backends/k8s/tasks.py`):

| Constant | Value | Location | Role |
|---|---|---|---|
| `activeDeadlineSeconds` | = task timeout (ms→s) | `tasks.py:796` | The **only** built-in pod self-kill. = the job's timeout (hours), **not** a lease. **SKIPPED for Kueue-gated gangs** (`tasks.py:795`). |
| `terminationGracePeriodSeconds` | **unset → k8s default 30 s** | (not in pod spec) | SIGTERM→SIGKILL window on delete. |
| `_GANG_GC_MAX_AGE_SECONDS` | 60 | `tasks.py:952` | Terminal gang pod retention before GC. |
| `_GC_INTERVAL_SECONDS` | 60 | `tasks.py:940` | Terminal-pod GC pass cadence. |
| `_POD_NOT_FOUND_GRACE_CYCLES` | 3 | `tasks.py:117` | Reconcile cycles a pod must be absent before "gone". |
| terminal force-delete | grace=0 | `tasks.py:1972` | Terminal gang pods are force-deleted. |

---

## 3. What was measured offline (real code, in-memory fakes)

`offline_timing.py` drives the real `Worker.handle_reconcile`,
`TaskAttempt.kill`, the task monitor loop, `Worker._serve`, and the real
`asyncio.wait_for(..., RECONCILE_RPC_TIMEOUT)` pattern over a fake container.
Wall-clock magnitudes are machine-dependent; the **granularity** they expose is
the finding. Representative run:

**M1 — `kill_grace`** (kill-initiated → attempt terminal):

```
cooperative    term=5000ms poll=0.1s  terminal=0.125s  sigterm@0.000s
wedged         term= 500ms poll=0.1s  terminal=0.128s  sigterm@0.000s
cooperative    term=5000ms poll=1.0s  terminal=1.060s  sigterm@0.000s
cooperative    term=5000ms poll=5.0s  terminal=5.000s  sigterm@0.000s   <- production poll
wedged         term=5000ms poll=5.0s  terminal=5.000s  sigterm@0.000s
slow-exit-1.5s term=5000ms poll=0.1s  terminal=2.096s  sigterm@0.000s
```

Finding: **SIGTERM is delivered within ~1 ms**, but *confirmed death* is bounded
by the **monitor poll_interval** (the loop reacts to `should_stop` only on its
next tick, `task_attempt.py:842/951`). At the production `poll_interval=5 s`,
`kill_grace ≈ 5 s` regardless of `term_timeout_ms`. A container that ignores
SIGKILL adds up to `_KILL_EXIT_WAIT_TIMEOUT=30 s`.

**M2 — reconcile zombie-kill** (server-side reconcile RTT + reconcile→kill):

```
poll=0.1s  handler_return=2.31ms  sigterm@0.001s  terminal@0.099s
poll=5.0s  handler_return=1.99ms  sigterm@0.000s  terminal@5.001s   <- production poll
```

Finding: the reconcile **handler itself is ~2 ms** (it dispatches the kill async,
`worker.py:999`); the visible death again tracks `poll_interval`. Server-side
reconcile processing is negligible — the network RTT (live-only, §5) is the real
cost.

**M3 — `transport_grace`** (RECONCILE_RPC_TIMEOUT enforcement):

```
RECONCILE_RPC_TIMEOUT (source)  3.0s
measured wait_for bound         3.003s  (timed_out=True)
```

Finding: a hung/partitioned worker is abandoned at exactly **3.0 s** per round.

**M4 — worker self-fence** (`_serve` return after lost contact):

```
heartbeat_timeout=0.1s  _serve returned @ 1.001s
heartbeat_timeout=1.5s  _serve returned @ 2.001s
heartbeat_timeout=2.5s  _serve returned @ 3.001s
PRODUCTION DEFAULT      heartbeat_timeout = 600.0s
```

Finding: self-fence latency = `heartbeat_timeout` rounded up to the next **1 s**
serve-loop poll, plus the reset/kill. At the production **600 s** this is the term
that makes re-placement *minutes*.

**M5 — formula evaluated:**

```
TODAY (heartbeat_timeout self-fence)  600 + 1 + 3 + 5 = 609s  (~10.2 min)
PROPOSED (20s dedicated lease)         20 + 1 + 3 + 5 =  29s
AGGRESSIVE (10s lease, 5s kill)        10 + 1 + 3 + 5 =  19s   (floor ~8-9s)
```

Run it:

```bash
.venv/bin/python offline_timing.py          # full
.venv/bin/python offline_timing.py --quick  # skip the slow 5s-poll cells
```

---

## 4. Interpretation — which terms dominate, and the gaps

1. **`lease_duration` dominates, by 100×.** There is no dedicated short lease
   today; the only lost-contact fence is the worker's **600 s** self-reset. So
   the skew-safe invariant, honestly applied, forces the root to wait **>10 min**
   before re-placing. To reach "seconds" the design must add a short dedicated
   agent↔worker lease (renewed every Poll) **and** make the worker self-fence on
   *that* lease, not on the 600 s heartbeat.

2. **k8s has no usable self-fence at all — the dominant risk.** A partitioned pod
   keeps running; `activeDeadlineSeconds` equals the *job timeout* (hours) and is
   **disabled for Kueue gangs** (`tasks.py:795`). So the k8s `lease_duration`
   term is effectively **infinite** unless a new mechanism is added (short
   `activeDeadlineSeconds = lease` for non-gang pods, a lease-checking sidecar /
   liveness probe, or accepting that k8s re-placement must wait for the agent to
   reconnect and delete). **This is the single most important thing the live k8s
   run must establish.**

3. **The floor is ~8–9 s**, set by `kill_grace` (worker monitor `poll_interval`,
   5 s) + `transport_grace` (3 s) + `max_skew` (~0). Lowering it means a tighter
   `poll_interval` (more CPU per worker) — a tunable, not free. `max_skew` is
   **negligible** because leases are monotonic-duration; clock offset is
   irrelevant, only rate drift over the lease window (<30 ms) counts.

---

## 5. Gated live-run plan (`partition_harness.py`)

Two terms cannot be measured offline: a **real** worker self-fence over a **real**
partition, and **k8s pod-create / pod-self-fence / pod-delete** latency against a
real apiserver. `partition_harness.py` holds the ready-to-run code. It is
**triple-gated**: it does nothing destructive without `--i-have-approval` **and**
`--confirm DESTROY-<target>` **and** an explicit target. With no flags it prints a
DRY-RUN (the exact commands, touching nothing). Run the dry-run, get sign-off,
then run gated **against a disposable test slice / scratch namespace only.**

### Run A — worker-daemon partition (needs user approval)

```bash
# safe preview:
python partition_harness.py --mode worker-daemon
# gated (test slice only):
python partition_harness.py --mode worker-daemon \
    --worker <test-vm> --zone <zone> --controller-ip <ip> \
    --heartbeat-timeout 20 --i-have-approval --confirm DESTROY-<test-vm>
```

- **Measures:** `self_fence_latency = t_selffence − t0` after dropping the
  worker→controller egress (`iptables -A OUTPUT -d <ctrl> -j DROP`), by polling
  until the running container vanishes; plus real reconcile RTT and re-register
  time on heal.
- **Expected:** `≈ heartbeat_timeout + ≤1 s + reset`. Launch the test worker with
  a short `heartbeat_timeout` (e.g. 20 s) so the run takes seconds and validates
  the *mechanism*; the production 600 s is then a config choice, not a mystery.
- **Why live:** confirms the container actually dies on self-reset (not just that
  `_serve` returns) and gives a real network RTT the offline harness can't.
- **Approval ask:** *"May I partition one disposable test-slice worker from the
  controller (reversible iptables rule, auto-healed) to time its self-fence? No
  production job, no cluster bounce."*

### Run B — k8s pod lifecycle + partition (needs user approval)

```bash
# safe preview:
python partition_harness.py --mode k8s-pod
# gated (scratch namespace only):
python partition_harness.py --mode k8s-pod \
    --namespace iris-spike --kubeconfig <test-kubeconfig> \
    --i-have-approval --confirm DESTROY-iris-spike
```

- **Measures:** (1) **pod-create latency** `kubectl apply → Running` (cold image
  pull vs warm — re-run to split); (2) **pod-self-fence** — whether a lease-less
  pod terminates on its own (it should *not*, except when
  `activeDeadlineSeconds` fires) and that deadline's granularity; (3) the
  **partition proxy** — cut the agent's apiserver access and confirm a RUNNING pod
  *survives* (proves the self-fence gap); (4) **pod-delete latency** graceful (30 s
  default grace) vs force (grace=0, the terminal-GC path).
- **Expected:** create ~seconds (warm) to tens of seconds (cold pull); self-fence
  via `activeDeadlineSeconds` only, at whole-second granularity; lease-less pod
  **survives** agent loss → confirms a new k8s self-fence is required; graceful
  delete bounded by `terminationGracePeriodSeconds` (~30 s), force ~1–2 s.
- **Why live:** k8s pod-create and apiserver behavior have no faithful offline
  model.
- **Approval ask:** *"May I create/delete a throwaway busybox pod in a scratch
  namespace and (manually) deny its agent's apiserver egress, to time pod create
  / self-fence / delete? Scratch namespace only, no Kueue gang, no production
  workload."*

---

## 6. Files

| File | What |
|---|---|
| `offline_timing.py` | Offline harness: M1–M5 over real iris code + fakes. Safe; no infra. |
| `partition_harness.py` | Triple-gated live partition harness (worker-daemon + k8s-pod). Dry-run by default. |
| `SPIKE.md` | This document. |
