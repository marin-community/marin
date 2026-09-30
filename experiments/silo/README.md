# silo

Sandboxes as nested containers inside Iris jobs: a Daytona replacement whose
quota we own, whose native snapshot format is a pinned OCI digest, and whose
isolation claims are proven rather than configured.

Requirements: `capability_env_gen/docs/infra/iris-sandbox-provider-brief.md`.
The capability pipeline's binding constraint was Daytona's 40-snapshot org
quota; here snapshots have no quota and sandbox capacity is a number the broker
reports.

## Shape

```
pipeline worker ──(dt.py / SiloClient)──> BROKER (1 Iris job)          control plane
      │                                     snapshots, placement, /capacity
      │  exec, sessions, files (direct, per-sandbox capability)
      ▼
    HOST (N privileged Iris jobs) ── containerd + runsc ──> SANDBOX       data plane
                                                            --network none
                                                            cgroup cpu.max/memory.max
                                                            cpuset (nproc == cpu)
```

* **Sandbox**: a nested container started from a pinned OCI digest, with no
  network interface, its own cgroup limits, and none of the host's mounts. Its
  PID 1 is a keepalive; work arrives by exec, as on Daytona.
* **Snapshot**: `{name, recipe bytes, resolved image, resource profile}`. Names
  use the pipeline's scheme byte for byte
  (`sha256(recipe + profile.cache_bytes())[:20]`). A `FROM`-only recipe needs no
  build and is active in about a second; a recipe with `RUN` steps is built by
  buildkit on a host (with network, at build time only) and cached there.
* **Broker**: control plane only. It hands back the host address and a
  per-sandbox capability (`HMAC(host_secret, sandbox_id)`), so exec output never
  flows through it.
* **Host**: a privileged Iris job (`CONTAINER_PROFILE_PRIVILEGED`, which Iris
  documents for exactly this). It runs on the default task image and fetches
  pinned, checksum-verified binaries at start (nerdctl-full, runsc, static
  busybox).

## Where the resource profile binds

At snapshot creation, exactly as on Daytona: `CreateSnapshotParams.resources`
(cpu cores, memory GB, disk GB) becomes part of the snapshot's identity and is
applied to every sandbox started from it: `cpu.max` quota, a cpuset of the same
size (so `nproc` reports the allotment, not the node's 192 cores), and a hard
`memory.max` with no swap. Disk is accounted against the host's budget but not
enforced per sandbox (the pipeline does not read disk back).

## Egress: how it is enforced, and how it is proven

Enforced by construction: sandboxes are created with `--network none`, i.e. a
network namespace with no interface except loopback (under gVisor, none at
all). There is no code path that creates a networked sandbox; a request for one
(`network_block_all=False`, an allow-list, `dt sandbox create` without
`--no-network`) is refused with an error saying what to do instead.

Proven, not inferred: `silo.acceptance` runs the same stdlib probe (DNS, raw TCP
to an IP, HTTPS by name) inside each sandbox and in its own process. The outside
run must reach all three (positive control); the inside run must be blocked on
all three. The Phase 0 spike did the same with the variable isolated to the
`--network` flag alone (`spike/receipts/phase0b-spike-03.json`).

## Inner runtime: gVisor by default

Measured on cw-us-east-02a (`spike/receipts/`): gVisor costs ~9% on a
syscall-heavy build step and nothing on CPU-bound work, starts in ~0.6 s, and
keeps the cgroup limits the pipeline gates on finite (it presents an emulated
cgroup v1 view; `memory.max_usage_in_bytes` is absent, so `memory_peak` reads
unknown). `runc` remains available per sandbox. The dedicated
`containerd-shim-runsc-v1` hangs when nested; gVisor runs through the runc shim
with `runsc` as the OCI binary.

## The ceiling

`GET /capacity` on the broker (or `SiloClient.capacity()`), per host and total.
A host's slots for a profile are
`min(cpu_budget * oversubscribe / cpu, memory_budget / memory, disk_budget / disk)`.
At honest CPU accounting (oversubscribe 1.0) a 96-cpu / 384 GB host holds 24
candidate (4 cpu / 8 GB) sandboxes; at 2.0, 48 (then memory binds). Each sandbox
keeps its own hard `cpu.max` either way. When every host is full, a create
waits (logging `AT CAPACITY`) and then fails with a 429 the pipeline's retry
already handles, rather than stalling silently.

## Measured on cw-us-east-02a (receipts in `spike/receipts/`)

| What | Result | Receipt |
|---|---|---|
| Acceptance, both runtimes, both profiles | 51/51: identity, exit codes, telemetry, isolation, async deletion, RUN-recipe build used offline | `acceptance-silo3-03.json` |
| Isolation | DNS, TCP-to-IP and HTTPS all blocked inside; all reached by the same probe outside | same |
| Sandbox create (warm) | ~0.4 s | same |
| Width: 500 trials, 250 concurrent, 90 s of CPU-bound build each | 500/500 ok, peak 250 in flight, 500/500 telemetry finite, 0 errors; create p50 2.6 s, p90 10.9 s under the burst | `width-silo3-01.json` |
| gVisor vs runc | +9% on a syscall-heavy step, +0% CPU-bound | `phase0b-spike-03.json` |

The width run used 6 hosts of 96 cpu / 384 GB at 2x CPU oversubscription: a
288-sandbox ceiling for the candidate profile, reported by `/capacity` before the
run started. What bounds width is how many hosts are up, not a quota.

## Two things the runtime gets wrong unless told

* **Exit codes.** `nerdctl exec` (2.1.2) exits 1 for any non-zero exit and
  appends `level=fatal msg="exec failed with exit code N"` to the command's
  stderr. The pipeline reads both (124 means "timed out"), so commands run
  through containerd's own `ctr task exec`, which reports the code exactly and
  adds nothing (`spike/receipts/phase0c-spike-02.txt`). Acceptance checks it on
  every run.
* **Waiting at capacity.** A create that waits for room must not hold a request
  thread: an early broker slept inside its 40-thread pool and, under a burst,
  queued the deletes that would have freed room. Creates now wait on an async
  sleep, and the broker refreshes hosts directly while they wait instead of
  trusting a 10 s-old heartbeat.

## Operating it

```bash
uv run python -m silo.launch --name cap up --cluster cw-us-east-02a --hosts 12 --host-cpu 32 --host-memory-gb 128
uv run python -m silo.launch status
uv run python -m silo.launch acceptance --image docker.io/library/python@sha256:...
uv run python -m silo.launch worker-env     # SILO_API_TOKEN, SILO_BROKER_RESOLVE_URL
uv run python -m silo.launch down
```

Workers need `SILO_API_TOKEN` and `SILO_BROKER_RESOLVE_URL`. Prefer
`worker-env --out <file>`, which writes them to a mode-0600 file and keeps the
token off stdout and out of transcripts. The resolve URL reaches the broker by
name through the Iris controller's proxy, so a broker restart does not strand
workers holding an old address.

`down` retires the secrets, so rotating after a leak is `down` then `up`.

Size hosts to pack. Whole-node asks (96 cpu) can sit pending on a busy CPU pool,
where 12 hosts of 32 cpu give the same ~192 candidate slots at 2x and schedule
at once.

### Liveness, quarantine, restarts

* A host silent for `SILO_HOST_SUSPECT_AFTER_SECONDS` (60) is **suspect**: no new
  placements, and its sandboxes answer a retryable 503, never "not found". Only
  `SILO_HOST_DEAD_AFTER_SECONDS` (600) of silence, or a refused connection, makes
  it **dead**. Both windows are raised to at least 2x the heartbeat cycle
  (`SILO_HEARTBEAT_SECONDS` 10 + `SILO_HEARTBEAT_TIMEOUT_SECONDS` 15), which
  broker and hosts read alike.
* Heartbeats run on their own thread pool; host calls run on `host_io` and
  `create` pools, and one host holds at most `SILO_HOST_MAX_INFLIGHT_CREATES` (8)
  creates. On 2026-09-29 heartbeats shared one pool with creates hung on sick
  hosts, and every host looked dead.
* A host that keeps failing creates like a sick host (timeouts), or reports an
  impossible allocation, is **quarantined** for placement and logged as
  `QUARANTINED`. `/capacity` lists health and quarantine per host, and every
  `AT CAPACITY` line counts suspect, dead and quarantined hosts.
* Pass `up --state-url <url>` so the broker persists snapshots and the host
  registry (an `s3://` URL works from CoreWeave pods as is). To replace a broker:
  `export-state` (only if the old one ran without a state URL), cancel it, then
  `up --hosts 0 --broker-job-name <new job name> --state-url <url>`. Hosts and
  workers re-find it by endpoint name.

## Swapping it into the pipeline

`uv run python -m silo.dt_shim --tools-src <daytona tools dir> --tools-out <dir>`
emits a drop-in `dt.py`: the original file (same CLI, JSON and call log) with
`client()` pointed at silo, the 40-snapshot budget wait disabled, and the silo
client embedded, since `dt.py` travels as one hash-pinned file.

## One deployment per cluster

A deployment (broker, hosts, and the workers that use it) lives on one cluster.
Workers talk to hosts directly, and CoreWeave clusters cannot reach each other's
node addresses: a TCP probe from 02a to rno2a and 08a, and from rno2a to 02a and
08a, was blocked every time while same-cluster controls connected. To use more
than one cluster, run one deployment per cluster (`--name ... --cluster cw-rno2a`)
and point each cluster's workers at their own. `SILO_BROKER_RESOLVE_URL` is the
same string on every cluster and resolves against the local controller.

Hosts are x86_64 only: the bootstrap pins amd64 binaries and refuses other
architectures, so cw-us-east-08a's arm64 GB200 pool is out. Public images pinned
by index digest (e.g. the python digest slot 3 uses) do carry arm64, but the
pipeline's own published images are amd64-only.

## Known gaps

* Per-sandbox disk is accounted, not enforced; a sandbox that fills the host's
  ephemeral storage gets the host pod evicted.
* Built (`RUN`) snapshots live on the host that built them and are rebuilt on
  first use elsewhere; they are not portable artifacts until published.
* Private registry pulls need `SILO_REGISTRY_AUTH`; the registry's htpasswd
  accounts are registry-wide, so any pull credential is also a push credential.
* Builder Dockerfiles with tag `FROM`s are accepted (Daytona accepted them) and
  recorded as `base_pinned: false`.
