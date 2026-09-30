# Brief: an Iris-backed sandbox provider to replace Daytona

**Audience:** the agent building this.  **Status:** requirements, not a design.
**Written:** 2026-09-22, from the surface `capability_pipeline/daytona_*.py`
(1,711 lines) actually calls today — not from Daytona's product surface.

> **DECIDED 2026-09-22 (Mark): the provider takes a pinned OCI digest as its
> native snapshot input.**  Design for that from the start; do not port
> Daytona's provider-local snapshot model.  §6 is the rationale and says what
> this deletes.  Relayed to session `daytona_iris`.

---

## 1. Why this is worth building

The capability pipeline's binding constraint on width is a **40-snapshot
org-wide Daytona quota**.  Every task under construction holds snapshots, so
the fleet caps out around **25–30 concurrent tasks** — and construction is the
slow stage (a single task took 8 h 30 m end to end).  Nothing else in the
pipeline is near its limit: the GLM fleet sustains ~900 concurrent sessions,
and CoreWeave CPU spills onto the H100 nodes.

Measured symptom, 2026-09-22: one construction slot stalled for **49 polls**
waiting on a snapshot; clearing 21 orphaned snapshots unblocked it
immediately.  The quota, not the work, was the queue.

So the goal is not feature parity with Daytona.  **The goal is the same
semantics with a quota we own.**

---

## 2. What we actually call

This is the entire provider surface the pipeline uses.  Anything not listed
here is not needed.

### 2.1 Client / snapshot lifecycle

| call | used for |
| --- | --- |
| `snapshot.create(params)` | build a named, immutable snapshot from an image + resources |
| `snapshot.get(name)` | resolve an existing snapshot by name; must 404 distinguishably |
| `snapshot.delete(snapshot)` | reclaim quota (note: `d.snapshot.delete(s)`, snapshots have no `.delete()`) |
| `client.create(params)` / `daytona.create(params)` | start a sandbox **from a snapshot** |
| `sandbox.delete()` | destroy a sandbox |
| `sandbox.id` | identity recorded in every trial receipt |

### 2.2 Process execution — two modes, both required

**One-shot** (the portable path):

```
sandbox.process.exec(command, cwd=..., env=..., timeout=...) -> {exit_code, result}
```

**Session** (long-running, needed because one-shot exec has no usable
streaming and we run multi-minute builds):

```
sandbox.process.create_session(session_id)
sandbox.process.execute_session_command(session_id, req) -> launched.cmd_id
sandbox.process.get_session_command(session_id, cmd_id)   -> status.exit_code
sandbox.process.get_session_command_logs(session_id, cmd_id) -> logs
sandbox.process.delete_session(session_id)
```

The session mode is polled, not streamed: `get_session_command` until
`exit_code` is non-null, then fetch logs once.  Keep that shape — the
controller's timeout accounting depends on being able to poll without
consuming output.

### 2.3 Filesystem

```
sandbox.fs.upload_file(data: bytes, target_path: str)
sandbox.fs.download_file(source_path: str) -> bytes
```

Directory upload/download are built on these in `daytona_environment.py`; you
do not need to provide them.

---

## 3. Invariants that are gate-relevant

These are not preferences.  Each one is load-bearing for a validation gate,
and a provider that silently drops one produces tasks that pass gates they
should fail.

### 3.1 `network_block_all=True` must actually block

Every candidate sandbox and every private-verifier sandbox is created with
`network_block_all=True` (`daytona_environment.py:280`,
`daytona_verifier.py:374`).  The quality reviewer certifies isolation as
`"verifier_isolation": "daytona-network-block-all"`.

**This is the single most important invariant.**  A solver that can reach the
network can fetch an answer, and a verifier that can reach the network is not
a sealed evaluator.  If the Iris provider cannot enforce egress blocking at
the sandbox boundary, say so loudly rather than approximating it — the
pipeline would rather stop than record an isolation claim it cannot support.

### 3.2 Fresh sandbox per trial, and reuse is checked

The repeated-diagnostics gate records `reused_sandbox_ids: []` and asserts it
is empty.  Each of the 3 attempts × N controls gets its own sandbox from the
same immutable snapshot.  Sandbox ids must therefore be unique and
non-recycled, and must appear in the receipt.

### 3.3 Snapshot identity is verified, not trusted

`daytona_snapshot.py` refuses to proceed on:

- `"resolved Daytona snapshot has the wrong provider identity"`
- `"reconstructed Daytona snapshot has the wrong identity"`
- `"resolved Daytona snapshot is not active"`

A snapshot is content-identified; resolving a name must return the snapshot
whose identity matches what the controller recorded, or fail closed.  Name
reuse with different content is the failure this guards against.

### 3.4 Error classes must be distinguishable

The lifecycle code branches on exactly three conditions:

| condition | detected as | meaning |
| --- | --- | --- |
| not found | `DaytonaNotFoundError` / HTTP 404 / `"not found"` in message | snapshot absent → create it |
| conflict | `DaytonaConflictError` / HTTP 409 | concurrent create won the race → resolve theirs |
| everything else | — | **do not retry**, surface it |

That last row matters.  A provider that returns a generic error for a missing
snapshot turns "create it" into a retry loop.

### 3.5 Deletion is asynchronous and must be observable

`wait_for_sandbox_deletion` exists because deletion returns before the
resource is gone.  We hit this directly: a cleanup reported `all_absent:
false` and only confirmed on a later re-list.  The provider needs a way to
ask "is it actually gone", and the answer must eventually become yes without
a second delete.

### 3.6 Readiness bound

`wait_for_snapshot_active` enforces **≤ 300 s total**, waits starting at zero.
A snapshot that is not active in 5 minutes is an error, not a longer wait.

### 3.7 Resource requests, and honest telemetry

Two profiles are in use (`daytona_resources.py`):

| profile | cpu | memory | disk |
| --- | --- | --- | --- |
| candidate (solver env) | 4 | 8 GB | 10 GB |
| verifier | 2 | 1 GB | 10 GB |

The pipeline reads back cgroup limits from *inside* the sandbox
(`parse_cgroup_telemetry`) and records `cpu_max`, `memory_max`,
`memory_current`, `memory_peak` — explicitly labelled
`"state": "observed_not_provider_verified"` and
`"host_visibility": "not_collected_not_used_for_limits"`.

So: **requests must produce real cgroup limits visible inside the sandbox.**
We do not need the provider to attest them; we need `/sys/fs/cgroup` to tell
the truth. An unlimited cgroup (`cpu.max = max`) is detected and recorded as
unverified — which is honest but degrades the evidence.

---

## 4. What you do *not* need to build

Worth stating, because Daytona is a large product and almost none of it is on
our path:

- No IDE, web UI, preview URLs, or port forwarding
- No git integration, no workspace templates
- No auto-stop / auto-archive lifecycle (we delete explicitly)
- No multi-user auth model — one org credential is what we use today
- No image *building* service **inside the provider**.  The provider accepts a
  pinned OCI digest (decided; §6) — it does not build images, and it does not
  push them.  Building and pushing stay outside it, in the isolated publisher.
- No streaming exec — polling is sufficient and is what we do

---

## 5. Mapping onto Iris — open questions, not answers

I have not designed this; these are the decisions I would expect the doc's
reader to make first.

1. **What is a "snapshot" on Iris?**  *Partly decided:* the input is a pinned
   OCI digest (§6).  It still needs to be immutable, content-identified, fast
   to instantiate many times, and cheap to hold.  The open part is **where the
   resource profile binds** — a digest alone carries no cpu/memory/disk, and
   `daytona_resources.py` computes a cache identity over
   `{image, cpu, memory_gb, disk_gb}`.  Decide whether a snapshot is
   `(digest, profile)` or whether the profile moves to sandbox creation; that
   choice determines what counts against the quota.
2. **What is the quota, and who owns it?**  The entire point is to replace a
   40-unit org cap.  Name the new limit explicitly and make it visible to the
   controller, because the controller currently cannot see its own ceiling —
   it discovers it by stalling.
3. **Egress blocking.**  CoreWeave egress is unrestricted by default (see the
   repo's operational notes).  Blocking it per-sandbox is the hard
   requirement and probably the main engineering risk.
4. **Sandbox startup latency.**  Construction creates sandboxes constantly; at
   8 h 30 m per task the per-sandbox cost is amortised, but the repeated
   diagnostics gate alone is 3 attempts × ~11 trials.  Measure it.
5. **Where does the provider run?**  Iris jobs land on `cw-us-east-02a`.  A
   sandbox-per-Iris-task model and a sandbox-inside-one-Iris-task model have
   very different quota and scheduling behaviour.

---

## 6. Custom images: a whole stage this decision deletes

**Read this before designing the snapshot model — it is the largest single
simplification available.**

### What the stage does today

When a builder needs a custom environment, Daytona builds a snapshot from a
Dockerfile recipe.  The resulting pointer,
`envgen.daytona/<snapshot>/dockerfile@sha256:<hash>`, **identifies a Daytona
recipe, not an OCI manifest** (`docs/builder_images.md`).  It is meaningful
only inside our Daytona org, so a task naming it would be captive to our
account forever.

The pipeline therefore runs a five-step conversion to make it portable:

1. **capture** the snapshot rootfs (trusted worker checks the snapshot is
   active and its recipe bytes match the hash-bound `source_recipe`)
2. **review** it — rootfs inspection plus a fresh independent GLM-5.3 review
3. **publish** those exact reviewed bytes as an OCI image to our own registry
   `envreg.208261-marin-gpu.coreweave.app` (blobs in
   `s3://marin-us-east-02a/registry/`)
4. **migrate** the spec pointer at `steps[0].verifier.verifier.runtime.image`
   to the pinned digest — `image_migration.py` enforces
   `host/path@sha256:<64 hex>`
5. **cold-pull** the published digest into **two fresh isolated Daytona
   sandboxes** to prove it boots from the registry, not from the local snapshot

This is not a TaskCompendium mechanism.  TaskCompendium lowers whatever image
reference the spec names.  A task using a pinned public digest skips the whole
stage — slot 3 exported with `docker.io/library/python@sha256:2f17fc04…` and
`custom_images.reason = "pinned_public_images"`.

### Why it exists, and why an Iris provider might delete it

Every one of those five steps exists for exactly one reason: **a Daytona
snapshot is not a portable artifact.**  The conversion is the cost of the
provider's native format being provider-local.

**This is settled (Mark, 2026-09-22): the provider takes a pinned OCI digest
as its native snapshot input.**

A builder authors a Dockerfile; it is built and pushed to the registry *as an
OCI digest from the start*; the provider is handed that digest.  The task then
names a portable reference from its first moment, and steps 1, 4 and 5 above
become unnecessary.  Step 2 (review) is still wanted — reviewing what ships in
a task environment is a safety property, not a portability one — and step 3
becomes an ordinary build-and-push rather than a rootfs-capture dance.

**This is currently blocking 4 of 10 slots** (`pending_image_capture`,
`pending_image_publication`), so it is not a theoretical tidiness win.

### The part that must stay isolated regardless

Do **not** fold registry publication into the sandbox provider to make those
runs flow.

The publisher credential is an `htpasswd` account that is **registry-wide**:
htpasswd authenticates identities but carries no repository authorization, so
its effective capability is pull, push and the server's deletion API across
the *entire* registry — including the `envgen/*` images owned by the
build_envs work.  It is fetched from GCP Secret Manager
(`capability-registry-publisher`) into a trusted isolated worker, never passed
as a CLI argument, and never injected into a GLM builder environment,
candidate sandbox, public task bundle or verifier artifact.

A sandbox provider is something builder-controlled code runs inside.  A
registry-wide push credential must not be reachable from there.  Keep the
publisher a separate trusted worker even if the capture stage goes away.

## 7. How to know it works

The pipeline is its own acceptance test, and it is strict enough to be a real
one.  In increasing order of confidence:

1. **Unit level** — the provider satisfies `daytona_snapshot.py`'s contract:
   404/409 distinguishable, identity verified, ≤300 s readiness, deletion
   observably async.
2. **One trial** — a single network-blocked sandbox runs a control and returns
   the expected reward, with a receipt carrying `sandbox_id` and cgroup
   telemetry.
3. **Isolation proof** — a deliberate egress attempt from inside a sandbox
   fails.  Do this explicitly; do not infer it from config.
4. **One full task** — re-run a known-good item end to end and require
   `EXPORTED: true` from `verify_export.py`, with
   `export_matches_harbor_bytes` and `taskspec_valid` both true.  The two
   already-exported tasks are the fixtures:
   `d14-hardware-fpnorm-combinational-0001` (slot 3) and the slot-5 arbiter
   task.
5. **The width test, which is the actual point** — run construction at
   **hundreds-wide concurrency** and show the snapshot ceiling is no longer
   what stops it.  If the new provider caps at 40 for different reasons, it
   has not solved the problem.

A provider that passes (1)–(4) and fails (5) is not a replacement; it is a
port.

---

## 8. Context the builder will want

- `capability_pipeline/daytona_environment.py` — Harbor `BaseEnvironment`
  implementation; the integration seam is here
- `capability_pipeline/daytona_verifier.py` — private-verifier isolation
- `capability_pipeline/daytona_snapshot.py` — lifecycle policy, the contract
  to satisfy
- `capability_pipeline/daytona_resources.py` — profiles and cache identity
- `capability_pipeline/daytona_telemetry.py` — cgroup read-back
- `docs/recipe.md` § "The gate chain, in order" — what the sandbox evidence
  feeds
- `docs/exports/d14-hardware-fpnorm-combinational-0001/build.md` — a worked
  example of a task passing every gate, including which evidence came from
  sandbox receipts
