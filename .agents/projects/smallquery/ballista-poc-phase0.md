# Ballista on Iris — Phase 0 findings

Issue: https://github.com/marin-community/marin/issues/7266
Parent: https://github.com/marin-community/marin/issues/6761

## Scope

Prove we can stand up a DataFusion-Ballista scheduler plus N executors on Iris and
run a few simple distributed SQL queries over our Parquet. Preemption, durability,
and object-store shuffle are out of scope for this PoC (tracked in the parent issue).

This document answers the two Phase 0 questions raised on #7266 before any building:

1. Do the prebuilt Ballista binaries have GCS support?
2. Where do we put the image so workers can see it?

It also evaluates the "have each task download and build the binary itself"
alternative.

Findings are against `apache/datafusion-ballista` `main`, which is the 54.x line
(54.0.0 released 2026-07-12).

## Q1 — Prebuilt binaries and GCS: no

The prebuilt scheduler and executor binaries do not read `gs://`. This is confirmed
at the source level, and it is not merely a missing compile flag — the runtime
registration is hardwired to S3.

Two independent facts:

- `ballista/core/Cargo.toml` pins the object_store crate to S3 only:

  ```
  object_store = { workspace = true, features = ["aws", "http"], optional = true }
  build-binary = ["aws-config", "aws-credential-types", "clap", "object_store"]
  default = ["arrow-ipc-optimizations"]
  ```

  The `gcp` object_store feature is not enabled, so `GoogleCloudStorage` is not even
  compiled in. The binary also pulls `aws-config` / `aws-credential-types` for S3
  credential discovery; there is no GCS analog.

- `ballista/core/src/object_store.rs` — `CustomObjectStoreRegistry::get_store()`
  dispatches on URL scheme and registers exactly three backends: local file
  (`""` / `file`), `http` / `https`, and `s3` (via `S3Options` config extension →
  `AmazonS3Builder`). Any other scheme, including `gs`, returns
  `get_store - store not supported`.

S3 support in the binaries is recent (the 53.x/54.x line). GCS was left out. Adding
it is a source change to Ballista, not a config or env toggle: per the
[Extending Components guide](https://datafusion.apache.org/ballista/user-guide/extending-components.html)
object stores are wired through compile-time `config_producer` / `runtime_producer`
/ `session_builder` hooks on the scheduler and executor, so a stock binary cannot be
taught a new store at runtime.

Native `gs://` therefore requires a small fork/patch:

1. add `gcp` to the object_store features in `ballista/core/Cargo.toml`;
2. handle the `gs` scheme in `CustomObjectStoreRegistry::get_store()`, building a
   `GoogleCloudStorage` store (mirror the existing S3 branch);
3. supply credentials. object_store's `gcp` feature resolves Application Default
   Credentials — `GOOGLE_APPLICATION_CREDENTIALS`, or the GCE metadata server. Iris
   GCP workers already run under a service account (the base image ships the gcloud
   SDK), so no key material or `GcsOptions` extension is strictly needed for a PoC;
   ADC is enough.

This is symmetric with the S3 code that just landed and is plausibly upstreamable,
so it need not become a long-lived fork.

## Q2 — Where the image lives: GHCR, pulled through the Artifact Registry mirror

Iris already has a first-class custom-task-image path, so nothing new is needed to
make an image visible to workers.

- Per-job override: `iris job run --task-image <ref> ...` (`lib/iris/src/iris/cli/job.py:982`),
  or SDK `task_image=`. The help text is explicit: "The image must already exist in a
  registry visible to workers."
- Hosting and distribution (`lib/iris/docs/image-push.md`): images are pushed to
  GHCR (`ghcr.io/marin-community/...`) as the single source of truth. GCP workers do
  not pull GHCR directly; the autoscaler / worker image resolver rewrites the tag to
  a per-continent Artifact Registry remote (pull-through) repo
  (`us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/...`, `europe-docker.pkg.dev/...`)
  based on the VM zone. First pull is a cache miss against GHCR; subsequent
  same-continent pulls are fast and egress-free. CoreWeave workers pull GHCR
  directly.

So the image path is: build a task image carrying `ballista-scheduler` and
`ballista-executor`, `docker push ghcr.io/marin-community/ballista:<ver>`, and run
the scheduler/executor jobs with `--task-image ghcr.io/marin-community/ballista:<ver>`.
The AR-mirror rewrite happens automatically; no multi-region push is required.

This is exactly the packaging the existing datafusion-on-Iris precedent uses (see
`jayendra13/zarr-datafusion/deploy/iris`, linked from the parent issue): a Dockerfile
compiles the Rust binaries, pushes to AR/GHCR, and jobs run against that image.

### The "build it on the task itself" alternative

wmoss's heavier fallback — have each scheduler/executor download and build the binary
at startup — is feasible: the Iris task base image already ships a Rust toolchain
(`lib/iris/Dockerfile:201-203`, rustup stable + cargo) plus `build-essential`. A
setup script could `cargo install --locked --git <repo> ballista-scheduler
ballista-executor`.

Not recommended as the default:

- A full Ballista build is multi-minute and pulls ~GB of crates. It reruns on every
  cold task unless the cargo target cache persists across tasks, which is not
  guaranteed for arbitrary task containers.
- It does not avoid the Q1 problem. `cargo install` of stock Ballista still yields
  S3-only binaries; to get `gs://` the build must point at the patched source
  anyway. So building-on-task only replaces "where is the image" with "where is the
  patched source", while adding per-task build latency across N executors.

Prebuilt custom image wins on determinism and executor start latency. Keep
build-on-task only as a quick local-iteration escape hatch.

## GCS access — the three PoC options, ranked

Given Q1, there are three ways to let Ballista read our data. The choice is
independent of Q2 (all three ship in the same custom image).

1. Native `gs://` via the small patch above. No FUSE, no privileged containers,
   uses ADC. This is the path smallquery ultimately wants and is upstreamable.
   Cost: a ~1-file Rust patch plus a Cargo feature.

2. gcsfuse mount + stock binaries. wmoss's suggested first PoC. Zero Ballista code —
   mount the bucket and register it as the local-file store, which is compiled in.
   New constraint found during this pass: an in-container FUSE mount needs the
   `mount()` syscall and `/dev/fuse`, which Iris only grants under
   `CONTAINER_PROFILE_PRIVILEGED` (`--privileged`), and that profile is admin-gated
   (`lib/iris/docs/container-profiles.md`; `DEFAULT` runs `--cap-drop ALL` + default
   seccomp and cannot mount). So gcsfuse means running the scheduler and every
   executor privileged, each mounting the bucket at an identical path (the scheduler
   hands file paths to executors, so mountpoints must match). Workable for a
   throwaway PoC, but the privilege escalation and per-executor mount make it more
   operational hassle than option 1's patch.

3. Prestage local Parquet. Bake or download a few hundred MB of region-local Parquet
   to task-local disk and query `file://`. No GCS and no FUSE at all. Proves the
   distributed scheduler/executor/shuffle mechanics fastest, but does not prove we
   read our object store.

Recommendation: use option 3 for the very first smoke (prove scheduler + executors +
a real shuffle run on Iris with nothing else in the way), then land option 1 (native
`gs://`) as the target for reading our data. Prefer option 1 over option 2 because
the small Rust patch avoids privileged containers on every executor and is the
direction the real system needs regardless. This is a decision for wmoss — option 2
is viable if we would rather not touch Ballista source at all for the PoC.

## Deployment shape on Iris (for reference, not this phase)

The mechanics are already covered by Iris primitives; no new capability is needed.

- Scheduler: a single-task job running `ballista-scheduler`, requesting one named
  port. It registers its address so executors can find it. Pin it
  `--no-preemptible` for the PoC.
- Executors: a self-registering worker pool of N tasks, mirroring
  `lib/iris/src/iris/client/worker_pool.py` (which already does
  `ctx.registry.register(name, address, metadata)` — `worker_pool.py:165`). Each
  executor resolves the scheduler via `ctx.resolver.resolve(...)`
  (`resolver.py:73`, `client.py:1133`) and binds/advertises on `IRIS_ADVERTISE_HOST`
  + its injected `IRIS_PORT_<name>` so peer Arrow Flight fetches use the routable IP.
- Client: `ballista-cli` (or the newly-published 54.0.0 Python wheel) against the
  scheduler endpoint; `CREATE EXTERNAL TABLE ... STORED AS PARQUET LOCATION ...` then
  `SELECT`.

The endpoint-registry and resolver names above are verified against the current Iris
client (the earlier plan on #7266 guessed `ctx.endpoint_registry`; the real accessor
is `ctx.registry`).

## Open questions for wmoss

- GCS access: native `gs://` patch (option 1, recommended) or gcsfuse (option 2,
  your earlier suggestion — note the privileged-container cost found above)?
- Ballista version: pin to 54.0.0 (latest, 2026-07-12, first with published Python
  wheels) unless there is a reason to track an older tag.
- If we go with the patch, keep it as a marin-side fork branch we build in the image,
  and open the GCS-store addition upstream in parallel?
