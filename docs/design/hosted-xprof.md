# Hosted XProf profiles

## Context

Levanter can capture JAX XPlane traces during training, but the traces normally
remain on the worker that produced them. Opening a trace requires finding the
worker, copying the data locally, and starting a compatible XProf process. This
is slow enough that profiles are easy to lose or ignore, especially after an
Iris job exits.

The profiling callback should publish a durable, authenticated link while
keeping capture opt-in and preserving JAX's full `ProfileOptions` surface. HLO
protobufs must remain separately opt-in because they are larger and expose more
of the compiled program.

## Design

`ProfilerConfig` schedules the existing JAX trace callback. When profiling is
enabled, the callback captures into the job's local run directory and uploads
the completed XPlane session to the storage backend selected by `MARIN_PREFIX`:

```text
tmp/ttl=<days>d/xprof/<run-id>/plugins/profile/steps-<start>-to-<end>/
```

The default lifetime is seven days. An explicit destination can replace the
derived TTL root, and uploads can be disabled without disabling local capture.
Local development configurations that do not resolve to GCS or S3 keep the
trace locally and do not print a hosted link.

Every JAX process captures by default. Each process uploads its newly created
session files into the same canonical step directory so XProf sees one
multi-host run. A barrier separates trace finalization from upload, and a
second barrier lets every host finish its upload attempt before a local upload
error propagates. A configured process index can restrict capture when a full
distributed trace is unnecessary.

`ProfileOptionsConfig` mirrors the public fields on `jax.profiler.ProfileOptions`,
including `enable_hlo_proto` and advanced options. Perfetto JSON remains opt-in
because XProf consumes XPlane directly; requesting the interactive Perfetto link
also enables the JSON artifact required by that flow.

An always-on Iris service hosts the pinned XProf standalone application at
`/proxy/xprof`. Its `/open?uri=...` gateway validates the object URI, stages the
profile tree into a disposable local cache, and redirects to the XProf run. The
staging work is asynchronous so object-store transfer does not consume the Iris
proxy's request timeout. The gateway rewrites XProf's root-relative frontend
paths for the proxy prefix but otherwise delegates to XProf's WSGI application.

## Trust boundary

The browser never receives object-store credentials. The Iris proxy provides
user authentication, while the service receives GCS workload identity and
CoreWeave S3 credentials. The gateway accepts only configured buckets and paths
matching `tmp/ttl=Nd/xprof/<run-id>`. It rejects arbitrary GCS, S3, local, and
non-TTL paths before storage access.

The cache is not a persistence layer. Iris may clear it on restart; the TTL
object tree remains the source of truth and can be staged again.

## Deployment

Pulumi owns the Iris job, resources, retry policy, endpoint, storage allowlist,
and exact XProf version. Pinning XProf in the service environment isolates its
private standalone API from Levanter's optional profiling dependency. Changes
to the gateway or its Iris storage dependencies trigger the deployment workflow
after merge.

The `xprof-marin` Pulumi stack requires one manual initialization before its
first deployment so the generated encrypted data key can be committed. The
service runbook documents that command and the CoreWeave deploy secrets.
