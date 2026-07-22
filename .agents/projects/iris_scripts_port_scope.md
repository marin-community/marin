# Iris operational scripts port scope

## Decision

Port the functionality, not the current script boundary. The OpenThoughts-Agent
directory contains 14 in-scope executable files. `launch_external_opencode_eval.py`
is deferred pending a native Marin Harbor evaluation configuration, and
`patch_tpu_inference.py` must be solved in an upstream or pinned fork. The 14
remaining scripts combine three layers
that should not remain coupled:

1. generic Iris lifecycle, Finelog, bundle, and CoreWeave-pod collection;
2. Marin-owned RL and Harbor result semantics; and
3. OpenThoughts launch conventions, hard-coded model locations, and secret files.

The first layer belongs in `marin-iris`; the second belongs in `marin-core`;
the third must be replaced by explicit Marin configuration or retired. No ported
module may depend on `/Users/benjaminfeuer/...`, invoke a second `iris` binary,
read a kubeconfig selected by `KUBECONFIG`, or decode `iris-task-env` from
Kubernetes.

## Base and target layout

This scope was prepared from Marin `main` at
`b22baf7b5bd5298faa2e944d8072ffed260f4561`.

```text
lib/iris/src/iris/
  diagnostics/
    __init__.py
    bundles.py             # stable local evidence bundle format
    jobs.py                # typed job inventory, lifecycle, retry policy
    finelog.py             # complete log acquisition and coverage checks
    coreweave.py           # configured K8s/S3 artifact collection
  cli/
    job.py                 # `iris job list`, `summary`, and new `watch` surface
    diagnostics.py         # `iris diagnostics ...` commands

lib/marin/src/marin/
  rl/iris_diagnostics.py             # RL progress and completed-run analysis
  evaluation/harbor_diagnostics.py   # Harbor/datagen/eval progress and analysis
  model_mirror.py                    # portable object-store mirror library
  cli/diagnostics.py                 # Marin-owned CLI wiring, if a top-level
                                     # `marin` command is introduced
```

`iris.diagnostics` is deliberately a client-side package: it reads existing
controller, Finelog, Kubernetes, and object-store state. It is not controller
logic, does not add schema columns, and does not alter task lifecycle.

## Inventory and exact disposition

| OpenThoughts source | Current responsibility | Marin destination | Disposition |
|---|---|---|---|
| `scripts/iris/iris_ops.py` | shell-based lifecycle polling; DNS retry; local manifest location | `iris.diagnostics.jobs`, `iris.diagnostics.bundles`, `iris.cli.job` | Port the typed state/bundle contract; replace subprocess calls with `connect_controller()` and `IrisClient`; add `iris job watch`. |
| `scripts/iris/list_iris_jobs.py` | one-user recent-job table; job-name heuristic | `iris.cli.job:list_jobs` | Extend the existing native command with `--user`, `--since`, JSON/CSV output and a stable table. Do not preserve the workload-type name heuristic in Iris. |
| `scripts/iris/coreweave_ops.py` | pod discovery, task/Ray/vLLM log capture, S3 trial copy, transient kubectl retry | `iris.diagnostics.coreweave` | Port as a configured collector built on `CloudK8sService`; use the cluster's pinned `kubeconfig_path`/context and a caller-provided object-store client. Delete secret decoding from `iris-task-env`. |
| `scripts/iris/watch_coreweave_rl.py` | active CoreWeave RL discovery, Finelog/pod/Ray sync, newest-500 `trace_jobs`, RL metrics table | `marin.rl.iris_diagnostics` using `iris.diagnostics.*` | Port as `marin rl iris-watch` or `iris diagnostics rl`; preserve the fleet-wide recent-trace policy and manifest. Move SkyRL/Harbor parsing out of Iris. |
| `scripts/iris/analyze_coreweave_rl_job.py` | resync/read shared RL evidence bundle and summarize it | `marin.rl.iris_diagnostics` | Fold into the same module as a completed-run subcommand. It must work from an existing bundle when a remote resync is unavailable. |
| `scripts/iris/analyze_coreweave_rl_job_live.sh` | 559-line live `kubectl` inspection shell | `marin.rl.iris_diagnostics` | Replace with explicit Python subcommands over the collector: summary, list, show trial, grep artifacts, pull selected trial, and flight-recorder capture. Retire the shell script after parity tests. |
| `scripts/iris/watch_iris_harbor.py` | active Harbor datagen/eval discovery across TPU and CoreWeave; results, mean, error count, Finelog and Ray/vLLM sync | `marin.evaluation.harbor_diagnostics` using `iris.diagnostics.*` | Port. It understands `run_tracegen.py`, `result.json`, and `exception.txt`, so it belongs with Marin's existing Harbor evaluator, not Iris. |
| `scripts/iris/analyze_iris_harbor_job.py` | complete Finelog acquisition through compaction; throughput/preemption/stage analysis; Harbor trial analysis | `iris.diagnostics.finelog` plus `marin.evaluation.harbor_diagnostics` | Split. Generic Finelog paging, deduplication and coverage become Iris code; Harbor line parsing, output layout, result and exception analysis remain Marin evaluation code. |
| `scripts/iris/analyze_iris_harbor_job_live.sh` | thin live watcher wrapper | command alias only | Delete after the Harbor watcher command exists. No separate implementation. |
| `scripts/iris/mirror_models.py` | route dispatch for HF→GCS, GCS→S3, HF→S3 | `marin.model_mirror` | Port as library routing plus a typed CLI. |
| `scripts/iris/mirror_hf_to_gcs.py` | resumable Hub model fan-out to GCS | `marin.model_mirror` | Port algorithm after making destination prefixes explicit configuration. Reuse `gcsfs`/`huggingface_hub`; centralize the include manifest. |
| `scripts/iris/mirror_gcs_to_s3.py` | resumable GCS→S3 transfer | `marin.model_mirror` | Port behind configured source/destination clients. Retain one-file-at-a-time transfer and terminal manifest. |
| `scripts/iris/mirror_hf_to_s3.py` | resumable in-cluster Hub→CoreWeave S3 transfer | `marin.model_mirror` | Port with the same configured S3 client. Keep it runnable in a CoreWeave task; do not infer credentials from an operator laptop. |
| `scripts/iris/launch_mirror.py` | OpenThoughts `IrisLauncher` submission wrappers for the two GCS routes | `marin.model_mirror` plus an Iris submit helper | Replace rather than copy. The implementation must build an `IrisClient.submit` request and use config-defined resources/images; it cannot import OpenThoughts `hpc.iris_launch_utils`. |

## Existing Marin capabilities to reuse

| Need | Existing capability | Required change |
|---|---|---|
| Controller access and credentials | `iris.cli.connect.connect_controller`, `IrisClient`, `iris job list/summary/logs` | Diagnostics calls these APIs directly instead of launching a second `iris` executable. |
| CoreWeave cluster selection | `IrisClusterConfig`, `CloudK8sService`, `lib/iris/config/*.yaml` | Read `kubeconfig_path` and `kube_context` from the selected cluster config. This fixes the stale `KUBECONFIG` class of failure centrally. |
| Kubernetes reads/exec/logs | `K8sService` protocol and `CloudK8sService` | Add only the read-only artifact operations the collector needs; preserve fakes at this protocol boundary. |
| Object-store configuration | `configure_client_s3`, `s3fs`, `gcsfs` | Create a single explicit diagnostic storage factory. It receives operator credentials from normal configuration; it must not read Kubernetes Secrets. |
| Finelog | `marin-finelog`, `LogClient`, deploy config utilities | Extract complete log acquisition into `iris.diagnostics.finelog`; keep compaction-race retries bounded and observable. |
| Harbor result semantics | `marin.evaluation.evaluators.harbor_evaluator` | Reuse its `result.json` conventions and make shared result/error parsing a public small helper. |
| Object-store evaluation records | `marin.evaluation.records` | Write a portable terminal summary record for successful and failed Harbor analysis instead of only a local report. |

## Dependencies that must move or disappear

### Direct OpenThoughts imports

`analyze_iris_harbor_job.py` imports
`hpc.iris.job_output_resolver.resolve_job_output_dir`. Its GCS-only output
resolver is a Harbor launch convention. Recreate it as a narrow
`marin.evaluation.harbor_diagnostics.resolve_output_uri()` that first consumes
the Iris job's recorded entrypoint/environment and then accepts an explicit
override. It must support both `gs://` and `s3://` through fsspec and never
consult an OpenThoughts local SQLite registry.

`launch_mirror.py` imports OpenThoughts `IrisLauncher` and `PROJECT_ROOT`.
Replace it with `connect_controller()` plus `IrisClient.submit`; resources,
image, destinations and target cluster become typed command/config options.

`hpc.iris.precache` dynamically imports `mirror_hf_to_gcs` and documents the
old command. It is a caller that must migrate to `marin.model_mirror`, then
be removed from the OpenThoughts dependency graph once its launchers have
moved. `hpc.model_mirror_registry` is likewise an application-local catalog;
do not copy it until Marin decides whether its object manifests belong in a
registry or are queried directly.

### Third-party dependencies

`marin-iris` already owns `marin-finelog`, `fsspec`, `gcsfs`, `s3fs`,
Kubernetes support, and `tabulate`. `marin-core` already owns the Harbor
extra and Hugging Face Hub through its existing extras. Avoid a new `boto3`
dependency unless an S3 operation cannot be expressed through the existing
configured filesystem; the current scripts use boto3 chiefly because they
manually recover task credentials.

`duckdb` is already a Finelog dependency. Keep it inside the Finelog-specific
module rather than making it an Iris controller dependency.

## Unsafe assumptions to remove before any port

- `/Users/benjaminfeuer/Documents/...` for Marin source, evidence bundles,
  config paths, task images and the `iris` executable.
- Fixed user `benjaminfeuer`, fixed CoreWeave buckets, and two historical
  kubeconfig paths. Cluster config is the source of truth.
- `KUBECONFIG` inheritance. CoreWeave operations must bind the selected
  cluster's configured path and context.
- Reading the `iris-task-env` Secret with `kubectl`, base64-decoding AWS
  credentials, and writing/using them on the launch host. This violates the
  task-secret boundary. Operator-side collection requires explicitly provided
  operator credentials, or an in-cluster collector task with a constrained
  output contract.
- `--secrets-env` and plaintext secret-file parsing. Secret values should use
  `EnvironmentSpec.env_vars` or cluster `inject_env`; token-bearing URLs must
  never appear in argv, manifests, logs, or reports.
- Parsing arbitrary shell entrypoints as the primary interface. Keep a
  bounded legacy parser for historical runs, but new Marin Harbor/RL launch
  records must write typed diagnostic metadata (dataset, output URI, trial
  count, serving endpoint, and workload kind) at submission.
- The runtime file-rewrite patch for `tpu-inference`, including its broad
  `except Exception`. The real fix is a versioned upstream dependency.

## Phased implementation plan

### Phase 1: typed Iris diagnostic substrate

Add `iris.diagnostics.bundles` and `iris.diagnostics.jobs` with a versioned
manifest schema and a `JobRef(cluster, JobName)` key. The default root is a
user cache directory (for example `~/.cache/marin/iris/jobs`), while every
command accepts `--bundle-root`. Preserve the current stable layout:

```text
<root>/jobs/<cluster>/<user>/<job>/manifest.json
```

Implement controller operations through `connect_controller`/`IrisClient`:
job list, summary, task log fetch, lifecycle polling, and retry classification.
Use `rigging.timing.Deadline`/`ExponentialBackoff`; distinguish a retry-exhausted
transport error from a terminal job state. Add `iris job watch <job>` and extend
`iris job list` rather than add a second inventory script.

Tests in `lib/iris/tests/cli/test_job.py` and a new
`lib/iris/tests/diagnostics/test_bundles.py` should cover canonical bundle
paths, invalid non-root IDs, listing filters, terminal state transitions, and
retry exhaustion with fake controller transport.

### Phase 2: read-only CoreWeave artifact collector

Add `iris.diagnostics.coreweave` on top of `K8sService`. It finds a task pod by
Iris labels/job metadata rather than substring matching, gets task and Ray/vLLM
logs, safely extracts a streamed tar archive, and records collection failures
without inserting HTML proxy pages into the status table. Keep the 100 MiB log
and 20 MiB non-log bounds configurable.

Add configured S3/GCS artifact listing and selected-trial copying. The collector
must take a filesystem/client provided by a storage factory; it cannot obtain
credentials from a pod secret. Define a clear unavailable-artifacts outcome
when the invoking principal lacks object-store credentials.

Tests in `lib/iris/tests/diagnostics/test_coreweave.py` should use a fake
`K8sService`, in-memory tar streams, and a fake filesystem. Cover an empty tar,
one retryable transport failure, safe relative paths, no token/secret output,
and the most-recent-N selection by object modification time.

### Phase 3: Marin RL and Harbor analyzers

Move the RL watcher and both RL analyzers into
`marin.rl.iris_diagnostics`. It owns RL command/metric patterns and
`trace_jobs`; it consumes only typed bundles and the Phase 2 collector. Its
default must retain the fleet-wide most-recent 500 traces and emit selected,
available, completed, and omitted counts in the bundle manifest.

Split `analyze_iris_harbor_job.py` before moving it:

- Finelog segment pagination, live/GCS merge, de-duplication, and coverage
  checks move to `iris.diagnostics.finelog`.
- Harbor trial discovery, `result.json` mean/error extraction,
  `exception.txt` frequency, task totals, stage quantiles, throughput, and
  preemption report move to `marin.evaluation.harbor_diagnostics`.

`watch_iris_harbor.py` becomes the active-job mode of that same Harbor module.
It must report dataset, completed/total/remaining, mean for direct results,
error counts from both `result.json` and `exception.txt`, Finelog state, Ray/vLLM
state, and trend. It must route controller requests through the Phase 1 retry
policy.

Port behavior tests from OpenThoughts into focused tests under
`lib/marin/tests/rl/` and `lib/marin/tests/evaluation/`, using fixture bundles
and fake filesystems. Do not retain tests that assert command-string assembly.

### Phase 4: mirror launch behavior

Implement `marin.model_mirror` with a typed `MirrorRoute` and a shared file
selection/manifest implementation. Destination URIs are explicit CLI/config
values; the two historical multi-region GCS defaults are not library defaults.
Add an optional `marin[mirror]` extra only if `huggingface-hub` cannot be made
available from an already-selected Marin extra. Submit mirror work via a small
Iris-client launcher, with an in-cluster route for CoreWeave S3.

Tests verify resumability using a fake Hub/filesystems, manifest-last behavior,
and route validation.

### Phase 5: cutover and deletion

Update launcher call sites and operational docs first. Run the new diagnostics
alongside the existing OpenThoughts watchers for one active CoreWeave RL job and
one Harbor TPU job, compare counts/means/error totals/artifact manifests, then
switch cron/monitor invocations.

After parity is demonstrated, remove all 14 in-scope OpenThoughts scripts, their Iris
tests, references in `scripts/README.md`, `.agents/ops/iris/ops.md`, and
OpenThoughts `hpc.iris.precache` imports. Delete the two shell wrappers rather
than retaining compatibility aliases. The excluded external-endpoint launcher
and TPU patch follow separate native-eval and upstream/fork workstreams.

## Acceptance criteria

- `iris job list` and `iris job watch` function from any supported local host
  without `IRIS_BIN`, `MARIN_MAIN_ROOT`, or `KUBECONFIG` overrides.
- Every monitor/analyzer addresses the same bundle for `(cluster, full root
  job id)` and can analyze an existing bundle when remote collection is absent.
- A CoreWeave bundle records Finelog, pod/Ray/vLLM collection outcomes and a
bounded, provenance-bearing trace selection without exposing object-store
credentials.
- Harbor reports agree with direct `result.json` and `exception.txt` fixtures,
including completed/remaining counts, mean reward, and grouped errors.
- Controller and Kubernetes transport retries are bounded, surfaced as
collection errors after exhaustion, and never mistaken for a job failure.
- No OpenThoughts import, absolute user path, user-specific bucket default,
secret-file parser, or runtime third-party source patch remains in Marin.
