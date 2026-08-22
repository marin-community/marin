# Iris Task Health Research

- Effort: Medium.
- Stop rule: New searches no longer changed the task contract or backend design.
- Date: 2026-08-21.

## Question

Can Iris make `/healthz` a task contract across worker-daemon and Kubernetes backends? Can Levanter training use this contract?

## Current Marin Context

PR [#8554](https://github.com/marin-community/marin/pull/8554) adds `/healthz` to the Levanter training-control server. The route reads `ProgressWatchdog` state. It returns 503 after a progress deadline and 200 in other states. The PR does not add an Iris probe. [Review feedback](https://github.com/marin-community/marin/pull/8554#issuecomment-5375624425) asks for an Iris feature.

Iris already moves immutable task configuration through two wire messages. `LaunchJobRequest` has named ports at [`controller.proto:35`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/iris/src/iris/rpc/controller.proto#L35). `RunTaskRequest` carries the task specification to each backend at [`job.proto:644`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/iris/src/iris/rpc/job.proto#L644). The controller stores named ports in `job_config` and rebuilds each run request from that row.

The worker daemon allocates host ports before it starts a container at [`task_attempt.py:566`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/iris/src/iris/cluster/worker/task_attempt.py#L566). Each task has a monitor thread at [`task_attempt.py:808`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/iris/src/iris/cluster/worker/task_attempt.py#L808). This thread already stops containers and reports terminal task errors. A task health probe fits this owner.

The Kubernetes backend builds Pod manifests directly at [`tasks.py:766`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/iris/src/iris/cluster/backends/k8s/tasks.py#L766). Task Pods have `restartPolicy: Never`, but they have no task probes at [`tasks.py:976`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/iris/src/iris/cluster/backends/k8s/tasks.py#L976). Several CoreWeave configs set `host_network: true`. A fixed health port can conflict when a node runs two task Pods.

The endpoint registry is a poor control input for task health. A task registers an endpoint after it starts. The Kubernetes backend must define its probe in the Pod manifest before start. Endpoint lease renewal also reports that a renewal thread runs, not that the task makes progress.

## Internal Prior Work

Issue [#4799](https://github.com/marin-community/marin/issues/4799) proposed an in-process task debug server. It identified named Iris ports and `get_job_info()` as the application boundary. It did not define task failure semantics or Kubernetes probes.

Issue [#5204](https://github.com/marin-community/marin/issues/5204) proposed controller synthetic checks. That work measures cluster service health. It does not measure one task process.

The Iris controller deployment already uses Kubernetes HTTP readiness and liveness probes at [`controller.py:275`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/iris/src/iris/cluster/platforms/k8s/controller.py#L275). The worker autoscaler also uses consecutive HTTP failures at [`runtime.py:136`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/iris/src/iris/cluster/controller/autoscaler/runtime.py#L136). These checks supply useful conventions, but they check Iris services and machines.

Issue [#6944](https://github.com/marin-community/marin/issues/6944) shows the cost of an aggressive liveness probe. A busy controller missed its probe deadline and Kubernetes stopped it. PR [#6945](https://github.com/marin-community/marin/pull/6945) increased its request timeout and failure threshold. A later [Iris incident](https://echo.oa.dev/wiki/21) found real transient health stalls without a necessary restart. These events support a long consecutive-failure window for expensive training.

Levanter starts one training-control server only on global JAX process zero at [`training_control.py:189`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/levanter/src/levanter/training_control.py#L189). Iris health is task-scoped. A multi-task job thus needs one server in each Iris task. The global process-zero server can continue to own the public control endpoint.

PR [#8543](https://github.com/marin-community/marin/pull/8543) records one related rank gap. `ProgressWatchdogConfig.create` returns no watchdog for nonzero processes when diagnostic capture is enabled. A health server on each task leader needs a local watchdog. Expensive diagnostic capture can stay on global process zero.

The watchdog checks deadlines every 60 seconds at [`progress_watchdog.py:19`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/levanter/src/levanter/callbacks/progress_watchdog.py#L19). The FSDP hero gives diagnostic capture 20 more seconds at [`launch.py:63`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/experiments/grug/moe_hero_fsdp/launch.py#L63). Iris must wait longer than these two budgets. This delay lets the application exit with code 124 and finish its diagnostic capture.

Fray owns the training submission type at [`types.py:694`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/fray/src/fray/types.py#L694). Its Iris backend maps that type to `IrisClient.submit` at [`iris_backend.py:668`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/fray/src/fray/iris_backend.py#L668). Training needs a Fray health-check field and an Iris conversion.

## External Prior Art

Kubernetes defines startup probes as a gate before liveness probes. A startup success enables liveness checks. Repeated liveness failures cause container termination. HTTP success includes status codes from 200 through 399. See the [Kubernetes probe contract](https://kubernetes.io/docs/concepts/workloads/pods/probes/).

Kubernetes supports `exec` probes. A zero exit code is healthy. This mechanism permits a dynamic task port and prevents fixed-port conflicts with `hostNetwork`. See the [Kubernetes probe configuration guide](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-probes/).

Docker health checks use a start period, a request timeout, and consecutive failures. This state model gives a useful parity target for the worker-daemon implementation. See the [Dockerfile `HEALTHCHECK` reference](https://docs.docker.com/reference/dockerfile#healthcheck).

Kubernetes warns that incorrect liveness checks can cause repeated restarts and load failures. Iris batch tasks use `restartPolicy: Never`, but false failures still consume retry limits. This supports an opt-in first release.

## Negative and Failed Leads

- No Kubernetes task liveness probe exists in the current Iris task manifest.
- Endpoint metadata cannot install a Kubernetes probe before the endpoint registers.
- A fixed port is unsafe on CoreWeave clusters that use host networking.
- A TCP probe only proves that a socket accepts connections. It cannot report a training stall.
- A readiness probe has no task-lifecycle value because Iris batch tasks do not use a Kubernetes Service.
- A controller-side probe adds control-plane and proxy dependencies to a local task verdict.
- An Iris deadline shorter than the application diagnostic path can replace a useful exit code with an external kill.

## Evidence Map

### Claim: Health configuration belongs in the task specification

- Support: The two backends receive `RunTaskRequest` before task start.
- Contradiction: Endpoint registration already carries an address and metadata, but it occurs after task start.
- Directness to Marin: High.
- Confidence: Stable.
- Action: Add `TaskHealthCheck` to launch and run messages. Store it in `job_config`.

### Claim: The two backends can use one failure state model

- Support: Kubernetes and Docker use startup state and consecutive failures.
- Contradiction: Kubernetes owns termination details, while the worker daemon writes its own terminal error.
- Directness to Marin: High.
- Confidence: Stable for lifecycle behavior. Diagnostic parity needs a test.
- Action: Share contract values. Keep backend execution separate.

### Claim: Training needs one health server per Iris task

- Support: Iris retries and reports each task independently.
- Contradiction: PR #8554 starts the server only on global process zero.
- Directness to Marin: High for multi-host training.
- Confidence: Stable.
- Action: Start the health route on one local leader in every task. Register the public control endpoint only on global process zero.

### Claim: Iris must act after the application watchdog

- Support: The Levanter watchdog can wait 60 seconds to detect a deadline and 20 seconds for diagnostics.
- Contradiction: A shorter Iris window restarts a hard deadlock more quickly.
- Directness to Marin: High for Grug FSDP training.
- Confidence: Stable for the current watchdog settings.
- Action: Use a 120-second Iris failure window for the first training rollout.

## Recommended Validation

### 1. Backend state parity

- Minimum test: Use one response sequence against the worker state machine and a rendered Kubernetes probe configuration.
- Baseline: No health check leaves current task behavior unchanged.
- Expected signal: Startup and consecutive-failure boundaries match.
- Falsifier: The same response sequence kills a task at different boundaries.
- Cost or risk: Low.

### 2. Deadlocked training process

- Minimum test: Hold the training thread after a progress event and keep the HTTP thread active.
- Baseline: A TCP-only check stays healthy.
- Expected signal: `/healthz` returns 503 and Iris fails the task within the configured threshold.
- Falsifier: The task stays RUNNING or reports worker failure.
- Cost or risk: Medium. Run this test on one GCP worker and one Kubernetes development backend.

### 3. Application exit wins the race

- Minimum test: Return 503 while the watchdog waits through its poll and diagnostic budgets.
- Baseline: A 30-second external window can stop the task first.
- Expected signal: Levanter exits with code 124 before Iris reaches its failure threshold.
- Falsifier: Iris kills the task first or the diagnostic does not finish.
- Cost or risk: Low with a fake clock. Medium on one development task.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| PR #8554 | Pull request | https://github.com/marin-community/marin/pull/8554 | Levanter report and missing probe | High | Direct change request. |
| Iris task code | Marin code | Pinned links above | Backend and port boundaries | High | Current `origin/main`. |
| Issue #4799 | GitHub issue | https://github.com/marin-community/marin/issues/4799 | In-process debug-server prior work | Medium | Closed without implementation. |
| Issue #5204 | GitHub issue | https://github.com/marin-community/marin/issues/5204 | Cluster-health boundary | Medium | Different health scope. |
| PR #8543 | Pull request | https://github.com/marin-community/marin/pull/8543 | Watchdog rank gap and startup deadline | High | Direct Levanter history. |
| Issue #6944 and PR #6945 | GitHub issue and pull request | https://github.com/marin-community/marin/issues/6944 | False liveness failure risk | High | Direct Iris incident and fix. |
| Echo wiki 21 | Incident record | https://echo.oa.dev/wiki/21 | Transient probe stalls | High | Direct Iris operations record. |
| Kubernetes probes | Official docs | https://kubernetes.io/docs/concepts/workloads/pods/probes/ | Probe semantics | High | Primary documentation. |
| Docker health checks | Official docs | https://docs.docker.com/reference/dockerfile#healthcheck | Worker state model | High | Primary documentation. |

## Handoff

- Keep endpoint registration and task health as separate contracts.
- Make health checks optional in the first release.
- Use one local task leader for multi-process tasks.
- Keep the application watchdog as the first termination path.
- Open questions: Default training time limits and the first training rollout group.
- Stop reason: More sources did not change the contract.
