# Iris Task Health Checks

Iris tasks can opt in to an HTTP health contract at `/healthz`. Iris will fail an unhealthy task through the normal retry path. The first user is Levanter training, where the progress watchdog can identify a stalled training process.

## Background

PR [#8554](https://github.com/marin-community/marin/pull/8554) adds a Levanter `/healthz` route, but no system probes the route. Iris has different task owners on worker-daemon and Kubernetes backends. The endpoint registry starts too late to configure Kubernetes probes. The [research brief](research.md) contains the code map and prior work.

## Challenges

Iris must give the two backends the same task-failure contract. The worker daemon owns Docker task processes. Kubernetes owns Pod liveness and task termination.

Iris named ports are dynamic. CoreWeave task Pods also use `hostNetwork`. One fixed health port can cause conflicts between tasks on the same node.

Training is process-scoped, while Iris health is task-scoped. Multi-host and multi-process jobs need one health server in each Iris task. Only global process zero can own checkpoint control.

## Design

Add an optional `TaskHealthCheck` to the Iris job specification. The contract has four required values: startup timeout, probe period, request timeout, and consecutive-failure threshold. The path is always `/healthz`. Iris reserves the internal named port `healthz` when the contract is present.

The application owns the HTTP response. Status codes from 200 through 399 mean healthy. A different status, connection error, or request timeout means unhealthy. Iris does not follow redirects or parse the response body. Applications can return a small JSON report for diagnosis.

The application reads its bind port from `IRIS_PORT_HEALTHZ`. On worker-daemon backends, Iris assigns a real host port. On Kubernetes, the value is zero, so the server can ask the kernel for a port. After listen starts, the application calls `publish_task_health(port)`. The helper records the selected port at the fixed internal path `/tmp/iris/health-port` for the Kubernetes exec probe. The worker daemon probes its allocated port directly.

The health state has two phases. During startup, Iris waits for the first successful request until `startup_timeout`. After the first success, each success clears the failure count. Iris fails the task after `failure_threshold` consecutive failures.

The worker-daemon startup clock begins when the run container starts. The Kubernetes clock begins when the task container starts, so it includes setup commands. This difference follows the current backend lifecycle. Kubernetes users must include setup time in `startup_timeout`.

The worker-daemon path extends the task monitor at [`task_attempt.py:808`](https://github.com/marin-community/marin/blob/442825abf3939d971c6f12d48af1ce835556d0dd/lib/iris/src/iris/cluster/worker/task_attempt.py#L808). It requests `http://127.0.0.1:<allocated-port>/healthz` with the configured timeout. At the threshold, it saves one `TASK_STATE_FAILED` result with the health error and stops the container.

A worker restart restores the health config from Docker labels. The monitor writes a live-phase marker after the first success. Adoption reads this marker. Without a marker, adoption keeps the original startup deadline from the container start time. It never grants a new startup window.

The Kubernetes path adds `startupProbe` and `livenessProbe` to the task Pod. Each probe runs the Iris health helper inside the task container. The helper reads `/tmp/iris/health-port` and requests localhost `/healthz`. This keeps Kubernetes-native probe control and supports dynamic ports with `hostNetwork`.

The Kubernetes helper keeps a local live-failure count for diagnostic output. On the threshold failure, it writes a bounded message to a dedicated termination file. The startup helper never writes this file. Because task Pods use `restartPolicy: Never`, kubelet termination makes the Pod fail. The existing Kubernetes backend then reports an application failure and applies `max_retries_failure`.

Do not add a readiness probe. Iris tasks are batch processes and do not receive traffic through Kubernetes Services. Do not route health checks through the controller proxy. A controller fault must not cause task failure.

`TaskHealthCheck` moves from `LaunchJobRequest` to `RunTaskRequest` through `job_config`. This is the same path as ports and timeouts. Fray adds the same optional field to `JobRequest` and maps it in `FrayIrisClient`.

Levanter keeps the fixed JSON report from PR #8554. `TrainingDashboard` binds the Iris health port and calls `publish_task_health`. One local process leader starts the server in each Iris task. Global JAX process zero also registers that server as the `training-control` endpoint and enables checkpoint requests. Other task leaders serve only local health traffic.

Health server startup is required when the task opts in. Bind and publication errors stop the task. Public endpoint registration stays best-effort and cannot close a working health server. The health server does not depend on checkpoint support.

`ProgressWatchdogConfig.create` will arm every JAX process. Only global process zero will capture the expensive diagnostic. Each task leader will then have local watchdog state for its health route.

The Levanter watchdog remains the first termination path. Iris is an outer dead-man switch for a blocked HTTP thread or a blocked process. The Iris failure window must exceed the watchdog poll interval and diagnostic budget.

If the application exits before the Iris threshold, the existing exit path owns the result. Iris keeps exit code 124 and does not replace it with a health-probe error.

For a coscheduled job, one unhealthy attempt restarts the gang through the current coscheduling path. Sibling attempts become `COSCHED_FAILED`. Only the unhealthy attempt charges the job failure budget.

The first rollout enables this contract for the Grug EP hero dispatcher. Use a 30-minute startup timeout, a 10-second period, a 3-second request timeout, and 13 consecutive failures. This gives Levanter at least 120 seconds after the first failure. Other Iris jobs remain unchanged until their submitters add a health contract.

## Costs / Risks

- A bad health function can kill useful work and consume the task retry limit.
- Each enabled task adds a small HTTP request every probe period.
- Kubernetes and worker-daemon errors come from different runtime paths. Tests must keep their failure boundaries equal.
- The change adds one persisted job-config field and one controller migration.
- A short Iris deadline can interrupt application diagnostics and replace a useful exit code.
- Enabled Kubernetes tasks reserve a termination-message file for Iris health diagnostics.

## Testing

Unit tests cover the startup and consecutive-failure state machine with a fake clock. They cover response success, connection failure, counter reset, and process-exit races.

Worker tests use a local HTTP server. They cover startup failure, 503 failure, terminal error text, process-exit races, and worker adoption.

Kubernetes manifest tests compare the configured startup and liveness probes with the job contract. Probe-helper tests cover failure diagnostics and recovery.

Levanter tests verify one server per Iris task leader, process-zero endpoint registration, and required health startup without checkpoint support.

## Open Questions

- Is 30 minutes sufficient for task setup and server start on Kubernetes?
- Should the first rollout include all Grug training jobs or only the EP hero?
- Should Iris later store live health state for the dashboard, or is the terminal task error sufficient?
