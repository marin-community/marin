# Agent Tips

* Use the connect/RPC abstractions to implement and perform RPC calls. DO NOT use httpx or raw HTTP.
* Use scripts/generate_protos.py to regenerate files after changing the `.proto` files.
* Prefer _shallow_, _functional_ code which returns control quickly to the user, vs callbacks or inheritance.

```
class Scheduler:
  def add_job():
  def add_worker():
  def compute_schedule() -> ScheduledJobs:

class Runner:
  def run_jobs(ScheduledJobs)
```

is preferable to:

```
class Scheduler:
  def __init__(self, job_creator: JobCreator):
    self.job_creator = job_creator
  def run(self):
    ... self.job_creator.create_job()
```

See [TESTING.md](TESTING.md) for the full testing policy.

## Documentation

ALWAYS read the docs for the appropriate area.
IF they disagree with the code, ALWAYS add a task to update them.

Documentation should be kept up-to-date as code changes. When implementing new features or making significant changes, update the relevant documentation files:

@README.md - Main overview, CLI reference, and quick start

## Operations

When troubleshooting, monitoring, or deploying a live Iris cluster, read [OPS.md](OPS.md) first.

## Imports

Don't use TYPE_CHECKING. Use the real import. If there is a circular dependency:

* Prefer to resolve it with refactoring when sensible
* Otherwise use a protocol if you simply need the type information

## RPC/API Accessibility

Any functionality exposed by the worker or controller dashboards must also be
available via RPC. The dashboards should be a friendly interface on top of the
machine accessible RPC API, and should not use internal APIs (except for
efficiency). For example, if we wanted to show the scheduling status for a task,
we should define a new RPC endpoint `/TestSchedule(task_id)` and use that from
the dashboard, rather than creating a scheduler and running it manually.

## Architecture Notes

### Concurrency Model

Platform operations (`terminate`, `create_slice`, etc.) shell out to `gcloud`
via `subprocess.run` and are thread-safe. When multiple independent platform
operations need to run (e.g. tearing down N slices), use
`concurrent.futures.ThreadPoolExecutor` — not asyncio. Always apply a hard
timeout so the CLI doesn't hang on a stuck gcloud call.

## Planning

Prefer _spiral_ plans over _linear_ plans. e.g. when implementing a new feature, make a plan which has step 1 as:

Step 1:
* Add the minimal changes to the `.proto` file
* Add the server stub code
* Add a client wiring
* Add an end-to-end test

Step 2:
* Extend proto with additional field
* Update server code
* Update client code
* Update tests

...

That is _each stage of the plan_ should be a independently testable,
self-contained unit of work. THis is preferable to plans which attempt to make
all of the changes for one area (e.g. all proto changes, then all server
changes, etc.)

When adding new modules or significant features:
1. Update the README with a brief overview and usage examples
2. Add detailed documentation to the appropriate docs/ file
3. Reference the documentation from this AGENTS.md file

**Key documentation areas:**

| Area | File | Description |
|------|------|-------------|
| Architecture | README.md | High-level architecture, CLI reference, quick start |
| Autoscaler Design | docs/autoscaler-v0-design.md | Technical specification, threading model |
| Thread Safety | docs/thread-safety.md | Thread management, test synchronization best practices |
| Original Design | docs/fray-zero.md | Rationale and design decisions |
| Task States | docs/task-states.md | Task state machine, transitions, retry semantics, dashboard display |
| User-Aware Job IDs | docs/users.md | User-prefixed job naming, user inference, and dashboard user aggregates |
| CoreWeave Integration | (below) | Platform, runtime, and networking for CoreWeave bare metal |

### CoreWeave Integration

Iris runs on CoreWeave bare-metal GPU nodes. The integration spans three layers:

**Platform** (`cluster/platform/coreweave.py`): Uses shared NodePools with CoreWeave autoscaling.
`iris cluster start` creates one NodePool per scale group (plus a controller pool) via
`ensure_nodepools()`. NodePool names are derived from config: `{label_prefix}-{scale_group_name}`.
Each NodePool has `autoscaling: true` and `targetNodes: 0`; CoreWeave provisions nodes on demand
when Pods are scheduled. Iris manages only Pods; cleanup/stop leaves NodePools alone (they scale
to zero when idle). The controller runs as a Deployment with a `ClusterIP` Service
(`iris-controller-svc`). Workers discover the controller via in-cluster DNS
(`iris-controller-svc.iris.svc.cluster.local:10000`).

**Runtime**: `kubernetes` (`cluster/runtime/kubernetes.py`). Each task is a separate K8s Pod that
can be scheduled independently (not co-located with the worker). Task Pods claim GPU/RDMA resources
directly from the device plugin. The worker Pod must **not** request GPU/RDMA resources (the
platform skips them when `runtime: kubernetes`). Task Pods get `hostNetwork`, tolerations,
service account, S3 secret refs, and an emptyDir-backed UV cache automatically.
Docker is not available on CoreWeave bare metal.

**Networking**: All traffic stays inside the CoreWeave VPC. Worker Pods use `hostNetwork: True`
(bypassing the Kubernetes overlay for RDMA/GPU performance). Task Pods set
`hostNetwork: True` + `dnsPolicy: ClusterFirstWithHostNet` in their Pod spec.

### Light Worker Mode (CoreWeave + runtime=kubernetes)

When `runtime: kubernetes` is configured, worker Pods are intentionally "light":
- Worker Pod must not request `nvidia.com/gpu` or `rdma/ib`.
- Task Pods created by `cluster/runtime/kubernetes.py` request accelerators per task.
- Worker Pod still uses the scale-group `nodeSelector` and `hostNetwork: true`.
- Worker Pod passes control-plane env needed for task-pod creation (for example
  `IRIS_SERVICE_ACCOUNT_NAME`, and `IRIS_S3_SECRET_NAME` when S3 is enabled).

Quick verification:
- Worker create log should show `resource_limits=none`.
- `kubectl get pod <worker> -o jsonpath='{.spec.containers[0].resources}'` should be empty.
- Task pod specs should include GPU limits when task resources request GPUs.

**Disk layout**: CoreWeave bare-metal nodes have a 15 GB RAM disk (`/dev/ram0`) as the root
filesystem and a multi-TB NVMe RAID (`/dev/md127`) mounted at `/mnt/local`. Bind mounts expose
it as `/var/lib/containerd`, `/var/lib/kubelet`, `/opt`, etc. The `cache_dir` must point to the
NVMe (e.g. `/mnt/local/iris-cache`) — the default `/var/cache/iris` lands on the tiny RAM disk
and will fill up immediately when installing CUDA packages.

All K8s resources (RBAC, ConfigMap, shared NodePools, Deployment, Service) are created
automatically by `iris cluster start` via `CoreweavePlatform.start_controller()`. RBAC
manifests (Namespace, ServiceAccount, ClusterRole, ClusterRoleBinding) are defined in
`CoreweavePlatform.ensure_rbac()` — no separate YAML files needed.

## Key Modules

### Time Utilities

Use `iris.time_utils` for all time-related operations instead of raw `datetime` or `time`:

| Class | Purpose |
|-------|---------|
| `Timestamp` | Point in time (epoch-based). Use for created_at, timestamps in logs, etc. |
| `Duration` | Time interval. Use for timeouts, intervals, configuration values. |
| `Deadline` | Monotonic deadline for timeout checks. Use in polling loops. |
| `Timer` | Elapsed time measurement. Use for performance tracking. |
| `ExponentialBackoff` | Retry/polling with backoff. Use `wait_until()` for condition polling. |

Example:
```python
from iris.time_utils import Timestamp, Duration, Deadline

created_at = Timestamp.now()
timeout = Duration.from_seconds(30.0)
deadline = Deadline.from_now(timeout)
deadline.wait_for(condition)

while not deadline.expired():
    if condition():
        break
    time.sleep(0.1)
```

### Deployment Topology

The controller is a plain GCE VM with no zone affinity to workers — it can run
in any zone and serve workers across all regions.

**When changing the controller zone**, update in `examples/marin.yaml`:
- `controller.gcp.zone` — the GCE zone
- Image tags use `ghcr.io/marin-community/...` format. The controller and
  autoscaler automatically rewrite these to AR remote repos for the VM's
  continent at boot time.

**Docker registries**: Bootstrap scripts in `platform/bootstrap.py` auto-detect
AR image tags and configure `gcloud auth configure-docker`. AR remote repos
proxy GHCR — see `docs/image-push.md` for setup.

### Multi-Region Image Push/Pull

Images are pushed only to **GHCR** (`ghcr.io/marin-community/`). GCP VMs pull
from **Artifact Registry remote repositories** that act as pull-through caches
for GHCR. See `docs/image-push.md` for full details.

**Push**: `iris build push` and `iris cluster start` push to GHCR only.

**Pull**: The autoscaler and controller bootstrap automatically rewrite GHCR
image tags to the AR remote repo for the VM's continent:
- `ghcr.io/org/image:v1` → `us-docker.pkg.dev/project/ghcr-mirror/org/image:v1`

Set `defaults.worker.docker_image` to a `ghcr.io/...` tag. Non-GHCR tags
(`docker.io`, existing AR tags) pass through unchanged.

**Bundle storage** (`controller.bundle_prefix`) is a GCS URI with no zone
affinity — globally accessible.

**Zone validation**: `cluster/config.py` validates that every scale group zone
appears in `platform.gcp.zones`. Multi-zone scale groups are auto-expanded by
`_expand_multi_zone_groups()`.

### Architecture Layers

Iris follows a clean layering architecture:

**Controller layer** (`cluster/controller/`): Task scheduling, autoscaling, and demand routing
- Depends on Platform layer for VM abstractions (Platform, SliceHandle, VmHandle)
- Owns autoscaling logic and scaling group state

**Platform layer** (`cluster/platform/`): Platform abstractions for managing VMs
- Does NOT depend on controller layer
- Four platform implementations with independent launch/teardown paths:
  - `gcp.py` — GCP TPU/VM slices, SSH bootstrap
  - `coreweave.py` — CoreWeave CKS, Kubernetes Pods on shared NodePools
  - `manual.py` — Pre-existing hosts, SSH bootstrap
  - `local.py` — Local development, in-process workers
- Changes to shared interfaces (worker CLI, bootstrap flow, proto schemas)
  must be applied to all four platforms

**Cluster layer** (`cluster/`): High-level orchestration
- `connect_cluster()` and `stop_all()` free functions for cluster lifecycle
- `stop_all()` terminates controller + all slices in parallel via ThreadPoolExecutor
  with a 60s hard timeout. Timed-out operations are logged at WARNING and abandoned.
- Configuration and platform abstractions

Key files:

```
src/iris/
├── cli/                         # CLI package (cluster, build, run, debug commands)
│   ├── main.py                  # Top-level iris group
│   ├── cluster.py               # Cluster lifecycle, controller, VM ops, dashboard
│   ├── build.py                 # Image build commands
│   ├── run.py                   # Command passthrough job submission
│   └── rpc.py                   # Dynamic RPC CLI
├── cluster/
│   ├── config.py                # General Iris configuration (load_config, IrisConfig)
│   ├── manager.py               # connect_cluster() + stop_all(dry_run) free functions
│   ├── controller/
│   │   ├── controller.py        # Controller with integrated autoscaler
│   │   ├── main.py              # Controller daemon CLI (serve command)
│   │   ├── autoscaler.py        # Core autoscaling logic and demand routing
│   │   ├── scaling_group.py     # Per-group state tracking and lifecycle
│   │   ├── config.py            # Autoscaler factory functions
│   │   ├── local.py             # LocalController for in-process testing
│   │   └── vm_lifecycle.py      # Controller lifecycle (start/stop/reload via Platform)
│   └── platform/
│       ├── base.py              # Platform protocol and SliceHandle/VmHandle
│       ├── gcp.py               # GCP TPU platform
│       ├── manual.py            # Pre-existing host platform
│       ├── local.py             # Local development platform
│       ├── coreweave.py         # CoreWeave CKS platform (shared NodePools)
│       ├── bootstrap.py         # Worker bootstrap script generation
│       ├── ssh.py               # SSH connection management
│       ├── factory.py           # Platform factory from config
│       └── debug.py             # Platform debugging utilities
```

See [README.md](README.md) for CLI usage and configuration examples.

### Dashboard Frontend

The controller and worker dashboards are client-side SPAs using Preact + HTM.

**Directory structure:**
```
src/iris/cluster/static/
├── controller/          # Controller dashboard
│   ├── app.js           # Main app (tabs, cluster summary, data fetching)
│   ├── jobs-tab.js      # Jobs table with pagination/sorting/tree view
│   ├── job-detail.js    # Job detail page with task list
│   ├── fleet-tab.js     # Fleet tab: worker health table with inline gauges
│   ├── worker-detail.js # Worker detail page: live resource gauges, task history, logs
│   ├── workers-tab.js   # Workers table
│   └── vms-tab.js       # VM management table
├── shared/              # Shared utilities and components
│   ├── components.js    # Reusable Preact components (MetricCard, Gauge, etc.)
│   ├── rpc.js           # Connect RPC client wrapper
│   ├── utils.js         # Formatting (dates, durations, bytes)
│   └── styles.css       # Consolidated CSS with design tokens
├── vendor/              # Third-party ES modules (vendored, not npm)
│   ├── preact.mjs       # UI framework
│   └── htm.mjs          # HTML template literals
└── worker/              # Worker dashboard components
    ├── app.js            # Worker dashboard: task list, aggregate resources
    └── task-detail.js    # Task detail: resource usage, auto-refresh
```

**Key patterns:**
- All data fetched via Connect RPC (e.g., `ListJobs`, `GetWorkerStatus`)
- No REST endpoints — RPC only. New dashboard features MUST have a backing RPC.
- State management with Preact hooks (`useState`, `useEffect`, `useCallback`)
- HTML templates via `htm.bind(h)` tagged template literals
- Auto-refresh: active pages poll their RPC endpoint (5s for worker/task detail, 30s for controller overview)
- Jobs displayed as a hierarchical tree based on name structure

#### Design System

The dashboard uses CSS custom properties (design tokens) defined in `:root` at the
top of `shared/styles.css`. All new styles should reference these tokens rather
than hard-coding colors, fonts, or shadows.

| Token family     | Examples                                    | Purpose                         |
|------------------|---------------------------------------------|---------------------------------|
| `--color-*`      | `--color-accent`, `--color-danger`          | Semantic colors                 |
| `--font-*`       | `--font-sans`, `--font-mono`                | Typography stacks               |
| `--radius-*`     | `--radius-sm`, `--radius-md`                | Border radius scale             |
| `--shadow-*`     | `--shadow-sm`, `--shadow-md`                | Elevation / depth               |

#### Shared Components (`shared/components.js`)

| Component        | Purpose                                               |
|------------------|-------------------------------------------------------|
| `MetricCard`     | Prominent number + label tile (e.g., "3 Running Tasks") |
| `Gauge`          | Horizontal bar gauge with ok/warning/danger thresholds |
| `InlineGauge`    | Compact gauge for use inside table cells               |
| `ResourceSection`| Titled wrapper for a group of Gauge bars               |
| `InfoRow`        | Simple label/value pair                                |
| `InfoCard`       | Card container with title                              |

When adding new dashboard components, add them to `shared/components.js` if
they are reusable across pages. Page-specific components (e.g., `StatusBadge`)
live in the page's own JS file.

#### Resource Display Conventions

- **Live utilization** from heartbeat snapshots (`WorkerResourceSnapshot`) is shown
  as Gauge bars with ok/warning/danger color thresholds (70%/90% by default).
- **Static capacity** from worker metadata (CPU cores, total memory) is shown as
  MetricCard tiles or Field label/value pairs.
- When live data is available, prefer gauges over raw numbers. Fall back to static
  capacity display when no heartbeat snapshots exist yet.
- Use `formatBytes()` for human-readable byte values. Use `formatRelativeTime()`
  for timestamps.

#### Navigation Flow

```
Controller Dashboard (/)
  ├── Jobs tab (default) → click job row → /job/{jobId}
  │     └── Job Detail → task list, task logs, resource usage
  ├── Fleet tab → click worker row → /worker/{id}
  │     └── Worker Detail → identity, live resources, task history, logs
  ├── Endpoints tab
  ├── Autoscaler tab
  ├── Logs tab
  └── Transactions tab

Worker Dashboard (:worker_port/)
  ├── Task list with aggregate resource summary
  └── Click task → /task/{taskId} → Task Detail (auto-refreshing)
```

**When modifying the dashboard:**
1. Run dashboard tests: `uv run pytest lib/iris/tests/e2e/test_dashboard.py -x -o "addopts="`
2. New UI features MUST have a corresponding RPC endpoint — no internal API calls
3. Follow existing component patterns (functional components, hooks)
4. Preserve CSS class names that E2E tests rely on (`.worker-detail-grid`, `.tab-btn`, `#log-container`, etc.)
5. Use design tokens from `:root` — do not hard-code colors or fonts

## Testing

See [TESTING.md](TESTING.md) for the full testing policy, E2E fixtures,
and run commands.

## Debugging Container Failures

**Exit code 137** = 128 + 9 = SIGKILL, typically OOM. Check:
- `ContainerStatus.oom_killed` field (from `docker inspect .State.OOMKilled`)
- Job's `resources.memory_bytes` vs what was requested
- Resource flow: `JobRequest.resources` → Iris protobuf → `ContainerConfig` → `docker --memory`

**Resource propagation path:**
```
fray.v2.ResourceConfig → iris.cluster.types.ResourceSpec.to_proto()
  → cluster_pb2.ResourceSpecProto → docker.py _docker_create() --memory/--cpus
```

**Key files for container debugging:**
- `cluster/runtime/docker.py`: Docker CLI wrapper, resource limits at lines 396-403
- `cluster/runtime/types.py`: ContainerStatus with oom_killed field
- `cluster/worker/task_attempt.py`: _format_exit_error() interprets signals
