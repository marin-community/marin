# Iris Agent Notes

- Use Connect/RPC for APIs and dashboards. Do not use `httpx` or raw HTTP.
- After changing `.proto` files, regenerate via `scripts/generate_protos.py`.
- Prefer shallow, functional code that returns control quickly; avoid callback-heavy or inheritance-driven designs.

The Iris testing policy lives in `TESTING.md`.

## Docs

Read the docs for the area you are changing. If docs disagree with code, update the docs (or add a task) in the same PR.

Key docs (kept intentionally short):

- `README.md` — overview + quick start
- `OPS.md` — operating / troubleshooting a live cluster
- `docs/autoscaler-v2.md` — autoscaler design + terminology
- `docs/controller-flow.md`, `docs/worker-flow.md` — controller/worker lifecycle
- `docs/task-states.md` — task state machine + retry semantics
- `docs/coreweave.md` — CoreWeave platform + `runtime=kubernetes` behavior

Resource model note: CPU demand is fungible and can route to any group; GPU/TPU demand is non-fungible and must match device type (and optionally variant). Priority configuration determines whether CPU spillover lands on accelerator groups.

## Imports

Avoid `TYPE_CHECKING`. Use real imports. If you hit a cycle:

- Prefer refactoring when sensible.
- Otherwise use a `Protocol` at the boundary.

## RPC / API Accessibility

Any functionality exposed by the worker or controller dashboards must also be available via RPC.
Dashboards should be a thin UI over the RPC API, not a second implementation path.

## Concurrency Model

Platform operations (`terminate`, `create_slice`, etc.) shell out via `subprocess.run` and are thread-safe.
For concurrent independent platform operations, use `concurrent.futures.ThreadPoolExecutor` (not asyncio) and apply hard timeouts so the CLI cannot hang indefinitely.

## Planning

Prefer spiral plans over linear plans: each stage should be independently testable (proto → server stub → client wiring → end-to-end test), then iterate.

### Light Worker Mode (CoreWeave + runtime=kubernetes)

When `runtime: kubernetes` is configured, worker Pods are intentionally "light":
- Worker Pod must not request `nvidia.com/gpu` or `rdma/ib`.
- Task Pods created by `src/iris/cluster/runtime/kubernetes.py` request accelerators per task.
- Worker Pod still uses the scale-group `nodeSelector` and `hostNetwork: true`.
- Worker Pod passes control-plane env needed for task-pod creation (for example
  `IRIS_SERVICE_ACCOUNT_NAME`, and `IRIS_S3_SECRET_NAME` when S3 is enabled).

Quick verification:
- Worker create log shows `resource_limits=none`.
- `kubectl get pod <worker> -o jsonpath='{.spec.containers[0].resources}'` is empty.
- Task pod specs include GPU limits when task resources request GPUs.

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

**Docker registries**: Bootstrap scripts in `src/iris/cluster/platform/bootstrap.py` auto-detect
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

**Zone validation**: `src/iris/cluster/config.py` validates that every scale group zone
appears in `platform.gcp.zones`. Multi-zone scale groups are auto-expanded by
`_expand_multi_zone_groups()`.

## Testing

See `TESTING.md` for the testing policy and commands.
