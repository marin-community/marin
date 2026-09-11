# Container Security Profiles

A job's **container profile** selects a named bundle of container/pod security
settings instead of exposing individual docker/k8s knobs. Five profiles exist
(defined in [`job.proto`](../src/iris/rpc/job.proto)), ordered by privilege:

| Profile | Selected via | Behavior |
|---|---|---|
| `CONTAINER_PROFILE_RESTRICTED` | `--container-profile CONTAINER_PROFILE_RESTRICTED` | Hardened: drops all Linux capabilities, blocks privilege escalation, keeps the default seccomp profile. No profiling cap. For untrusted/sandboxed workloads. |
| `CONTAINER_PROFILE_DEFAULT` | default (or `--container-profile CONTAINER_PROFILE_DEFAULT`) | `SYS_PTRACE` for profiling (plus `SYS_RESOURCE` on TPU). The everyday training/eval pod. |
| `CONTAINER_PROFILE_GVISOR` | `--container-profile CONTAINER_PROFILE_GVISOR` | Runs the whole container under the gVisor runtime (docker `--runtime=runsc` / k8s `runtimeClassName: gvisor`). In-container root gets the docker default capability set (so `setuid`/`apt` work), while the intercepted guest kernel isolates the host. **Not elevated** — safe to grant without admin. Requires `runsc` installed on the worker/node; CPU-only (no GPU/TPU passthrough). |
| `CONTAINER_PROFILE_DOCKER_ACCESS` | `--container-profile CONTAINER_PROFILE_DOCKER_ACCESS` | DEFAULT **plus** the host docker socket (`/var/run/docker.sock`) — lets the container drive the host Docker daemon to build images or run sibling containers. **Elevated.** |
| `CONTAINER_PROFILE_PRIVILEGED` | `--container-profile CONTAINER_PROFILE_PRIVILEGED` | Full `--privileged` / `securityContext.privileged` with broad capabilities. Needed to run nested runtimes inside the container (e.g. a gVisor `runsc` sandbox). **Elevated.** |

`CONTAINER_PROFILE_UNSPECIFIED` (the wire default) resolves to
`CONTAINER_PROFILE_DEFAULT`. The CLI choice is case-insensitive.

## Elevated profiles require authorization

`DOCKER_ACCESS` and `PRIVILEGED` are **host-root-equivalent**: a mounted docker
socket can launch a privileged container that mounts the host filesystem, and a
privileged container can escape to the host directly. They are therefore gated
at submission:

- **With an auth provider configured:** only the `admin` role may submit an
  elevated profile (a trusted-loopback caller resolves to admin). Everyone else
  gets `PERMISSION_DENIED`.
- **In null-auth mode (no provider):** every caller is the anonymous admin, so
  the gate is a no-op and elevated profiles are allowed — the operator has
  already opted into an untrusted cluster. This mirrors how the `PRODUCTION`
  priority band behaves in null-auth.

`RESTRICTED` and `DEFAULT` are unprivileged and need no authorization — anyone
may use them. `RESTRICTED` is strictly safer than the default.

The numeric ordering is used only to decide whether a profile is *elevated*; it
is **not** a relative-danger ladder. `DOCKER_ACCESS` and `PRIVILEGED` are
distinct dangerous capabilities — neither implies the other.

## How profiles map to each backend

The accepted profile is stamped on the job by the controller after the
authorization check, carried on the dispatched `RunTaskRequest`, and applied by
whichever backend runs the task.

### Docker worker backend

| Profile | Docker flags |
|---|---|
| `RESTRICTED` | `--cap-drop ALL --security-opt no-new-privileges` (default seccomp applies; no `SYS_PTRACE`) |
| `DEFAULT` | `--cap-drop ALL --cap-add SYS_PTRACE --security-opt no-new-privileges` |
| `DOCKER_ACCESS` | DEFAULT **+** `-v /var/run/docker.sock:/var/run/docker.sock` |
| `PRIVILEGED` | `--privileged --cap-add SYS_PTRACE` |

**TPU note:** a TPU task is always `--privileged` for device passthrough,
regardless of profile. On TPU, `RESTRICTED`/`DEFAULT` cannot fully sandbox the
container — the effective profile is privileged. This is logged when the
container is created.

### Kubernetes backend

| Profile | `securityContext` |
|---|---|
| `RESTRICTED` | `capabilities.drop:[ALL]`, `allowPrivilegeEscalation:false`, `seccompProfile.type:RuntimeDefault` (not full PSS Restricted — `runAsNonRoot` is not forced) |
| `DEFAULT` | `capabilities.add:[SYS_PTRACE (+SYS_RESOURCE on TPU)]` |
| `GVISOR` | DEFAULT `securityContext` (no privilege) plus `spec.runtimeClassName: gvisor` on the pod — the node RuntimeClass provides isolation, not the container context |
| `DOCKER_ACCESS` | **rejected** — k8s nodes run containerd, not dockerd, so there is no host docker socket. Use the docker worker backend, or `PRIVILEGED` with an in-pod runtime. |
| `PRIVILEGED` | `privileged:true`, `allowPrivilegeEscalation:true`, plus the DEFAULT caps |

## Running under gVisor

The `GVISOR` profile runs the **whole** container under gVisor's `runsc` runtime
(docker `--runtime=runsc`; k8s `runtimeClassName: gvisor`). The worker/node's
container runtime — running as root — builds the sandbox, so the container itself
needs no privilege: in-container root gets the docker default capability set
(`setuid`, `apt`, etc. work) while the intercepted guest kernel isolates the
host. This makes it safe to hand out **without** the admin role, unlike
`PRIVILEGED`. It is CPU-only — `runsc` cannot pass a GPU/TPU through, so
accelerator tasks are rejected at submit.

**Operator prerequisite:** `runsc` must be installed and registered on the
workers — as a docker runtime named `runsc` in `/etc/docker/daemon.json` on
docker-worker clusters, or as a `RuntimeClass` named `gvisor` on k8s. Without it,
a `GVISOR` job fails at container creation.

To instead run an *inner* gVisor sandbox around an untrusted child image while
keeping the outer container normal, use `CONTAINER_PROFILE_PRIVILEGED` and run
`runsc` (or `docker run --runtime=runsc ...`) yourself inside the container; the
parent must be privileged because nested `runsc` creates user/mount namespaces.

## See also

- [`priority-bands.md`](priority-bands.md) — the parallel admin-gated job knob
- [`auth-loopback-transition.md`](auth-loopback-transition.md) — how loopback
  callers resolve to the admin role
