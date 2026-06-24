# Container Security Profiles

A job's **container profile** selects a named bundle of container/pod security
settings instead of exposing individual docker/k8s knobs. Four profiles exist
(defined in [`job.proto`](../src/iris/rpc/job.proto)), ordered by privilege:

| Profile | Selected via | Behavior |
|---|---|---|
| `RESTRICTED` | `--container-profile restricted` | Hardened: drops all Linux capabilities, blocks privilege escalation, keeps the default seccomp profile. No profiling cap. For untrusted/sandboxed workloads. |
| `DEFAULT` | default (or `--container-profile default`) | Today's behavior: `SYS_PTRACE` for profiling (plus `SYS_RESOURCE` on TPU). The everyday training/eval pod. |
| `DOCKER_ACCESS` | `--container-profile docker_access` | DEFAULT **plus** the host docker socket (`/var/run/docker.sock`) — lets the container drive the host Docker daemon to build images or run sibling containers. **Elevated.** |
| `PRIVILEGED` | `--container-profile privileged` | Full `--privileged` / `securityContext.privileged` with broad capabilities. Needed to run nested runtimes inside the container (e.g. a gVisor `runsc` sandbox). **Elevated.** |

`UNSPECIFIED` (the wire default) resolves to `DEFAULT`.

## Elevated profiles require authorization

`DOCKER_ACCESS` and `PRIVILEGED` are **host-root-equivalent**: a mounted docker
socket can launch a privileged container that mounts the host filesystem, and a
privileged container can escape to the host directly. They are therefore gated
at submission:

- **With an auth provider configured:** only the `admin` role may submit an
  elevated profile (a trusted-loopback caller resolves to admin). Everyone else
  gets `PERMISSION_DENIED`.
- **In null-auth mode (no provider):** elevated profiles are **rejected by
  default**. An operator running a trusted single-tenant cluster can opt in with
  `auth.allow_unauthenticated_elevated_profiles: true` in the cluster config.

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
| `DOCKER_ACCESS` | **rejected** — k8s nodes run containerd, not dockerd, so there is no host docker socket. Use the docker worker backend, or `PRIVILEGED` with an in-pod runtime. |
| `PRIVILEGED` | `privileged:true`, `allowPrivilegeEscalation:true`, plus the DEFAULT caps |

## Running gVisor inside a container

To run a gVisor (`runsc`) sandbox to isolate an untrusted child workload, submit
with `--container-profile privileged` and run `runsc` (or
`docker run --runtime=runsc ...`) inside your container. The parent must be
privileged because `runsc` creates user/mount namespaces. (Running the *whole*
pod under a gVisor `RuntimeClass` is a separate, operator-side feature and
requires `runsc` installed on the nodes.)

## See also

- [`priority-bands.md`](priority-bands.md) — the parallel admin-gated job knob
- [`auth-loopback-transition.md`](auth-loopback-transition.md) — how loopback
  callers resolve to the admin role
