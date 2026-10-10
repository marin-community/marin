# Container Security Profiles

A job's **container profile** selects a named bundle of container/pod security
settings instead of exposing individual docker/k8s knobs. Profiles are defined
in [`job.proto`](../src/iris/rpc/job.proto):

| Profile | Selected via | Behavior |
|---|---|---|
| `CONTAINER_PROFILE_RESTRICTED` | `--container-profile CONTAINER_PROFILE_RESTRICTED` | Hardened: drops all Linux capabilities, blocks privilege escalation, keeps the default seccomp profile. No profiling cap. For untrusted/sandboxed workloads. |
| `CONTAINER_PROFILE_DEFAULT` | default (or `--container-profile CONTAINER_PROFILE_DEFAULT`) | `SYS_PTRACE` for profiling (plus `SYS_RESOURCE` on TPU). The everyday training/eval pod. |
| `CONTAINER_PROFILE_SANDBOX` | `--container-profile CONTAINER_PROFILE_SANDBOX` | For model-controlled workloads. Runs the whole container under the gVisor runtime (docker `--runtime=runsc` / k8s `runtimeClassName: gvisor`). The task receives only the job's own `env_vars` and Iris task identity: no cluster `task_env`, injected secrets, or object-store keys; no controller address or task token; no node-shared caches; no Kubernetes service account token. Its [egress policy](#egress-policy) defaults to `INTERNET` and cannot be `CLUSTER`. The submitting client sends no workspace bundle and copies nothing from its own environment or a parent job's. **Not elevated.** CPU-only. See [Sandbox jobs](#sandbox-jobs). |
| `CONTAINER_PROFILE_DOCKER_ACCESS` | `--container-profile CONTAINER_PROFILE_DOCKER_ACCESS` | DEFAULT **plus** the host docker socket (`/var/run/docker.sock`) — lets the container drive the host Docker daemon to build images or run sibling containers. **Elevated.** |
| `CONTAINER_PROFILE_PRIVILEGED` | `--container-profile CONTAINER_PROFILE_PRIVILEGED` | Full `--privileged` / `securityContext.privileged` with broad capabilities. Needed to run nested runtimes inside the container (e.g. a gVisor `runsc` sandbox). **Elevated.** |

`CONTAINER_PROFILE_UNSPECIFIED` (the wire default) resolves to
`CONTAINER_PROFILE_DEFAULT`. The CLI choice is case-insensitive.

### Temporary compatibility for legacy gVisor clients

`CONTAINER_PROFILE_GVISOR` (wire value 5) is temporarily accepted for running
clients that predate `SANDBOX`. It keeps the gVisor runtime and the original
behavior: workspace bundles, inherited environment, cluster credentials,
shared caches, service account, and `CLUSTER` egress by default. On Kubernetes
it uses CNI pod networking for cluster egress, even when GPU tasks use host
networking, and keeps the log-shipper and output-uploader sidecars. It is CPU-only and
does not require an elevated role. It does not provide `SANDBOX`'s isolation
from cluster services or credentials.

New clients must use `SANDBOX`. The controller logs each legacy submission with
`uses deprecated CONTAINER_PROFILE_GVISOR` and its job ID; use those records
to find remaining callers. The controller rollout owner removes the deprecated
profile and its runtime branches after those submitters have upgraded and
all their profile-5 jobs have finished. Migration 0052 already converted
stored GVISOR jobs to SANDBOX; this compatibility path does not reverse that
migration or change the behavior of profile 6.

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
| `SANDBOX` | `--runtime runsc`; the worker's `task_env` and controller address are withheld and cache bind mounts are omitted |
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
| `SANDBOX` | DEFAULT `securityContext` (no privilege) plus `spec.runtimeClassName: gvisor` on the pod — the node RuntimeClass provides isolation, not the container context — and `automountServiceAccountToken: false`, no `serviceAccountName`, no `envFrom` task-env Secret, no cluster `task_env`, and no node cache `hostPath` mounts (cache paths stay in the container layer) |
| `DOCKER_ACCESS` | **rejected** — k8s nodes run containerd, not dockerd, so there is no host docker socket. Use the docker worker backend, or `PRIVILEGED` with an in-pod runtime. |
| `PRIVILEGED` | `privileged:true`, `allowPrivilegeEscalation:true`, plus the DEFAULT caps |

## Running under gVisor

`SANDBOX` runs the **whole** container under gVisor's `runsc` runtime
(docker `--runtime=runsc`; k8s `runtimeClassName: gvisor`). The worker/node's
container runtime — running as root — builds the sandbox, so the container itself
needs no privilege: in-container root gets the docker default capability set
(`setuid`, `apt`, etc. work) while the intercepted guest kernel isolates the
host. This makes it safe to hand out **without** the admin role, unlike
`PRIVILEGED`. It is CPU-only — `runsc` cannot pass a GPU/TPU through, so
accelerator tasks are rejected at submit.

**Operator prerequisite:** `runsc` must be installed and registered on the
workers — as a docker runtime named `runsc` in `/etc/docker/daemon.json` on
docker-worker clusters (worker bootstrap does this), or as a `RuntimeClass`
named `gvisor` on k8s. Without it, a `SANDBOX` job fails at container creation.

To instead run an *inner* gVisor sandbox around an untrusted child image while
keeping the outer container normal, use `CONTAINER_PROFILE_PRIVILEGED` and run
`runsc` (or `docker run --runtime=runsc ...`) yourself inside the container; the
parent must be privileged because nested `runsc` creates user/mount namespaces.

## Sandbox jobs

`SANDBOX` is for a job whose commands come from a model or another untrusted
source, such as a shellbox machine. A task normally assembles its environment
from three places: the cluster (`defaults.task_env`, `defaults.inject_env`, and
on Kubernetes the `iris-task-env` Secret holding the object-store keys), Iris
itself (`IRIS_*` identity, the controller address, cache paths), and the job
(`EnvironmentSpec.env_vars`, which by default also copies `HF_TOKEN` and
`WANDB_API_KEY` from the submitting process, and in a child job the parent's
variables). Under `SANDBOX` the task receives its explicit `env_vars` and Iris
runtime variables needed to start the task, but no controller address or task
token. Kubernetes service-link environment injection is disabled for these pods.

The profile still supports `ExecInContainer`, which reaches the task through
the worker or `kubectl exec`. The job cannot carry a workspace bundle, and
the client rejects `extras`, `pip_packages` and `sync_packages`; setup scripts
run verbatim and default to none.

## Child job environments

Non-sandbox children normally inherit the parent's submitted variables from
`IRIS_JOB_ENV`. The parent environment is captured when its task starts, so
removing a secret from `os.environ` later does not remove it from that snapshot.
To submit a child with only the job variables you specify, use
`EnvironmentInheritance.EXPLICIT`:

```python
from iris.cluster.types import EnvironmentInheritance, EnvironmentSpec

client.submit(
    entrypoint,
    "child",
    resources,
    environment=EnvironmentSpec(
        env_vars={"TASK_CONFIG": "value"},
        inheritance=EnvironmentInheritance.EXPLICIT,
    ),
)
```

This also skips the usual variables copied from the submitting process,
including `HF_TOKEN`, `WANDB_API_KEY`, and provenance. The child still inherits
the parent's setup scripts and placement constraints. Cluster `task_env`,
`inject_env`, and Iris runtime variables still reach non-sandbox tasks; use
`CONTAINER_PROFILE_SANDBOX` when the task must be isolated from those inputs.

## Egress policy

Every job has an egress policy, set with `LaunchJobRequest.egress_policy`
(`IrisClient.submit(egress_policy=...)`, `iris job run --egress-policy`). The
profile decides what the task receives from the cluster; the egress policy
decides what its network reaches.

| Policy | Reaches | Default for |
|---|---|---|
| `EGRESS_POLICY_CLUSTER` | the cluster network: the controller, workers, other tasks and, where configured, the node's host network | every profile but `SANDBOX`, which rejects it |
| `EGRESS_POLICY_INTERNET` | public IPv4 addresses and DNS | `SANDBOX` |
| `EGRESS_POLICY_NONE` | DNS on Kubernetes; loopback only on Docker workers | |

`INTERNET` and `NONE` reach neither the controller, finelog, worker RPC ports,
other tasks nor the cloud metadata server. A task under either cannot launch
child jobs, register endpoints, or fetch anything from the controller,
whatever its profile.

### Docker workers

`CLUSTER` runs the container with `--network host`, and `NONE` with
`--network none`, which leaves only loopback; setup scripts that download
packages fail. Exec and file transfer go through `docker exec` and work under
both.

`INTERNET` runs the container on the `iris-egress` bridge network. A bridge
container reaches the VPC and the GCE metadata server (`169.254.169.254`)
through the VM's routes, and the worker container has no `NET_ADMIN`, so the
filter lives on the host. Worker bootstrap creates `iris-egress` on the bridge
interface `iris-egress0` with inter-container traffic disabled, and adds host
`iptables` rules: an `IRIS-EGRESS` chain, jumped to from `DOCKER-USER` for
traffic from `iris-egress0`, that drops `10.0.0.0/8`, `172.16.0.0/12`,
`192.168.0.0/16`, `100.64.0.0/10` and `169.254.0.0/16`, and an `INPUT` rule
that drops everything from `iris-egress0` to the host itself. It then writes
`/run/iris/egress/resolv.conf`, naming the public resolvers `8.8.8.8` and
`1.1.1.1`, which the container mounts as its `/etc/resolv.conf`: Docker's
embedded resolver forwards to the metadata server and relies on NAT rules
gVisor does not apply.

The worker runs an `INTERNET` task only while that file exists, and fails the
task otherwise. Bootstrap writes it last and `/run` is the host's tmpfs, so a
failed install or a reboot that has not yet rerun bootstrap leaves the worker
refusing `INTERNET` tasks instead of running them unfiltered. The scheduler
does not know which workers have the filter; on a fleet where bootstrap
failed, such a task fails and retries.

### Kubernetes

`CLUSTER` keeps the pod's configured host networking and sidecars. Under
`INTERNET` or `NONE` the pod drops host networking and carries the
`iris.egress` label with value `internet` or `none`. `iris cluster start`
creates one NetworkPolicy per value, `iris-egress-internet` and
`iris-egress-none`, in the Iris namespace. Both deny all ingress and allow
egress to DNS in `kube-system`. The internet policy also allows `0.0.0.0/0`
except the same five ranges and, when it lies outside those,
`kubernetes_provider.service_cidr`. `kubectl exec` goes through the kubelet
and is unaffected. The policies have effect only on a cluster whose network
plugin enforces NetworkPolicy. Workdir files too large for the pod's
ConfigMap are fetched from the controller by an init container and fail
under either policy.

Such a pod also has no log-shipping or output-upload sidecar. Containers in a
pod share its network, so a sidecar's route is also the task's: finelog serves
every job's logs to private-network peers without a token, and the uploader
needs the object-store keys and a route to the object store. The job's task
logs are therefore not in finelog, and `/iris/outputs` is not archived. Docker
workers ship logs and upload outputs from the worker process, outside the
container, so both keep working there.

## Task tokens

A controller with auth enabled gives every task outside `SANDBOX` a token in
`IRIS_TASK_TOKEN`. The token names the job's owner with the role `task`, and
the task's own Iris client (`iris_ctx()`, `IrisClient.in_cluster`) presents it
on every controller RPC. Owner-gated calls (child jobs, endpoint registration,
`ExecInContainer`) then behave as for the owner. The role carries no admin
authority: a task may give a child job the elevated container profile or the
`PRODUCTION`/`SYSTEM` priority band only when the child's parent already
holds it, and the token's `job_id` claim names that parent, so a task of one job
cannot borrow another job's privileges. Each dispatch mints a fresh token,
valid for 30 days and not revocable. A running container's token is never
refreshed: an attempt that outlives it loses controller access, the same
limit the worker token has. Under null auth no token is minted.

## Cluster network trust

The controller still trusts its network. Every production cluster lists the
private address ranges in `auth.trusted_cidrs`, and the controller
authenticates a caller from those ranges that presents no token as the
anonymous admin (`CidrAuthenticator` in `rigging.server_auth`). Until those
ranges are narrowed, a task with `CLUSTER` egress that drops its token can
still submit jobs with any container profile and, since the anonymous admin
passes the owner check on `ExecInContainer`, run commands in any task on the
cluster. This is why `SANDBOX` rejects `CLUSTER`.

## See also

- [`priority-bands.md`](priority-bands.md) — the parallel admin-gated job knob
- [`auth-loopback-transition.md`](auth-loopback-transition.md) — how loopback
  callers resolve to the admin role
