# Debugging log for the Iris Rust proxy auth handoff

Authenticate controller RPCs at the Rust public listener, then pass only the
verified identity to the private Python server for authorization.

## Initial status

On `cw-us-west-04a`, the controller Deployment became ready and loaded its
configured signing key and trusted CIDRs. Both an operator port-forward and an
in-pod request to `127.0.0.1:10000` failed controller RPCs with `Missing
authentication`.

Pulumi prerequisites, Kueue resources, the controller RBAC, and
`iris-controller-env` were present. The rollout was stopped and the restored
Deployment removed.

## Root cause

The native listener adds `X-Forwarded-For` before forwarding every request to
the private Uvicorn server. Rigging intentionally refuses loopback and CIDR
authentication when that header is present, so the Rust-to-Python hop discards
the direct caller's trusted-network identity.

The Python listener is loopback-only, so it can trust direct private ingress.
The public Rust listener must reject unauthenticated RPCs before forwarding and
pass authenticated callers' identities without asking Python to interpret the
original public headers again.

## Changes to make

- Add a boundary regression covering trusted direct and untrusted direct
  controller requests through the native listener.
- Authenticate JWT, IAP, trusted-network, and permissive-mode callers in Rust.
- Strip public credentials and caller-supplied internal headers before the
  private hop.
- Stamp the verified user, role, and audience. Python trusts private ingress and
  retains method and resource authorization.
- Build the native extension from the checked-out Rust source in local tests and
  the controller Docker image.

## Results

The regression failed before the fix: both the direct and explicitly forwarded
controller RPCs returned HTTP 401.

The native listener now owns public authentication. It removes `Authorization`,
cookies, IAP assertions, and spoofed Iris-internal headers before forwarding to
Uvicorn. A request that Rust authenticates carries a percent-encoded identity
stamp. Python does not re-read the public credential; it trusts its loopback
ingress, accepts the stamp when present, and applies Iris authorization. Rust
rejects missing authentication on controller RPC paths while still forwarding
public dashboard, login, health, bundle, and static routes.

The boundary regressions pass against a source-built `marin-iris-native`: a
trusted direct request and a valid bearer request return HTTP 200, a forwarded
request without credentials returns HTTP 401, and a caller-supplied identity
stamp cannot bypass Rust.

## CI rollout attempt

The guarded `cw-us-west-04a` restart began from no controller Deployment, no
controller service endpoints, and no task pods. While the restart command was
waiting for its initial controller tunnel, a new `iris-controller` Deployment
using image `ghcr.io/marin-community/iris-controller:2c2ae09` appeared. The
tunnel connected as that pod started, so the command treated it as an existing
controller and attempted a pre-deploy checkpoint. The connection closed before
the checkpoint RPC completed, and the command exited nonzero.

The unexpected Deployment was deleted immediately. The namespace returned to
the verified pre-rollout state: no controller Deployment, no controller pods,
and no service endpoints. No task was launched and no second rollout was
attempted.

## CI rollout confirmation

The retry started the empty CI controller explicitly from the current working
tree. Docker built `marin-iris-native` from `lib/iris/rust` for amd64 and arm64,
then published controller, worker, and task images with tree hash
`30c084f0d3`. The controller became healthy and reported that exact version.
Trusted port-forward RPCs succeeded; an unauthenticated `ListJobs` request with
an external `X-Forwarded-For` returned HTTP 401.

Job `/power/rust-proxy-roll-continuity-20260724` reached `RUNNING` before the
checkpointed controller restart. Its task used pod
`iris-power-rust-proxy-roll-continui-1d0e07d6-0-e955935ac8fe7977`, UID
`205bee95-3248-4c29-9644-cbd2829934da`, on node `g8fd930`.

The restart checkpoint contained the job. After the controller returned healthy
on `30c084f0d3`, the job remained `RUNNING`; the pod name, UID, node, start time,
attempt ID, and zero restart count were unchanged. `task exec` also succeeded
through the restarted controller, exercising its pod lookup against the new
pod-name format.

The controller's Kubernetes metrics poller separately logs a pre-existing parse
error for CPU values such as `503286n`. The same controller image was running
before and after the continuity restart, and the error did not affect scheduling
or task execution.

## Future work

- [ ] Expose native endpoint-registry and JWT-cache counters in the controller
  dashboard if operators need them alongside RPC timing statistics.
- [ ] Make controller discovery distinguish a healthy existing controller from
  a controller that appears mid-discovery, so a pre-deploy checkpoint cannot
  race startup.
