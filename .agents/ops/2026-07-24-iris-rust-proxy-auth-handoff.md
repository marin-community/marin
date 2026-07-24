---
date: 2026-07-24
system: iris
severity: degraded
resolution: fixed
pr: https://github.com/marin-community/marin/pull/7583
issue: none
---

# TL;DR

- The first native-proxy rollout on `cw-us-west-04a` returned `Missing
  authentication` for trusted loopback controller RPCs.
- Rust added `X-Forwarded-For` before forwarding to Python, so Rigging refused
  to classify the private hop as trusted ingress.
- Rust now makes public authentication decisions, strips public credentials and
  spoofed internal headers, and stamps the verified identity. Python trusts its
  loopback-only listener and performs Iris authorization.
- A guarded restart kept
  `/power/rust-proxy-roll-continuity-20260724` on the same task attempt, pod UID,
  node, start time, and zero restart count.

# Original problem report

Roll the CI Iris controller with the native Rust proxy enabled and prove that
the new pod naming remains compatible. A task had to remain running before and
after the controller restart. Any rollout error required rollback and a stop.

# Investigation path

1. The replacement controller loaded its configured signing key and trusted
   CIDRs, but controller RPCs through both port-forward and
   `127.0.0.1:10000` returned `Missing authentication`.
2. Pulumi prerequisites, Kueue resources, controller RBAC, and
   `iris-controller-env` were present. Missing cluster authorization was ruled
   out.
3. The Rust listener was found to add `X-Forwarded-For` before every private
   hop. Rigging intentionally rejects loopback/CIDR authentication when that
   header is present.
4. The authentication boundary moved to Rust. Rust verifies JWT, IAP,
   trusted-network, and permissive-mode requests; Python no longer interprets
   public credentials on its loopback listener.
5. Boundary tests built `marin-iris-native` from `lib/iris/rust`. Trusted and
   bearer-authenticated requests returned HTTP 200, untrusted requests returned
   HTTP 401, and a caller-supplied identity stamp did not bypass Rust.
6. The first guarded retry raced a newly appearing controller Deployment during
   discovery. The checkpoint RPC connection closed, so the Deployment was
   removed and the namespace returned to its pre-rollout state.
7. The next retry started the empty CI controller explicitly on image tree hash
   `30c084f0d3`. An unauthenticated controller RPC with external
   `X-Forwarded-For` returned HTTP 401.
8. The continuity task reached `RUNNING` before restart. After the checkpointed
   restart, its attempt ID, pod UID, node, start time, and zero restart count
   were unchanged. `task exec` succeeded through the new controller.

# User course corrections

- The initial handoff retained Python authentication decisions. The user
  clarified that Rust owns authentication and Python should trust every request
  arriving on its loopback-only listener.
- The user asked why Python needed proof that a request came from the local
  proxy. Confirming that Python binds only to loopback removed the redundant
  authentication layer.
- The user required a running task across the CI restart. This caught pod-name
  compatibility at the controller lookup boundary instead of treating
  readiness as sufficient validation.

# Root cause

The Rust listener forwarded a public-proxy header to the private listener.
Rigging correctly treated a loopback request carrying `X-Forwarded-For` as
potentially spoofed and did not grant its trusted-network identity. Python then
rejected an RPC that Rust had already accepted for forwarding.

# Fix

The native listener now authenticates public requests, removes
`Authorization`, cookies, IAP assertions, and caller-supplied Iris identity
headers, then adds a percent-encoded verified user, role, and audience stamp.
The loopback-only Python policy trusts unstamped private calls as its internal
admin identity and uses a present stamp for method and resource authorization.

Controller images build the native extension from the checked-out Rust source.
The CI rollout verified both public rejection behavior and task continuity.

# How OPS.md could have shortened this

Add a native-proxy boundary check to `lib/iris/OPS.md`: test one trusted RPC,
one unauthenticated RPC with an external forwarding header, and one
caller-supplied internal identity header before rolling a non-CI controller.
This distinguishes proxy handoff failures from cluster IAM or signing-key
failures.

# Artifacts

- Pull request: https://github.com/marin-community/marin/pull/7583
- CI continuity job: `/power/rust-proxy-roll-continuity-20260724`
- Follow-up RPC metrics work: Weaver issue #610
