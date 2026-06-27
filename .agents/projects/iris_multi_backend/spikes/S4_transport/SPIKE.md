# SPIKE S4 — transport + interactive path

Resolves the two open S4 questions in [`design.md`](../../design.md) ("On-demand interactive
transport") and pins the numbers behind [`spec.md`](../../spec.md) §1 / §1.1.

- **(a) Does "agent dials home" over Connect unary work cleanly for the `Poll()` wire?**
  **Yes.** A loopback prototype runs `RemoteAgentService.Poll` with the agent as the dialing Connect
  client and the root as the server — no inbound endpoint on the cluster, NAT/IAP-friendly — using the
  real iris stack (`connectrpc` ASGI server under uvicorn + `ConnectClientSync`). A static
  `system:controller` bearer token rides the dial-home wire and is verified by the same
  `rigging.server_auth.AuthInterceptor` the controller uses; an un-tokened Poll is rejected
  (`Missing or malformed Authorization header`). The §1.1 interactive piggyback works end-to-end:
  an `exec` issued on the root rides DOWN in a `PollResponse`, runs on the agent, and the result rides
  UP in the next `PollRequest`.

- **(b) Is the poll-piggyback interactive path acceptable, or do we need a held stream?**
  **The poll piggyback is acceptable for `exec` / `profile` / `process-status` if the agent does a
  "fast-follow" poll** (poll again immediately when a result is pending instead of waiting a full
  cadence). At a 1 s steady cadence that gives **~0.5 s mean / ~1 s p95** interactive latency. A held
  stream gives **~2 ms** (cadence-independent) but costs a permanent server-push connection per backend.
  Recommendation below: **default = Poll + fast-follow; opt-in held stream only for trusted in-VPC,
  latency-sensitive backends.**

---

## 1. The prototype

Throwaway loopback prototype, all new files under this directory. It uses the real iris RPC idioms:
`connectrpc` `ConnectASGIApplication` + `Endpoint.unary`/`Endpoint.server_stream` on the server (the
same primitives the buf-generated `*_connect.py` wire up), `ConnectClientSync.execute_unary` /
`execute_server_stream` on the client, served by `uvicorn`, with `rigging` bearer auth on the wire.

| File | What it is |
|---|---|
| `remote_agent.proto` | Trimmed-but-faithful cut of spec §1/§1.1: `RemoteAgentService.Poll` + `InteractiveCommand`/`CommandResult` piggyback, plus a held-stream `CommandStream`/`ReportResult` pair for the comparison. |
| `remote_agent_pb2.py` | `protoc --python_out` output (protobuf 6.33.6 runtime; protoc 29.3). Generated, committed so the spike runs without buf. |
| `remote_agent_connect.py` | Hand-written Connect binding mirroring iris's generated `*_connect.py`: `RemoteAgentServiceASGIApplication` + `RemoteAgentServiceClientSync`. Written by hand only to avoid pulling the buf remote-plugin toolchain into a throwaway spike. |
| `root_server.py` | Root = **server**. `RootState` holds desired-state + the interactive-command plumbing; the root-side `exec_*` (modeling `RemoteTaskBackend.exec_in_container`) is a **blocking** call that parks on a `threading.Event` until the matching `CommandResult` returns. `AuthInterceptor` + `StaticTokenVerifier` verify the `system:controller` token. Served on loopback by uvicorn. |
| `agent_client.py` | Agent = **dialing client**. `PollAgent` runs the §1.1 piggyback loop on a cadence (with optional fast-follow); `StreamAgent` runs the held-stream variant. A `BearerTokenInjector` attaches the token. The "stub" executor stands in for the worker's real `exec_in_container`. |
| `bench.py` | Harness: starts the server, dials it, issues N blocking `exec` commands per configuration with phase sampled uniformly across the poll cycle, reports the latency distribution. |

### How to run

```bash
cd .agents/projects/iris_multi_backend/spikes/S4_transport
# regenerate the pb2 if you edit the proto (optional; checked in):
protoc --python_out=. remote_agent.proto
uv run python bench.py        # ~5–6 min wall (it sleeps a lot to sample poll phases)
```

The harness asserts every returned `ExecResponse` is correct (exit 0, expected stdout), so a green run
also proves the round-trip is functionally correct, not just timed.

---

## 2. Measured interactive latency vs cadence

End-to-end "user feels it" latency: from the blocking `exec` call on the root to the result in hand.
N=30 per row, phase sampled uniformly across the cadence. Loopback (no IAP) — see §3 for the IAP offset.

```
loopback unary Poll RTT (floor)    n=50  mean=  0.4ms  p50=  0.4ms  p95=  0.5ms

== Poll piggyback (command DOWN one Poll, result UP the NEXT Poll) ==
cadence=0.5s  no fast-follow        mean= 774.5ms  p50= 761.0ms  p95= 997.3ms  min= 509.2ms  max= 999.7ms
cadence=1.0s  no fast-follow        mean=1545.5ms  p50=1518.0ms  p95=1989.2ms  min=1026.1ms  max=1994.5ms
cadence=2.0s  no fast-follow        mean=3087.8ms  p50=3029.3ms  p95=3975.2ms  min=2062.9ms  max=3989.7ms

== Poll piggyback + fast-follow (extra Poll fired to return the result) ==
cadence=0.5s  fast-follow           mean= 274.0ms  p50= 260.4ms  p95= 496.0ms  min=   9.3ms  max= 500.5ms
cadence=1.0s  fast-follow           mean= 545.2ms  p50= 516.8ms  p95= 986.3ms  min=  28.8ms  max= 997.1ms
cadence=2.0s  fast-follow           mean=1087.6ms  p50=1030.4ms  p95=1971.8ms  min=  61.8ms  max=1988.8ms

== Held stream (CommandStream push + unary ReportResult) ==
held stream (cadence-independent)   mean=   1.8ms  p50=   2.0ms  p95=   2.3ms  min=  0.7ms  max=  3.4ms
```

**The numbers match a clean closed-form model** (C = cadence), which lets us extrapolate to any cadence:

- **Poll piggyback, no fast-follow ≈ 1.5·C** (range `[1·C, 2·C]`). The command waits ~U(0,C) to be
  picked up on the next Poll (rides DOWN), then the result *must* wait one full cadence to ride UP on the
  *next* PollRequest — because the command arrived inside a *response*, the agent can't answer in the
  same request that delivered it. Mean = C/2 + C = 1.5·C; worst case 2·C (command lands just after a
  Poll).
- **Poll piggyback + fast-follow ≈ 0.5·C** (range `[~0, 1·C]`). The DOWN leg is unchanged (~C/2 mean),
  but the agent returns the result by firing an *extra* Poll immediately, so the UP leg collapses to one
  RTT. This is a ~3× win and removes the "result stuck for a whole cadence" tail. **It is a one-line
  client change** (poll again now if results are pending) and needs no protocol change.
- **Held stream ≈ 2·RTT + exec**, independent of cadence — single-digit ms on loopback.

The unary floor is **0.4 ms**, so on loopback the transport adds essentially nothing; the latency you
see is *entirely* the cadence-driven wait. That is the whole point: **Poll cadence sets interactive
latency** (spec §1.1), and fast-follow halves it.

---

## 3. IAP / cross-region offset (not measured live — see §5)

Loopback hides per-call network cost. Over the real IAP edge each unary call is a TLS + IAP-proxy
round-trip; budget **~30–100 ms RTT** cross-region. The effect is asymmetric and *favors the poll path*:

- **Poll piggyback:** the cadence term dominates (hundreds of ms to seconds); a per-call IAP RTT of tens
  of ms is in the noise. Conclusion unchanged.
- **Held stream:** latency becomes `2·IAP_RTT + exec` ≈ **60–200 ms** — still far below any
  cadence-driven number, but the held stream also needs a *long-lived* server-push connection through
  IAP, and IAP / Google L7 LBs cap idle stream lifetime (minutes), so a held stream over IAP needs
  keepalives and reconnect handling. In-VPC (no IAP edge) it is clean.

---

## 4. Recommendation: poll channel vs held stream

**Default for v1: Poll piggyback with fast-follow.** It is the design's "one wire" with no new RPC, no
inbound endpoint, a single in-flight Poll per backend (deltas stay ordered, spec §1), and it is
NAT/IAP-friendly by construction. With fast-follow, a **1 s steady cadence → ~0.5 s mean / ~1 s p95**
interactive latency — fine for `exec` (run-a-command), `profile`, and `process-status`, which are
human-interactive, not keystroke-interactive. Adopt **fast-follow** as part of v1 (it is free and
removes the full-cadence tail) and pick the steady cadence from S3's lease math, not from interactivity.

**Add an opt-in held stream only for trusted, in-VPC, latency-sensitive backends.** The held-stream
variant (`CommandStream` server-push + unary `ReportResult`) buys ~2 ms + one IAP RTT, cadence-independent,
at the cost of a permanent push connection per backend and reconnect/keepalive plumbing (and it does not
survive an IAP idle-timeout cleanly). Worth it only for high call volume or sub-100 ms interactivity on a
backend that can hold the connection. Gate it on a per-backend `transport.interactive: held_stream` flag.

**Preferred middle ground if poll latency ever bites: adaptive cadence / long-poll, not a second RPC.**
- *Adaptive cadence*: idle backends poll slowly (2–5 s, saves QPS); when an interactive command is
  outstanding or recently seen, the agent drops to a fast cadence (200–500 ms) or fast-follows. Gets
  near-held latency on demand without a permanent stream.
- *Long-poll*: the server holds the one `Poll` open until a command is ready (or a max-hold ~25 s, then
  returns empty). This is the simplest "held" upgrade that still rides the single Poll RPC, stays
  IAP-friendly, and needs no new service — recommended over a streaming RPC if fast-follow ever proves
  insufficient. (Not prototyped here; mechanically it is the server awaiting a per-backend
  `asyncio.Event` inside the existing `poll` handler.)

**Out of scope for both:** a *PTY/interactive TTY* `exec` (keystroke streaming) needs real bidi and is
not served by request/response exec — call it out of scope, as spec already does for bulk logs (those
stay in each backend's finelog, proxied per `backend_id`, never tunneled over Poll).

---

## 5. IAP / service-account bootstrap (DESIGN ONLY — nothing live was touched)

The agent reaches `iris.oa.dev` behind IAP with **two stacked credentials**, both already implemented in
the tree — no new auth primitive is needed, only a new *mint* and a new *role binding*:

1. **IAP edge token (`Proxy-Authorization`).** An OIDC ID token minted from the agent VM/pod's *ambient
   service account* via `rigging.auth.IapServiceAccountTokenProvider`
   ([`rigging/auth.py:123`](../../../../../lib/rigging/src/rigging/auth.py)), which calls
   `google.oauth2.id_token.fetch_id_token(audience=<IAP OAuth client id>)`. This is exactly the
   service-account path the CLI/clients use today (`iris.cli.connect` notes the IAP OIDC ID token rides
   `Proxy-Authorization`). The agent SA is added to the IAP-protected backend's access policy at
   onboarding. **No human OAuth, no inbound endpoint.**

2. **Application bearer (`Authorization: Bearer <jwt>`).** A backend-scoped `system:controller` JWT,
   attached by `rigging.auth.BearerTokenInjector` (the spike proves this attach/verify path on loopback).
   It is minted parallel to the worker token:
   - Today `_create_worker_jwt`
     ([`controller/auth.py:462`](../../../../../lib/iris/src/iris/cluster/controller/auth.py)) does
     `jwt_mgr.create_token(WORKER_USER, "worker", key_id)` and records an `api_keys` row for
     audit/revocation; the token is handed to workers via `composer.py:202`
     (`auth_token=auth.worker_token`).
   - Add a sibling `_create_controller_jwt` minted in `create_controller_auth`
     ([`controller/auth.py:322`](../../../../../lib/iris/src/iris/cluster/controller/auth.py)) that calls
     `create_token(CONTROLLER_USER, "controller", key_id)` with a `backend_id` claim (the JWT payload at
     `auth.py:203` is a plain dict — add `"backend_id"` and `"allowed_rpcs"`). Server-side,
     `RemoteAgentService` binds every call's subject to its `backend_id` and is **default-deny outside
     `RemoteAgentService`**; loopback-trust is disabled on this path (spec §6). Revocation reuses the
     existing `api_keys` revocation set.

**Bootstrap flow** (`iris backend add <id>` → `iris agent serve --backend <id> --controller iris.oa.dev`):
the root mints the `system:controller` JWT, records the `api_keys` row, and hands the JWT to the agent at
serve time (env var, same delivery shape as `worker_token`); the agent's SA is granted IAP access. On
each `Poll` the agent sends both headers; IAP verifies the SA ID token at the edge, and
`AuthInterceptor`/`JwtTokenManager.verify` ([`auth.py:212`](../../../../../lib/iris/src/iris/cluster/controller/auth.py))
verifies the bearer and resolves the `system:controller` identity + `backend_id`. The JWT is HS256-signed
with the controller's persistent `controller_secrets` key, so verification never hits the DB on the hot
path — same property the worker token has today.

> Spike status: **CLOSED.** Dial-home Connect works cleanly; poll-piggyback+fast-follow is the v1
> interactive transport; held stream is an opt-in for in-VPC latency-sensitive backends; long-poll is the
> recommended escalation before a second RPC. IAP/SA bootstrap reuses existing rigging + auth primitives.
