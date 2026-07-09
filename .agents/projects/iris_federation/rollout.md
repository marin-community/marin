# Iris multi-cluster rollout: federating `marin` (GCP) → `rno2a` + `us-east-02a` (CoreWeave)

Companion to `design.md` (Model D / peers) and `cluster_native_model.md`. This doc
covers the *remaining coding work and rollout procedure* to bring the built-but-inert
federation substrate into production: `marin` (GCP, IAP) federates whole jobs to the two
CoreWeave clusters, gated by authenticated user, over an IP-restricted + JWT channel.

Reviewed by codex (2026-07-08); its corrections are folded in. Source paths are under
`lib/iris/src/iris/cluster/` unless noted.

---

## 0. What's already built vs. what's left

The federation substrate is **implemented and merged** (PRs #6814→#6921). This rollout
adds the *auth gating, GPU meta-scheduling, cross-controller trust, CW networking/config,
and `submitting_user` identity* around it.

**Works end-to-end today** (`test_federation_handoff.py`, `_fold_exclusion.py`, `_exec_proxy.py`):
peer registry + authenticated per-peer connection (`federation/peer.py`); submit-time router
(explicit `cluster=<peer>` pin → prefer-local → live attribute match → whole-job handoff,
`federation/router.py`); capability heartbeat (`ListBackends`, 30s); handoff via `LaunchJob`
carrying `FederationHandoff{requester_id, owner_principal}`; delta-sync mirror-back
(`FederationSync`, 3s pull); routed cancel; proxied exec/profile; log relay to a shared
finelog; structural fold-exclusion via the `jobs.cluster`/`tasks.cluster` coordinate. EdDSA
JWT auth with a JWKS endpoint, IAP-email identity, CIDR/loopback trust, config-derived
`RolePolicy`, and an existing cross-controller **delegation-token** pattern for the finelog
relay (`create_delegation_token`, `auth.py:200`). CW object storage (LOTA) — no change.

**Not built yet (this rollout):**
1. Per-cluster user authorization — `PeerConfig` has no allow policy (`config.py:660`);
   `AllowPolicy.users` is per-*backend* only (`config.py:626,648`); `user_admitted()` is
   exact/`*` only (`backend.py:289`).
2. `submitting_user` persistence — the verified principal (`identity.user_id`) is computed
   in `launch_job` then discarded; only the friendly owner (`jobs.user_id`, plain string, no
   FK) is stored. **No `users` table** (dropped #6948, migrations `0039`/`0040`).
3. Cross-controller RPC trust — `JwksVerifier` trusts only its own issuer (`auth.py:95,339,369`);
   `_peer_credentials` presents an **empty bearer** to a cidr/null-auth peer (`peer.py:96`,
   `rigging/credentials.py:131`).
4. GPU meta-scheduling depth — attribute match only, no capacity/GPU-count (`router.py:83`);
   first-match, no balancing; no submit-side target flag (`cli/job.py:828`); CW advertises **no**
   attributes under the implicit single-backend form (`config.py:1130`).
5. CW networking — both CW controllers are `ClusterIP:10000` (rno2a) / `/proxy`-only ingress
   (us-east-02a); no GCP→CW RPC path; ephemeral signing keys.
6. `peers:` config — none declared; federation is inert.

---

## 1. Two design decisions (need the user's call)

**(a) `submitting_user` is a plain string, not an FK to a users table.** The brief said
"pointing at the users table," but #6948 (`738e6240a7`) deliberately removed `users`/`api_keys`
and rebuilt auth on stateless JWTs + a config-derived in-memory `RolePolicy`; migration `0040`
even stripped the `jobs.user_id → users` FK. **Recommendation:** `jobs.submitting_user` is a
plain string (an email, or `local_admin`); the per-cluster allowlist lives in **config**
(consistent with "config is the sole source of truth for identity"). Resurrecting `users`
reverses a recent deliberate decision.

**(b) The CW-side gate must key on a *signed* `submitting_user` claim (codex).** The peer
today trusts the parent because the parent connects as an admin machine principal and trusts
the `owner_principal` string it asserts. That is **not** independent of a buggy/compromised
root. For the CW re-check to be a real boundary, the federation JWT the root presents must
**carry the `submitting_user` claim**, and the CW controller must verify
`request.federation.submitting_user == token.submitting_user` before applying its own allowlist.
This is a lightweight, per-handoff signed assertion — *not* the full per-user cross-cluster
token model (deferred). **Recommendation: adopt this "signed-submitter Model 1.5."** The
channel is secured by (IP allowlist) **AND** (this federation JWT) — satisfying "both limit
IPs and require the federated JWT."

The rest of the plan assumes **(a) string column** and **(b) signed-submitter Model 1.5**.

---

## 2. Coding workstreams

### WS-1 — `submitting_user` identity: capture, inherit, persist, propagate
*`controller/schema.py`, `controller/migrations/0041_*`, `controller/service.py:1219-1257`,
`controller/ops/job.py:207`, `controller/writes.py:172,430`, `federation/store.py`,
`rpc/controller.proto` (`FederationHandoff`), `controller/auth.py`.*

1. Migration `0041_job_submitting_user`: `jobs.submitting_user VARCHAR NOT NULL DEFAULT ''`
   (no FK). Also add it to `federated_jobs` (RECEIVED rows on the peer, SENT rows on the parent)
   so the peer can gate and both sides can render it.
2. Resolve the principal at submit (`launch_job`, after `identity = get_verified_identity()`),
   `submitting_user_from_identity(identity, request)`: IAP → `identity.user_id` (email);
   CIDR/loopback (`ANONYMOUS_ADMIN.user_id == "anonymous"`) → **`local_admin`**; received
   handoff → `request.federation.submitting_user`; null-auth → `local_admin`.
3. **Inheritance (codex):** a local **child** job must inherit the parent's `submitting_user`,
   not re-resolve to the acting caller. Thread it through the child-submit path so a federated
   subtree keeps the root's submitter.
4. Persist via `insert_job_and_config` → `writes.insert_job(submitting_user=…)`; leave
   `jobs.user_id` (friendly owner) untouched.
5. Wire (WS-4): add `string submitting_user = 3;` to `FederationHandoff`; the root sets it from
   its resolved principal; the CW controller verifies it against the token claim (WS-2/WS-4),
   persists it on the RECEIVED row, and uses it as the gate key. `owner_principal` stays the
   friendly-owner attribution it is today.
6. **Sync-back (codex):** ensure `submitting_user` is stored on the parent SENT row at handoff
   (so it is never reconstructed from peer deltas), and surface it on `JobStatus` (proto +
   dashboard). Regenerate protos; update the Vue/TS dashboard.

### WS-2 — Per-cluster user authorization (the allowlist)
*`config.py` (`PeerConfig`, `AllowPolicy`, `AuthConfig`), `controller/backend.py:289`,
`federation/router.py`, `federation/manager.py`, `controller/service.py:1237` (handoff admission).*

1. Domain matching in `user_admitted`: extend exact/`*` to also match a domain entry
   `"*@openathena.ai"` (suffix match on the email's domain). One helper, reused by backend and
   peer policy. (Confirm pattern is on the **email**: `user@openathena.ai`.)
2. Add `allow_policy: AllowPolicy` to `PeerConfig` (mirror `BackendConfig.allow_policy`; default
   `["*"]`).
3. Enforce at the **root**, keyed on `submitting_user`: thread it into `RoutingRequest`; a
   `cluster=<peer>` pin whose policy rejects the submitter → `PERMISSION_DENIED`; auto-match
   skips non-admitting peers. (Gate on `submitting_user`, **not** `owner_principal` — on `marin`,
   `unprovisioned_role: admin` makes every IAP user admin, so the friendly owner is attacker-
   chosen; only the authenticated principal is trustworthy — codex confirmed.)
4. Enforce at the **CW peer** (independent boundary): on `request.HasField("federation")`, the
   federation token proves the requester (its verified issuer == `federation.requester_id`), and the
   handoff's asserted `submitting_user` (proto) is checked against `auth.allowed_submitters:
   ["*@openathena.ai"]`, with `local_admin` rejected outright. **Implemented (WS-4):** we trust a
   verified peer's asserted submitter rather than signing it into the token — the token is a static
   per-connection credential proving the requester only; see the trust-boundary note in the goal
   artifact. Reject otherwise.

### WS-3 — GPU meta-scheduler for federated routing
*`cli/job.py:828` (`job run` opts) / `build_job_constraints`, `federation/router.py`,
`federation/peer.py`, `federation/manager.py:279`, CW backend config.*

**Phase 3a (v1):**
1. Submit-side target flag: `job run` has no target-cluster option, and the top-level `--cluster`
   selects the *connection* (`cli/main.py:50`) — a collision. Add a distinct flag (e.g.
   `--target-cluster <peer>`) that appends the `cluster EQ <peer>` constraint. Document both meanings.
2. **CW must advertise device attributes (codex — important).** Under the implicit single-backend
   form, the synthesized backend has **no** `attributes`, so the heartbeat advertises nothing and a
   `device-variant=h100` job won't match. Convert each CW config to the **explicit `backends:`** form
   with `attributes: {device-type: gpu, device-variant: h100, region: …}` (or add scale-group→attribute
   derivation code). Validate against live `ListBackends` after restart.

**Phase 3b (improvement — deferred, land after 3a proves out):**
3. Capacity-aware selection: `BackendSummary` carries `worker_count`/`pending`/`running`, but
   `capacity_health` is **empty for K8s/CW** (only filled for the worker variant, `service.py:3030`) —
   score on worker/pending/running (and add a CW capacity signal if needed), replacing first-match.
4. GPU-count gating: `gpu-count` is `routing=False`; advertise available/per-VM count and gate on it.
5. **Retarget-on-`RESOURCE_EXHAUSTED` is NOT a small tweak (codex).** Once a SENT handle is
   persisted for a peer, rerouting must update `jobs.cluster`, `federated_jobs.peer_id`, sync
   cursor + cancel state, and avoid duplicate peer submissions. Design separately; do not ship in v1.

### WS-4 — Cross-controller trust (federation JWT + public-key distribution)
*`config.py` (`AuthConfig`), `controller/auth.py:95,200,339-373`, `federation/peer.py:96`,
`iris/rpc/auth.py:85`, `cli/cluster.py` (`init-keys`), `controller/service.py:3141` (`FederationSync`).*

1. Persistent signing keys on **both** CW controllers (`iris cluster init-keys`) — required
   anyway because CW relays logs to the shared finelog (`require_persistent_signing_key`).
2. Distribute the GCP root's **public key** to each CW controller: new `AuthConfig` surface
   `federation_peers: {marin: {public_key: "<PEM>", ...}}`, wired into a **separate federation
   verifier** (see 4). PEM is copied statically from `init-keys` output (CW can't dial GCP in the
   pull model). This is the "public key distribution" task.
3. Mint + present the federation JWT GCP→CW: `_peer_credentials` can't mint today (codex) — thread
   the local `JwtTokenManager` into peer-connection construction. Mint a short-TTL token with
   `aud="federation"`, `role` restricted, an **issuer bound to the configured peer id**, a
   `requester_id` claim, and the `submitting_user` claim (WS-2). Attach on `Authorization`.
4. **Separate federation verifier — do NOT add `aud="federation"` to the control-plane verifier
   (codex).** `authorize_method()` only blocks endpoint-scoped identities (`rpc/auth.py:85`), so a
   federation bearer accepted by the general verifier would become a full RPC identity. Build a
   distinct verifier that (a) trusts the configured peer public keys, (b) accepts only
   `aud="federation"`, and (c) is **method-scoped** to the federation RPC subset
   (`LaunchJob`-with-federation, `TerminateJob`, `FederationSync`, `ListBackends`, `ProfileTask`,
   `ExecInContainer`).
5. **`FederationSync` requester binding (codex).** Today any authenticated caller can pass any
   `requester_id` (`service.py:3141`). Bind the request's `requester_id` to the token's peer identity
   and reject a mismatch — for `FederationSync`, the handoff guard, and routed cancel.

### WS-5 — CoreWeave networking (IP allowlist + JWT ingress)
*CW k8s manifests / Traefik (`scripts/install_traefik_proxy.py` analog), `docs/coreweave.md`, marin VM.*

1. Expose **only the federation RPC subset** (WS-4.4), on a new dedicated ingress distinct from the
   `/proxy`-only route — **not** the whole controller RPC surface (codex). TLS via cert-manager;
   hostnames `iris-fed-rno2a.oa.dev`, `iris-fed-us-east-02a.oa.dev`.
2. IP-restrict via a Traefik `ipAllowList` (source-range) to the marin controller's egress IP.
3. Give the marin GCE controller a **stable static egress IP** (reserved external IP or Cloud NAT
   with a reserved IP) so CW allowlists one address.
4. **rno2a must become enforcing (codex — critical).** rno2a is null-auth today; a `signing_key` +
   JWT verifier alone stays **permissive** if there's no provider and no `trusted_cidrs`
   (`auth.py:436`, `rigging/server_auth.py:582`). The federation ingress path must run the enforcing
   federation verifier so an unauthenticated inbound request is rejected. Both IP-allowlist and JWT
   are required; neither alone suffices.

### WS-6 — Config wiring
*`lib/iris/config/{marin,cw-rno2a,cw-us-east-02a}.yaml`.*

- **`name:` is load-bearing (all three configs).** Startup feeds `cluster_config.name` into both the
  JWT issuer (`iss`) and the federation `cluster_id`/`requester_id` (`controller/main.py`). None of the
  current YAMLs declare `name`, so as-is they mint under the fallback issuer and send an empty requester
  id — the `federation_peers` trust and the requester binding would both fail. Set `name: marin`,
  `name: cw-rno2a`, `name: cw-us-east-02a` as part of this step; the peer id keys and `allow_policy`
  targets must match these exactly.
- `marin.yaml`: add `peers:` for `cw-rno2a` and `cw-us-east-02a` (`controller_address` = the federation
  ingress URL, `cluster` = the peer manifest name for credential resolution, `dashboard_url`,
  `allow_policy: {users: ["*@openathena.ai"]}`).
- CW configs: persistent `auth.signing_key`; trust anchor `auth.federation_peers: {marin: "<PEM>"}`;
  inbound allowlist `auth.allowed_submitters: ["*@openathena.ai"]`; the federation ingress.
  `cw-rno2a.yaml` is null-auth today and must gain the enforcing federation verifier config. Backend
  device attributes are auto-derived from `scale_groups.resources` (the backend-attr-derivation
  follow-up, weaver #411) — no manual `attributes:` duplication.

---

## 3. Recommended v1 slice (codex)

Land a reduced, safe first cut; defer 3b and retargeting:
`submitting_user` capture + inheritance + persistence (WS-1) · `PeerConfig.allow_policy` +
domain matching + root/peer enforcement (WS-2) · the separate, method-scoped federation verifier
with issuer/requester/submitter binding + `FederationSync` requester binding (WS-4.3–4.5) ·
rno2a→enforcing + IP-restricted federation ingress (WS-5) · explicit `--target-cluster` +
CW explicit backends advertising attributes (WS-3a) · config (WS-6). **Tests:** a rejected
non-OA handoff (root and CW-side), a `token.submitter ≠ handoff.submitter` rejection, an
unauthenticated/wrong-IP ingress rejection, and a **federated child-job** case (see §4).

---

## 4. Open questions to resolve before coding
1. `submitting_user` as a plain string (recommended) vs. resurrecting a `users` table.
2. Signed-submitter Model 1.5 (recommended) vs. plain parent-asserts (weaker) vs. full per-user
   cross-cluster tokens (deferred).
3. **Child-job federation (codex — must design first).** The peer writes changelog rows for a child
   under a received root, but the parent ignores any delta lacking a SENT handle (`writes.py:430`,
   `federation_store.py:149`). Decide how a federated subtree's child jobs are mirrored/attributed
   before relying on RE-driven children.
4. Does the CW allowlist include `local_admin` (a Marin operator SSH'd to the GCP controller), or is
   CW strictly `*@openathena.ai`? **RESOLVED:** `local_admin` is never a valid federation submitter —
   an enforcing parent refuses to federate a local_admin submission (service-side, with a clear
   "authenticate first" error) and the CW peer rejects it outright regardless of its allowlist. CW is
   strictly `*@openathena.ai`.
5. GPU routing for v1: explicit `--target-cluster` only (recommended) vs. capacity-aware auto-routing.
6. Allowlist pattern syntax: confirm domain-on-email (`*@openathena.ai`).

## 5. Rollout procedure (sequenced, reversible)
Federation is inert without `peers:`; all code lands dark. **Never restart a CW or the marin
controller without explicit user approval** (AGENTS.md).

- **P0 land code dark.** Merge WS-1..WS-4 + migration `0041`. Config strictness (pydantic forbids
  unknown keys, `config.py:156`) means **config must land after code**, never before. Full test pass.
- **P1 CW keys + trust anchor.** `init-keys` on both CW controllers; copy the GCP root public key into
  each CW `federation_peers`. (Config staged, not applied.)
- **P2 CW networking.** Stand up the IP-restricted, enforcing federation ingress (federation RPC
  subset only). Reserve + allowlist the marin static egress IP. Verify from the marin VM: `ListBackends`
  succeeds **with** the federation JWT, refused **without** it / from a non-allowlisted IP.
- **P3 CW config + restart (approval).** Set `name:`, and apply `auth.allowed_submitters`,
  `auth.signing_key`, `auth.federation_peers`, rno2a→enforcing (device attributes auto-derive from
  `scale_groups.resources` once weaver #411 lands — no manual `attributes:`). Restart CW controllers
  (builds from local tree — ensure merged code). Confirm CW advertises GPU attributes.
- **P4 enable on marin (approval).** Add `peers:` with `allow_policy` scoped to **one OA test user**.
  Restart marin. Confirm the Peers tab shows both CW clusters reachable.
- **P5 smoke.** As the OA test user: `--target-cluster rno2a` GPU job → handoff → runs on CW → status/
  logs mirror back → cancel → exec/profile proxy. Verify a **non-OA** user rejected at submit and at CW;
  a non-allowlisted IP refused. Then widen to `*@openathena.ai`.
- **P6 GPU auto-routing (WS-3b)** — optional, independently shippable, after explicit targeting proves out.

**Rollback:** removing `peers:` + restart stops new routing but **existing handed-off jobs remain
federated rows and can wedge if a peer disappears (codex)** — define a terminalization/cleanup path
(cancel-and-tombstone the SENT handles) as part of rollback tooling. Ingress can be torn down; CW config
keys are additive; migration `0041` is an additive column (harmless if left).

## 6. Risks / watch-items
- Every marin IAP user is admin (`unprovisioned_role: admin`), so the per-peer allowlist is the
  **only** thing restricting CW access — the root gate and the signed CW-side re-check are both load-bearing.
- Federation tokens are stateless and unrevoked: bound only by short TTL + IP allowlist + issuer/aud/
  requester binding + TLS. Keep TTL short.
- CW backend `attributes` are static config; mis-declaration silently mis-routes — validate against live
  `ListBackends` post-restart.
- Ephemeral→persistent key cutover on CW invalidates outstanding worker/endpoint/federation tokens and
  proxy share-links; do it in a restart window with retained-previous-public-keys understood by all controllers.
- Cross-cluster budget/spend is unenforced (federated tasks excluded from local budget; no cross-cluster
  admission) — CW multi-tenant overspend is unbounded. Known limitation / follow-up.
- Reverse-dial transport (peer behind NAT) is out of scope — the pull model requires the CW federation
  ingress to be reachable inbound.
