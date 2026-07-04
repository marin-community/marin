# Research — unified server auth & secret configuration

Current-state audit for the auth-audit umbrella (#6942). All refs are
`path:line` against `main` at `5a6f64cbeef5e1962ed367deb3aaf72956ddb4d1`.
This is the working digest behind `design.md`; it also serves as the "map the
current auth architecture" deliverable the umbrella asks for.

## 1. The seam already exists — auth is centralized in `lib/rigging`

Server auth is **not scattered**. `lib/rigging` owns the mechanism; iris and
finelog layer service policy on top. The four "sloppy" gaps are all *around*
this seam, not a missing seam.

### Client side (`rigging/auth.py`, `rigging/credentials.py`, `rigging/connect.py`)

- Token **providers** mint/attach bearer material: `StaticTokenProvider`,
  `GcpAccessTokenProvider`, `IapServiceAccountTokenProvider` (SA/CI, via
  `fetch_id_token`), `IapRefreshTokenProvider` (human desktop-OAuth re-mint).
  `rigging/auth.py:76-184`. `BearerTokenInjector(provider, header)`
  (`auth.py:290-323`) is a Connect *metadata* interceptor, so the header rides
  every RPC shape (unary + streaming).
- `credentials_for(cluster, auth)` (`credentials.py:154-170`) is *the* "how a
  client finds its tokens" answer: app bearer (`Authorization`) resolved
  env-override → login file → ambient provider; IAP edge bearer
  (`Proxy-Authorization`) resolved cached-human-login → ambient SA. Returns a
  `ClientCredentials` bundling both so a caller can't attach one and forget the
  other (`credentials.py:51-77`).
- `connect(transport, factory, *, auth)` (`connect.py:270-316`) composes a
  transport (`DirectTransport` / `TunnelTransport`) × auth (`NoAuth` / `JwtAuth`
  / `IapAuth` / `ChainedAuth`). Transport-URL schemes:
  `http`/`https`/`iap+https`/`ssh+gcp`/`k8s` (`connect.py:146-213`). The
  controller hop is just the opaque URL path `/proxy/<name>`
  (`proxy_path`, `connect.py:132-134`).
- The narrow cross-lib config slice every client agrees on:
  `ClusterAuth`/`IapAuth`/`AuthProvider` in `cluster_manifest.py:40-118`.

### Server side (`rigging/server_auth.py`)

- **Verifiers** (`TokenVerifier` protocol): `StaticTokenVerifier`,
  `GcpAccessTokenVerifier`, `IapIdTokenVerifier`, `IapAssertionVerifier`
  (`server_auth.py:136-310`). The IAP/Google verifiers are inherently Python
  (`google-auth`, `requests`, `google.oauth2.id_token`).
- **Authenticator chain** — each authenticator returns
  `AUTHENTICATED` / `ABSENT` / `REJECTED` (`AuthDecision`,
  `server_auth.py:374-379`); `resolve_auth` walks in order, first
  AUTHENTICATED wins, first REJECTED stops (a bad credential is never
  downgraded to ambient trust), all-ABSENT raises (`server_auth.py:531-550`).
  Authenticators: `JwtAuthenticator`, `BestEffortJwtAuthenticator`,
  `IapAssertionAuthenticator`, `CidrAuthenticator`, `LoopbackAuthenticator`,
  `AnonymousAuthenticator` (`server_auth.py:412-528`).
- **`RequestAuthPolicy`** (`server_auth.py:553-627`) — `enforcing()` builds
  `[Jwt?, IapAssertion?, Cidr?, Loopback, Anonymous?]`; `permissive()` builds
  `[BestEffortJwt?, Anonymous]`. The chain fully decides every request, so the
  enforcement points mount unconditionally and never branch on an "auth on"
  flag.
- **Enforcement points**: `PolicyAuthInterceptor` (Connect RPC,
  `server_auth.py:630-679`) and `RouteAuthMiddleware` (HTTP routes,
  `server_auth.py:728-785`). Routes are annotated `@public`/`@requires_auth`;
  **unannotated routes are denied** (`_RouteAuth.DENY`, `server_auth.py:696`).
- **Network-location trust model** (`_direct_peer_ip`, `server_auth.py:313-345`):
  trust the transport *socket peer* only — a request carrying `X-Forwarded-For`
  or a port-0 (uvicorn `forwarded_allow_ips="*"`) peer is refused CIDR/loopback
  trust. This is the load-bearing anti-spoofing invariant.

## 2. Iris service policy (on top of the rigging seam)

- **JWT minting** — `JwtTokenManager` (`iris/cluster/controller/auth.py`).
  HS256; signing key **minted on the controller** and persisted in the
  `controller_secrets` DB table (`auth.py:157-181`, schema
  `controller/schema.py:669-675`), `secrets.token_hex(32)`, INSERT-OR-IGNORE.
  Claims `{sub, role, jti, iat, exp}`, `DEFAULT_JWT_TTL_SECONDS = 30 days`
  (`auth.py:39`). Verify is pure crypto + in-memory `_revoked_jtis` set — **no
  DB hit on the hot path** (`auth.py:250-279`).
- **Endpoint-scoped tokens** — `create_endpoint_token`
  (`auth.py:226-248`): `aud = endpoint wire name`, `scope = "proxy"`,
  `role = "endpoint"` (zero RPC authority). `DEFAULT_ENDPOINT_TOKEN_TTL = 3600`,
  **`MAX_ENDPOINT_TOKEN_TTL_SECONDS = 86400` (24h)** at `auth.py:53`, clamped in
  the `MintEndpointToken` RPC at `service.py:2656`. **This answers #6937's open
  question about the max token TTL for long-running datagen callers.**
- **RBAC** — `iris/rpc/auth.py`: persisted roles `{admin, user, worker}`
  (`schema.py:48`), transient `dashboard` (read-only IAP tier) + `endpoint`
  (proxy-only). `POLICY` maps `AuthzAction → allowed roles` (`rpc/auth.py:32-47`);
  `authorize_method` is the per-RPC gate wired into `PolicyAuthInterceptor`
  (`rpc/auth.py:89-112`) — it denies any `audience != None` (endpoint-scoped)
  identity from every control RPC and restricts `dashboard` to a read-only
  allowlist.
- **Policy assembly** — `request_auth_policy(config.auth)`
  (`auth.py:340-353`) picks `enforcing` vs `permissive` off
  `ControllerAuth.provider`; wired once in `Controller.__init__`
  (`controller.py:462-468`) and threaded through every surface in
  `dashboard.py:347-379,504`.
- **IAP edge / GCLB** — `iris/scripts/iap_gclb.py`, `iris/docs/iap-gclb.md`.
  Per-backend-service IAP; shared frontend; `audiences` (login ID token) vs
  `programmatic_audiences` (SA edge token) vs `signed_header_audience` (the
  `X-Goog-IAP-JWT-Assertion` backend audience). `public-proxy` stands up a
  second **IAP-free** backend routing `/proxy/*` while the dashboard/RPC surface
  stays IAP-gated; the controller's `_authorize_proxy` is the sole gate.
- **User store** (relevant to #6592) — `users_table`
  (`schema.py:245-253`). **Crux:** the `login` RPC auto-provisions *any*
  IAP-admitted identity at the write-capable default role `"user"` via
  `ensure_user` (`service.py:2584-2620`). There is **no role-change RPC**;
  `set_user_role` is only called at startup for `admin_users` and null-auth
  anonymous (`auth.py:404-405,433-436`). `IapAuthConfig.unprovisioned_role`
  (default `"dashboard"`, read-only) applies only to tokenless assertion
  callers who never ran `iris login`.

## 3. Finelog server policy (the second, drifting implementation)

- `lib/finelog/rust/src/server/auth.rs` re-implements the same conceptual model
  in Rust as a **declarative layer stack**: `enum Verdict { Allow, Fallthrough,
  Reject }` (`auth.rs:76-84`) mirrors rigging's AUTHENTICATED/ABSENT/REJECTED.
- **Two layer kinds only**: `cidr` and `jwt`, internally tagged on `type`,
  snake_case (`AuthLayerConfig`, `auth.rs:245-259`). Loaded from the
  `FINELOG_AUTH_POLICY` env var / `--auth-policy` (`main.rs:63-64`,
  `AuthPolicy::parse`, `auth.rs:416-430`).
- **Default-deny**: the stack walk falls off the end to `false`
  (`AuthPolicy::admits`, `auth.rs:450-459`); an **empty `[]` list is rejected**
  as a total lockout (`auth.rs:419-423`); omitting the var yields
  `allow_localhost` (`127.0.0.0/8` + `::1/128`, `auth.rs:401-408`) — loopback
  only, never open.
- **CIDR-first ordering** is load-bearing (a trusted on-VPC controller must be
  admitted before the jwt layer rejects a `worker_token` it cannot verify;
  `auth.rs:38-43`). rigging orders jwt-first — a documented divergence (#6861).
- Same anti-spoofing model as rigging: matches the socket peer only, no
  `X-Forwarded-For` path (`auth.rs:35-36,493-497`). JWT is HS256, keys HMAC'd as
  **raw ASCII bytes** (PyJWT parity, verified by cross-language test
  `jwt_verifier_accepts_pyjwt_minted_token`, `auth.rs:681-696`),
  `MIN_SECRET_BYTES = 16`, `EXP_LEEWAY_SECONDS = 60`.
- **The schema is defined twice** — Rust structs vs Python dataclasses
  (`finelog/deploy/config.py:80-122`: `CidrAuthLayer`, `JwtAuthLayer`,
  `JwtKeyEntry`, `auth_policy_json`). The only contract between them is a pinned
  JSON test (`finelog/tests/test_config.py:136-163`). No shared enforcement code.
- **`assert_inlineable_auth`** (`finelog/deploy/config.py:131-142`) already
  implements the exact policy #6873 wants: it **refuses to inline a `jwt` layer's
  HS256 keys** into a plaintext deploy artifact and forces a secret source.
- Stale doc: `finelog/AGENTS.md:29-31` still claims "Finelog ships **no auth**"
  — contradicted by `auth.rs`. Fix as part of this work.

## 4. The plaintext-secret path (#6873)

- `config_to_dict` (`iris/cluster/config.py:1323-1328`) is
  `model_dump(mode="json", exclude_none=True)` — **no redaction**. Secret
  fields are plain `str`, so configured values are emitted verbatim.
- Secret-bearing config fields: `ClusterFinelogConfig.delegation_key` /
  `.static_token` (`config.py:690-693`), `PeerConfig.static_token`
  (`config.py:661-664`), `StaticAuthConfig.tokens` (`config.py:506`),
  `IapAuthConfig.oauth_client_secret` (`config.py:512`), `WorkerConfig.auth_token`
  (`config.py:402`). (`CoreweaveControllerConfig.tls_secret` is a k8s Secret
  *name*, not raw material — safe.)
- Emitted into two broadly-readable artifacts: the k8s ConfigMap
  (`_config_json_for_configmap`,
  `iris/cluster/platforms/k8s/controller.py:1131-1136`, mounted at
  `/etc/iris/config.json`) and the GCE VM startup-script instance metadata
  (`iris/cluster/platforms/gcp/controller_bootstrap.py:264` →
  `iris/cluster/platforms/gcp/handles.py:217-223`). Both are readable by anyone
  with `get configmap` / `compute.instances.get`. Note: the controller's
  ClusterRole grants **no** `secrets` access (`platforms/k8s/controller.py:678-728`)
  and deliberately resolves operator secrets in the operator's shell — the
  controller "never has these secrets" (`platforms/k8s/controller.py:397-399`);
  a runtime secret-read path must not silently invert that posture.
- **No Secret Manager anywhere in the controller runtime.** The only in-cluster
  secret channel is a k8s Secret + `envFrom` for task env (`inject_env.py`,
  `k8s/controller.py:952-966`), sourced from the operator's shell env. GCP
  Secret Manager appears only in CI tooling (`infra/tpu-ci/setup.py:58-94`).
- Resolve boundary: `load_config` (`config.py:1287-1315`) is the single
  deserialization point for both artifacts (`main.py:337`) — the natural place
  to resolve a secret reference before secrets are read at
  `finelog_relay.py:80-84` / `federation/peer.py:106`.
- `rigging/redaction.py` already has `SENSITIVE_KEY_RE` + `is_sensitive_key_name`
  (`redaction.py:26,71`) matching `token|secret|password|key|…` — used for log
  scrubbing, **never for config serialization**. Reusable to detect which
  fields are secret-bearing.

## 5. The four gaps → linked issues

| Gap | State | Issue |
|---|---|---|
| Two auth implementations (rigging code-assembled, finelog declarative JSON) drift; different default posture, ordering, no shared CIDR schema | Divergent | **#6861** |
| Controller shared secrets (`delegation_key`, `static_token`, `tokens`) ship in plaintext ConfigMap / GCE metadata; no secret-reference path | Insecure | **#6873** |
| No headless service-account onboarding — SA→IAP flow is implemented but standing up a CI/cron identity is undocumented tribal knowledge; jobs fall back to SSH tunnels | Undocumented | **#6580** |
| iris `login` auto-provisions any IAP-admitted identity at write-capable `"user"`; no allowlist / role-grant path; users tracked inconsistently | Loose | **#6592** |
| No single "roll out a new authed service" doc | Missing | *(this umbrella, #6942)* |
| Native `/proxy` ingress rollout + `MAX_ENDPOINT_TOKEN_TTL` question | Ops rollout | **#6937** (answered: 24h) / **#6857** (merged) |

## 6. Prior design docs (already decided — do not contradict)

- `2026-06-20_rigging_connection_auth.md` — the `rigging.connect` client seam
  (shipped). Auth split invariant: transport-generic injectors in rigging;
  minting/roles/loopback/token-store/login in iris. rigging is a dependency leaf.
- `2026-07-02_iris_per_endpoint_ingress_auth.md` — per-endpoint ingress auth
  (PR #6857, merged): `EndpointAccess {PRIVATE,PUBLIC,BEARER}`, scoped
  `VerifiedIdentity.audience`, `MintEndpointToken`, route-scoped auth moved into
  `rigging.server_auth`.
- `20260312_iris_auth_design.md` — the foundational JWT model (HS256,
  `controller_secrets` key, 30-day TTL, in-memory revocation, roles). Deferred:
  unauthenticated bundle downloads, no token refresh, single-role model.
- `iris/docs/iap-gclb.md`, `iris/docs/auth-loopback-transition.md` — the IAP
  edge and loopback-trust operator references; both current.

**Canonical invariants any new design must preserve:** (1) auth split by layer,
rigging is the leaf; (2) one chain everywhere, mount unconditionally; (3)
default-deny for unannotated routes / unconfigured stacks; (4) network trust
only for genuine direct socket peers (never `X-Forwarded-For`); (5) edge auth
(IAP, `Proxy-Authorization`) composed as a unit with app auth (JWT,
`Authorization`); (6) scoped tokens are proxy-only, denied at every RPC.
