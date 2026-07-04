# Unified server auth & secret configuration

_Why are we doing this? What's the benefit?_

Server auth is already centralized in `lib/rigging` — verifiers, the
authenticator chain, and the two enforcement points a service mounts
unconditionally ([`server_auth.py`](https://github.com/marin-community/marin/blob/5a6f64cbeef5e1962ed367deb3aaf72956ddb4d1/lib/rigging/src/rigging/server_auth.py)).
The "sloppiness" the audit names is four concrete gaps *around* that seam, each
already tracked as a sub-issue. This doc is the umbrella: it picks a single
standard pattern for **defining a new authed service**, **where auth config and
secrets live**, and **how a client finds a service and its tokens** — so those
sub-issues (#6861, #6873, #6580, #6592) close out as one coherent posture
instead of four point fixes. Background and full current-state map:
[`research.md`](./research.md).

## Challenges

The hard part is that auth is enforced by *two* implementations that already
mirror each other conceptually but drift in detail. rigging assembles its chain
**in Python code** (`RequestAuthPolicy.enforcing(...)`); finelog parses a
**declarative JSON layer stack** in Rust (`FINELOG_AUTH_POLICY`,
[`auth.rs`](https://github.com/marin-community/marin/blob/5a6f64cbeef5e1962ed367deb3aaf72956ddb4d1/lib/finelog/rust/src/server/auth.rs)).
They differ in default posture (finelog is strictly default-deny with an
allow-localhost fallback; rigging only installs authenticators when a verifier
is present), layer ordering (finelog cidr-first, rigging jwt-first), and layer
vocabulary (finelog has a general `cidr` layer; rigging had only `Loopback`
until `CidrAuthenticator` was added). The IAP/Google verifiers are inherently
Python (`google-auth`), so they can never move into finelog's Rust — only the
*pure* layers (cidr, HS256-jwt, the walk) overlap. Any unification must respect
that split.

The second hard part is secret handling: the finelog **server** already refuses
to inline HS256 keys into a plaintext deploy artifact
(`assert_inlineable_auth`), but the iris **controller config** inlines the same
class of secret (`delegation_key`, `static_token`, `StaticAuthConfig.tokens`)
straight into a world-readable ConfigMap / GCE startup-metadata via
`config_to_dict` (#6873). We need one secret-supply abstraction both sides use.

## Costs / Risks

- Churn in a load-bearing, security-sensitive path with no user-visible feature.
  Every regression here is a potential auth bypass, so changes must be
  test-gated and rolled out behind the existing default-deny invariants.
- A shared declarative schema adds a config surface (and a cross-language
  contract test) that must stay in lockstep with two parsers.
- The `secrets.py` GCP path adds an optional `google-cloud-secret-manager`
  dependency; it must stay an extra so the rigging leaf stays light.
- The pyo3 "shared engine" option (see Open Questions) would turn rigging from a
  pure-Python leaf into a compiled wheel that every service builds — a real
  packaging cost we should not pay unless the schema-only unification proves
  insufficient.

## Design

Five parts. Parts 1–2 are the new mechanism; 3–5 apply it consistently. The
parts are independently landable (Part 2 does not depend on Part 1; Part 5 adds
a new RPC; the #6592 flip ships as its own PR with an ops announcement) — this
is one design, but the implementation splits along those seams.

**1. Declarative auth-stack schema (#6861).** Introduce
`rigging.auth_config.AuthStackConfig` — an ordered list of typed layers parsed
from JSON/YAML, deny-by-default. This is the *request-chain* schema; it does
**not** model the login-exchange verifiers (`static`/`gcp`/`iap_id_token`),
which stay in code behind the `Login` RPC, since the request chain's verifier is
always the service JWT (`JwtTokenManager`, `auth.py:447-470`). Request-stack
layer types: `jwt` (with an `optional` flag for the best-effort / permissive
case), `iap_assertion`, `cidr`, `loopback`, and a terminal `anonymous`. rigging
gains `RequestAuthPolicy.from_config(stack, *, jwt_verifier, iap_assertion_verifier)`
that compiles the layer list into the existing authenticator chain;
`enforcing()`/`permissive()` become thin wrappers over it, and iris's
`request_auth_policy(config.auth)` (`auth.py:340-353`) builds a stack instead of
assembling in code. **No behavior change** is pinned by a state→stack table
(spec §1.3) mapping every current `ControllerAuth` provider to its exact
compiled chain, verified by re-running `test_server_auth.py` against the config
path.

What is genuinely *shared* with finelog is narrower than "one schema, two
parsers": the **ordered-list wire convention, the default-deny/allow-localhost
semantics, and the `cidr` layer** — pinned by a cross-language contract test
(finelog has the seed, `test_config.py:136-163`). The `jwt` layer differs by
necessity: rigging injects a Python verifier (the IAP/Google verifiers are
`google-auth`-bound and cannot move to Rust's stack), while finelog embeds HS256
keys directly in the policy JSON (`JwtAuthLayer`,
[`config.py:104-117`](https://github.com/marin-community/marin/blob/5a6f64cbeef5e1962ed367deb3aaf72956ddb4d1/lib/finelog/src/finelog/deploy/config.py#L104)).
The win is that the two stacks stop drifting in *composition* (default posture,
ordering, CIDR vocabulary) — which is exactly the drift #6861 names.

**2. `rigging/secrets.py` — reference-based secret supply (#6873).** A secret
reference resolved by scheme, so config carries a *reference*, never raw
material:

```python
# rigging/secrets.py
def resolve_secret(ref: str) -> str:
    """env:NAME | gcp-secret://projects/<p>/secrets/<n>/versions/<v> | file:/path | raw literal.
    A scheme-shaped ref (^[a-z0-9+-]+:) with an unknown scheme raises — never a silent literal fallback."""
```

The resolve boundary is the **controller runtime**, not the shared loader: the
deploy path is `load_config → config_to_dict → ConfigMap/metadata`, so resolving
in `load_config` would either re-inline raw secrets into the artifact or make
every reference-configured deploy fail the guard. Instead `load_config` parses
only; the controller `serve` entrypoint resolves the referenced fields once
after load, before building consumers (`finelog_relay.py`, `federation/peer.py`).
The five secret-bearing fields are marked with an explicit
`SecretRefStr` annotation (**not** a name heuristic — the existing
`is_sensitive_key_name` regex misses `delegation_key` and false-matches the whole
`auth` block). The two artifact-render sites
(`_config_json_for_configmap`, `controller_bootstrap.py:264`) call
`assert_no_inlined_secrets`, which walks only the marked fields and raises if any
non-empty value is not a reference — mirroring finelog's `assert_inlineable_auth`.
**Where JWT secrets live** (the umbrella's explicit question): per-service
*signing* keys stay **minted on the controller** in `controller_secrets`
(`auth.py:157-181`) — never in config. Only *cross-process shared* secrets
(finelog `delegation_key`, peer `static_token`) use a reference. Mint-on-server
is the default; Secret Manager is reserved for what must be identical across
processes. The k8s path is the shipped one — a Secret + `envFrom` → `env:NAME` —
so we deliberately do **not** add a `k8s-secret://` scheme (it would need
`secrets: get` on the controller ClusterRole, which today grants none —
`platforms/k8s/controller.py:678-728,397-399`); GCE uses `gcp-secret://` (the VM
SA needs `roles/secretmanager.secretAccessor` on the referenced secrets).
finelog's Rust server already parses `FINELOG_AUTH_POLICY` itself, and
`assert_inlineable_auth` already forces its whole jwt-bearing policy through a
secret source (env/k8s Secret) — so finelog needs no new Rust resolver.

**3. Consistent posture for iris + finelog.** Both express their request stack in
the schema from Part 1: default-deny, allow-localhost fallback, a `cidr` layer
for direct-VPC/loopback trust (both already distrust `X-Forwarded-For` —
`server_auth.py:313-345`, `auth.rs:35-36`), a `jwt` layer, IAP in front. finelog
gets a docs fix (`finelog/AGENTS.md:29-31` still claims finelog "ships no auth",
contradicted by `auth.rs`) and its jwt policy continues to flow through a secret
source per Part 2.

**4. Standard patterns + the rollout doc (the missing artifact).** A new page,
`lib/rigging/docs/authed-service.md`, walks a service author through the recipe:
(a) mount `PolicyAuthInterceptor` + `RouteAuthMiddleware` unconditionally; (b)
declare the auth stack in config (Part 1); (c) inject a `TokenVerifier` + role
resolver (mint-on-server JWT, or reuse iris's); (d) annotate routes
`@public`/`@requires_auth`; (e) read identity via `get_verified_identity()` and
authorize against the service's own policy; (f) front with IAP (`iap_gclb.py`,
GCP) or Traefik (`install_traefik_proxy.py`, CoreWeave), or expose as an Iris
endpoint at `/proxy/<name>`; (g) shared secrets via a reference. The doc calls
out the sharp edge that a `cidr` layer grants `ANONYMOUS_ADMIN`
(`server_auth.py:493-528`): list operator-trust ranges only, never an ingress
hop's source ranges. The **client** recipe sits alongside:
`credentials_for(cluster, auth)` → `ClientCredentials.interceptors()` →
`connect(transport, factory, auth=...)`, with the human vs SA vs loopback paths.

**5. Headless SA onboarding + user tightening (#6580, #6592).** Add an `iris user
grant` CLI (it is the only non-startup role-grant path, for humans and service
accounts alike) that orchestrates the three distinct planes: (1) the gcloud
`roles/iap.httpsResourceAccessor` IAM binding [operator-local], (2) *print* the
`auth.iap.audiences` config edit if the IAP client id is absent [config change,
needs redeploy — cannot be applied live], and (3) a new admin-only `SetUserRole`
RPC that provisions the identity at a chosen role in the user store [live
controller]. The `SetUserRole` RPC is what #6592 needs — today there is no
role-change RPC (`set_user_role` runs only at startup). With it, the #6592 flip
is scoped to **IAP clusters only**: `login` provisions IAP identities at
`unprovisioned_role` (read-only `dashboard`) instead of write-capable `"user"`,
making the user store the allowlist. `gcp`/`static` login provisioning is left
as-is, because there the login verifier (e.g. `GcpAccessTokenVerifier`'s project
check, `server_auth.py:189-196`) already *is* the allowlist. The runbook notes
the claims trap: role is baked into the 30-day JWT and verification never reads
the user store, so a grant takes effect only after the user re-runs `iris login`.

## Testing

Unit: the shared `cidr`/walk contract test (rigging-parsed ≡ finelog-parsed,
extended from `test_config.py:136-163`); `resolve_secret` per scheme with a faked
Secret Manager / env, including that an unknown scheme-shaped ref *raises*.
Behavioral: `RequestAuthPolicy` round-trip tests that every entry in the
state→stack table (spec §1.3) — built `from_config` — admits/denies identically
to today's `enforcing()`/`permissive()` chains, re-running the existing
`test_server_auth.py` cases (loopback trust, `X-Forwarded-For` spoof rejection,
best-effort worker-JWT attribution, scoped-token RPC denial) against the config
path. The load-bearing #6873 test is a **deploy-path round trip**: a
reference-configured cluster → the rendered ConfigMap / GCE metadata contains the
*reference string* (never the value) and `assert_no_inlined_secrets` passes,
while a raw-secret field raises; then the controller `serve` path resolves the
reference at startup. Rollout: exercised against the Iris dev controller (old
workers + new controller) before marin.

## Open Questions

- **Schema-only vs a pyo3 shared engine.** The maintainer OK'd a Rust layer in
  rigging via pyo3. Recommendation: **do the schema unification first** (Part 1,
  no packaging change) and treat a shared Rust *engine* (extracting the pure
  cidr/hs256/walk primitives into a crate both link) as a deferred, separately-
  designed Phase 2. The honest reasoning: (a) the observed drift is *composition*
  drift — default posture, ordering, CIDR vocabulary — which a shared schema
  fixes and an engine would not; (b) rigging's chain must *interleave*
  Python-only IAP layers with the pure ones, so a Rust engine could only own the
  cidr/hs256 primitives (a few hundred lines already pinned by the contract
  test), not the chain; (c) the real cost of pyo3 is the build matrix and losing
  the pure-Python leaf, not "shipping a wheel" (finelog already ships prebuilt
  wheels). Do reviewers want to commit to the pyo3 engine now regardless?
- **Secret backend on k8s.** This design uses `gcp-secret://` for GCE and the
  shipped Secret-`envFrom`→`env:NAME` pattern for k8s, and adds an RBAC-free
  `file:/path` scheme for mounted Secret volumes. Is that the posture reviewers
  want, or is a first-class runtime Secret-Manager fetch (accepting the added
  controller IAM) preferred on k8s too?
- **#6592 rollout.** The flip to provision IAP identities at read-only
  `unprovisioned_role` is a behavior change for every IAP cluster (today first
  login grants write). Land it in this umbrella behind an ops announcement, or as
  a follow-up once `SetUserRole` exists so operators can pre-grant current users?
- **Out of scope (confirming, not asking):** unauthenticated bundle downloads
  (deferred in `20260312_iris_auth_design.md`), 30-day-JWT refresh, and
  generalizing scoped tokens to a `scopes` set stay out — flag if you disagree.
