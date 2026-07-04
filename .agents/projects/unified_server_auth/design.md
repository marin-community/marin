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
until `CidrAuthenticator` was added). The *server-side* verify overlap (cidr,
jwt, and — contrary to the first draft — IAP/Google token verification, which is
just JWKS-based JWT crypto, feasible in Rust) is what a shared engine could own;
the split that genuinely stays Python is client-side token *minting* (see
Phasing).

The second hard part is secret handling: the iris **controller config** inlines
shared secrets (`delegation_key`, `static_token`, `StaticAuthConfig.tokens`)
straight into a world-readable ConfigMap / GCE startup-metadata via
`config_to_dict` (#6873), even though the finelog **server** already refuses to
inline the same class (`assert_inlineable_auth`). The greenfield token switch
(Part 0) removes the *largest* offender at the root — `delegation_key` becomes a
public key — leaving a smaller residue that a secret-supply abstraction handles.

## Costs / Risks

- Churn in a load-bearing, security-sensitive path with no user-visible feature.
  Every regression here is a potential auth bypass, so changes must be
  test-gated and rolled out behind the default-deny invariants.
- The token-format switch (Part 0) is cheap *now* — greenfield, ~3 sites — but it
  is a breaking change; do it before any token is issued, not after.
- Asymmetric tokens add **key rotation** as an operational concern (JWKS overlap
  windows) that a single symmetric secret didn't have. Standard, but real.
- A shared declarative schema adds a config surface + a cross-language conformance
  suite that must stay in lockstep.
- The `secrets.py` GCP path adds an optional `google-cloud-secret-manager`
  dependency; it stays an extra so the rigging leaf stays light.
- The Phase-2 Rust engine would give rigging a compiled wheel (native artifacts
  for a broadly-installed client lib) — deferred and gated, not free.

## Design

**No service tokens are deployed yet** — the token surface is HS256, fully
contained in `JwtTokenManager` (two mint sites + one verify, `auth.py:224,248,262`)
plus finelog's Rust HMAC verify. That greenfield window makes one foundational
choice nearly free, and five parts follow from it. The parts are independently
landable (the #6592 flip ships as its own PR with an ops announcement) — one
design, split along those seams.

**0. Foundational: asymmetric, JWKS-verified service tokens.** Because nothing is
deployed, adopt **public-key JWTs (EdDSA/Ed25519) now** instead of HS256. The
controller is the **signing authority**: the private key is minted on the
controller (`controller_secrets`, unchanged storage), and public keys are
published at a `/.well-known/jwks.json` endpoint. Every verifier — finelog, peers,
the request chain — holds only the **public** key. This is standard (RFC 8037;
PyJWT `algorithm="EdDSA"`, Rust `jsonwebtoken` EdDSA) and reshapes the rest:

- **The shared-secret class that #6873 is about dissolves.** `delegation_key` today
  is a *shared symmetric HMAC secret* that finelog verifies with
  (`auth.rs`, `hmac`, `MIN_SECRET_BYTES`) and "anyone who reads it can mint." As
  an asymmetric setup it becomes the controller's **public** key — safe to inline
  and distribute, not a secret — so finelog verifies delegation JWTs with no
  secret at all. The `SecretRef` machinery (Part 2) shrinks to the small residue
  of genuinely-symmetric material (dev static tokens, the IAP OAuth client
  secret).
- **Mint and verify decouple** — exactly the "controller could mint keys for iris
  etc." idea: the controller signs; iris/finelog/services/peers only verify. None
  of them need mint capability, so a compromised verifier can't forge tokens.
- **Federation and `/proxy` paths get simpler, not harder.** Each cluster's
  controller is its own JWKS *issuer*; a verifier (including a `/proxy` gate
  checking a scoped token that was minted by another cluster) resolves the right
  public key by the token's `iss` claim — the standard cross-issuer JWKS pattern,
  with no shared secret to distribute. The one thing to get right is `iss`-keyed
  key resolution + rotation (JWKS overlap), which is well-trodden.
- **One verification mechanism everywhere.** JWKS-based JWT verify already serves
  the IAP assertion (ES256) and Google ID token (RS256); service tokens (EdDSA)
  now use the *same* machinery, and finelog verifies them with the same
  `jsonwebtoken` crate the Rust engine (Phase 2) would use.

Five parts follow.

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

The schema shape follows **Istio's `RequestAuthentication`/`AuthorizationPolicy`**
(external prior art, research §7.1): the `jwt` layer mirrors Istio's `jwtRules`
(`issuer`, `audiences`, `jwksUri`/inline `jwks`, `fromHeaders`+prefix,
`algorithms`), and the `cidr` layer mirrors Istio's explicit proxy-trust model —
a `numTrustedProxies`-equivalent and an explicit choice between *peer socket IP*
and *`X-Forwarded-For`-derived* client IP, so trust is never silently lent to a
forwarded header (we keep our existing socket-peer-only default). We keep
first-match (Istio unions ALLOW with DENY-precedence) and document that
divergence.

What is *shared* with finelog is the **ordered-list wire convention, the
default-deny/allow-localhost semantics, and the `cidr` layer**. The drift fix is
the lesson from Casbin — whose N hand-maintained per-language ports are
self-admittedly non-uniform, i.e. our finelog/rigging split at ecosystem scale:
a **shared conformance test-vector suite** (input request → expected verdict/trace)
that *both* the Python and Rust evaluators must pass in CI, seeded from finelog's
existing pin (`test_config.py:136-163`). The `jwt` layer stays per-implementation
in Phase 1 (rigging injects a Python verifier; finelog embeds HS256 keys in the
policy JSON, `JwtAuthLayer`,
[`config.py:104-117`](https://github.com/marin-community/marin/blob/5a6f64cbeef5e1962ed367deb3aaf72956ddb4d1/lib/finelog/src/finelog/deploy/config.py#L104)),
because unifying the *evaluator* is the optional Rust-engine phase below, not the
schema. The Phase-1 win is that the two stacks stop drifting in *composition*
(default posture, ordering, CIDR vocabulary) — exactly the drift #6861 names —
with zero packaging change.

**2. `rigging/secrets.py` — reference-based secret supply, resolved over an
ordered source path (#6873).** A secret field is a `SecretSpec`: an **ordered list
of references**, resolved top-to-bottom, first-*present* wins. The reference
conventions match established practice (systemd `LoadCredential`, 1Password
`op://`, Vault `vault:path#key`; research §7.2):

```python
# rigging/secrets.py — resolve an ordered path, first present wins
def resolve_secret_spec(spec: SecretSpec) -> ResolvedSecret:   # {value, source}
    """Try each source in order:
        env:NAME | file:/abs/path | gcp-secret://projects/<p>/secrets/<s>/versions/<v>
    ABSENT here (unset env / missing file / secret NOT_FOUND) -> try the next source.
    FAILED here (IAM denied / unreachable / unreadable / malformed) -> raise, never fall through.
    An unknown scheme raises; a bare literal is dev-only and rejected by the render guard."""
```

The **absent-vs-failed** rule is the search-path safety discipline: an *absent*
source (unset env / missing file / secret NOT_FOUND) skips to the next, but a
*configured-but-erroring* one (denied IAM, unreachable) **fails hard** rather than
silently shadowing to a staler/weaker source — the same `ABSENT`/`REJECTED`
semantics the request-auth chain already enforces (`resolve_auth` walks on absent,
halts on reject, never downgrades), so one principle governs both stacks. The
resolving source is logged so a shadow is visible, not silent. A **default path
keyed on the field name** (`env:IRIS_DELEGATION_KEY → file:/etc/iris/secrets/… →
gcp-secret://…/versions/<v>`) means the common case needs **no per-field config**
— a new service *inherits* a secret home (the umbrella's goal), and each
environment populates whichever link it uses. That order matches the
platform-injection preference (research §7.2): on GKE, External Secrets Operator
or the CSI driver puts the secret at `env:`/`file:` so the controller links no
Secret-Manager SDK; only GCE reads `gcp-secret://` directly via the attached SA.
`gcp-secret://` mirrors GCP's resource name, so its **version segment is
mandatory** (pin in prod, not `latest`). No `k8s-secret://` scheme: a runtime k8s
Secret read would need `secrets: get`, which the controller ClusterRole grants
none of today (`platforms/k8s/controller.py:678-728`).

The resolve boundary is the **controller runtime**, not the shared loader (spec
§2.1): resolving in `load_config` would either re-inline raw secrets into the
rendered ConfigMap/metadata or fail the guard on every reference-configured
deploy. So `load_config` parses only; the `serve` entrypoint resolves the marked
fields after load. Fields carry an explicit `SecretRefSpec` annotation (**not** a
name heuristic — `is_sensitive_key_name` misses `delegation_key` and false-matches
the whole `auth` block), and the two render sites call `assert_no_inlined_secrets`,
which raises if **any entry in a path** is a raw value (mirroring finelog's
`assert_inlineable_auth`). **Where JWT secrets live** (the umbrella's explicit
question): with Part 0, the answer is cleaner than "reference the shared secret" —
the controller's **private signing key** stays minted on the controller
(`controller_secrets`, `auth.py:157-181`), never in config; **public** keys go out
via JWKS (not secrets); and `delegation_key` — #6873's headline symmetric secret —
is **retired**, replaced by the controller's public key that finelog verifies with.
So `SecretSpec` covers only the residual genuinely-symmetric material: dev/CI static
tokens (`StaticAuthConfig.tokens`) and the IAP OAuth client secret. Whether
`peers.static_token` survives at all is an open question (Part 0 lets a peer verify
via JWKS instead of a pre-shared bearer). finelog's Rust server parses
`FINELOG_AUTH_POLICY` itself, and `assert_inlineable_auth` already forces its whole
jwt-bearing policy through a secret source — so finelog needs no new Rust resolver.

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

**5. Headless SA onboarding + user tightening (#6580, #6592).** The recommended
answer to headless IAP access is **Workload Identity Federation — keyless**
(research §7.2): it retires both the downloaded SA key *and* the SSH-tunnel
fallback the cost-manager job fell back to. GitHub CI mints its OIDC token → WIF →
impersonates a dedicated minimal `iap-caller` SA → `token_format: id_token`,
`id_token_audience = <IAP client id>`; GCE/GKE cron uses its attached SA →
`generateIdToken(audience=<IAP client id>)`. (A custom-audience IAP ID token needs
the impersonation path, so we keep one keyless `iap-caller` SA as the impersonation
target — but never export its key.) An `iris user grant` CLI (the only non-startup
role-grant path, humans and service accounts alike) orchestrates the three planes:
(1) the gcloud `roles/iap.httpsResourceAccessor` IAM binding — and, for a headless
SA, the `workloadIdentityUser` + `serviceAccountTokenCreator` WIF bindings —
[operator-local]; (2) *print* the `auth.iap.audiences` config edit if the IAP
client id is absent [config change, needs redeploy — not applied live]; (3) a new
admin-only `SetUserRole` RPC that provisions the identity at a chosen role [live
controller]. `SetUserRole` is what #6592 needs — today there is no role-change RPC
(`set_user_role` runs only at startup). With it, the #6592 flip is scoped to **IAP
clusters only**: `login` provisions IAP identities at `unprovisioned_role`
(read-only `dashboard`) instead of write-capable `"user"`, making the user store
the allowlist. `gcp`/`static` login provisioning is left as-is, because there the
login verifier (`GcpAccessTokenVerifier`'s project check, `server_auth.py:189-196`)
already *is* the allowlist. Claims trap for the runbook: role is baked into the
30-day JWT and verification never reads the user store, so a grant takes effect
only after the user re-runs `iris login`.

## Testing

Unit: the **shared conformance test-vectors** (input request → expected
verdict/trace) run against both the rigging and finelog evaluators, seeded from
finelog's pin (`test_config.py:136-163`) — this is the drift gate (research §7.1,
the Casbin lesson). `resolve_secret_spec` per scheme with a faked Secret Manager /
env, covering the **absent-vs-failed** rule (an unset env skips to the next
source; a denied-IAM source *raises* rather than falling through) and that an
unknown scheme raises. Behavioral: `RequestAuthPolicy` round-trip tests that every
entry in the state→stack table (spec §1.3) — built `from_config` — admits/denies
identically to today's `enforcing()`/`permissive()` chains, re-running the
existing `test_server_auth.py` cases (loopback trust, `X-Forwarded-For` spoof
rejection, best-effort worker-JWT attribution, scoped-token RPC denial) against
the config path. The load-bearing #6873 test is a **deploy-path round trip**: a
reference-configured cluster → the rendered ConfigMap / GCE metadata contains the
*reference string* (never the value) and `assert_no_inlined_secrets` passes, while
a raw-secret field raises; then the controller `serve` path resolves the reference
at startup. Rollout: exercised against the Iris dev controller (old workers + new
controller) before marin.

## Phasing & the Rust engine

The work lands in phases; the Rust engine is an **optional final phase**, and the
research (§7.3) changes what that phase is:

- **Phase 1 (this design's core):** the declarative schema + conformance vectors +
  `secrets.py` + the `SetUserRole`/WIF onboarding. No packaging change; kills the
  *observed* (composition) drift and closes the plaintext-secret leak.
- **Phase 2 (final, optional): a shared Rust *verify* engine.** I was wrong in the
  first draft that "the IAP/Google verifiers can't move to Rust." They can:
  **server-side verification is standard JWT crypto** — IAP's
  `X-Goog-IAP-JWT-Assertion` is an ES256 JWT (keys at
  `gstatic.com/iap/verify/public_key`), a Google login ID token is RS256 (JWKS at
  `googleapis.com/oauth2/v3/certs`), both verify natively with `jsonwebtoken` + a
  JWKS cache, and finelog **already** verifies HS256 + CIDR in Rust. A pyo3/abi3
  wheel **already ships from this repo** (`lib/finelog/rust/pyext/`,
  `abi3-py312`), so the engine is *generalizing finelog's `auth.rs` + porting
  `server_auth.py`* behind an existing packaging pattern — not greenfield. finelog
  would link the crate natively; rigging via pyo3. The engine returns
  `VerifiedIdentity{email, matched_layer}` and Python assigns the role (the
  email→role resolver hits iris's DB — keeping it a Python callback avoids GIL
  contention and matches today's layering).
- **Client-side minting stays Python — and stays *standard*, not bespoke.** The
  concern "do we need something bespoke?" resolves by adopting standard flows and
  keeping them in Python's mature libraries: **WIF (keyless)** for headless
  (research §7.2 — removes the SA-key handling *and* the SSH-tunnel fallback), the
  standard **installed-app OAuth** flow for humans, and **`iris login` as an
  OAuth2 token exchange (RFC 8693)** — an IAP-verified Google identity in, a
  controller-signed EdDSA service JWT out. None of that is a bespoke *format*; it's
  `google-auth`/`google-auth-oauthlib` library calls. The only slightly-custom code
  is the console-paste variant of the desktop flow for SSH sessions, kept purely as
  a convenience. Crucially, minting is *not* ported to Rust — so there is no bespoke
  refresh-token re-mint to reimplement; the "long pole" was an artifact of
  considering a Rust port, and it disappears by keeping mint in Python.
- **Why the engine is last, honestly:** not capability — verification is easy and
  half-built. It is (a) marginal value: Phase 1 already removes the drift we
  actually see; the engine removes *latent* semantic drift the conformance vectors
  already largely cover; (b) turning the broadly-installed rigging leaf into a
  native-wheel package is a real distribution change (musllinux/older-manylinux
  reach) worth deferring until verify-first proves the pipeline. **Recommendation:
  build Phase 2 only if the conformance vectors surface semantic drift the schema
  can't prevent** — and if built, make it **verify-only**, keeping mint in Python.

## Open Questions

- **Signature algorithm: EdDSA vs ES256.** Recommendation **EdDSA (Ed25519)** —
  small, fast, modern, one curve; supported by PyJWT (`cryptography`) and Rust
  `jsonwebtoken`. ES256 is the more conservative choice if some future non-Rust/
  non-Python verifier matters. Either is standard; pick one.
- **Does any symmetric bearer survive?** With Part 0, `delegation_key` retires and
  a peer could verify via JWKS instead of `peers.static_token`. Do we keep a
  symmetric static-token path at all (dev/CI convenience, guarded by `SecretSpec`),
  or make *every* production path asymmetric and treat static tokens as test-only?
- **Biscuit — optional attenuation on top, decided separately.** With asymmetric
  JWTs as the standard core, Biscuit's marginal value narrows to *offline
  attenuation* for the scoped `/proxy` capability tokens (#6857): a holder narrows
  a token locally (self-mint), dissolving the "in-job processes can't self-mint"
  limitation. It is a *token format* replacing the `jwt` **layer**, not the stack
  (CIDR/loopback + Google/IAP verifiers stay), and the asymmetric win it also
  offered is now already ours via Part 0. Recommendation: skip Biscuit for the
  core; evaluate it separately *only if* offline capability-attenuation is wanted.
- **Commit to the Phase-2 Rust verify-engine now, or gate it on drift?**
  Recommendation: gate it on the conformance vectors showing real semantic drift;
  verify-only; mint stays Python. Schedule unconditionally instead?
- **Ordered `SecretSpec` default path** (adopted per your suggestion:
  `env: → file: → gcp-secret://`, first-present-wins, fail-hard on a
  configured-but-erroring source). Keep the default field-name→source mapping, or
  require every secret to be an explicit list?
- **#6592 rollout.** The flip to provision IAP identities at read-only
  `unprovisioned_role` is a behavior change for every IAP cluster. Land it in this
  umbrella behind an ops announcement, or as a follow-up once `SetUserRole` lets
  operators pre-grant current users?
- **Out of scope (confirming, not asking):** unauthenticated bundle downloads
  (deferred in `20260312_iris_auth_design.md`) and 30-day-JWT refresh stay out —
  flag if you disagree.
