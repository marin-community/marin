# Spec — unified server auth & secret configuration

Contracts pinned by `design.md`. New code lives in `lib/rigging` (the shared
leaf); iris and finelog consume it. Signatures are the public surface reviewers
agree to; bodies are illustrative. Paths are under `main` at
`5a6f64cbeef5e1962ed367deb3aaf72956ddb4d1`.

## 0. Token format & signing authority (greenfield — Part 0)

Service tokens switch from HS256 to **asymmetric EdDSA (Ed25519)**. Nothing is
deployed, so this is a contained edit (mint sites `auth.py:224,248`; verify
`auth.py:262`; finelog `auth.rs`; relay `finelog_relay.py`), not a migration. The
key is **per-cluster** (each controller is its own `iss`); a single marin-wide
private key is rejected (shared blast radius).

### 0.1 Per-plane audience discipline (load-bearing security invariant)

One signing key removes the incidental plane isolation a *dedicated* symmetric
`delegation_key` gave, so audience binding is **mandatory** (RFC 8725). Every
minted token carries an `aud` naming exactly one plane, and every verifier
**requires** its expected `aud`:

| Token | `iss` | `aud` | other claims | TTL | Verified by |
|---|---|---|---|---|---|
| control-plane session (dev-only) | `<cluster>` | `iris` | `role`, `jti` (log-correlation only) | ~1h (`SESSION_TOKEN_TTL_SECONDS`) | **only** the issuing controller |
| control-plane worker | `<cluster>` | `iris` | `role="worker"`, `jti` | 30d (`WORKER_TOKEN_TTL_SECONDS`, a residual) | **only** the issuing controller |
| delegation (relay→shared finelog) | `<cluster>` | `finelog` | `role="finelog-relay"` | ≤1h | any federated finelog |
| endpoint / `/proxy` | `<cluster>` | `iris-proxy` | `scope="proxy"`, `endpoint=<name>` | ≤24h | the `/proxy` gate |
| peer (**reserved, not minted**) | `<cluster-a>` | `iris-peer` | — | short | (reserved — see note below) |

The control-plane **session** row is dev-only: a deployed cluster authenticates
users via IAP and mints them no token (§0.3, §3), so the only `aud="iris"` user
token is the one `LocalCluster` mints for its in-process auto-login. `jti` is now
only a log-correlation id — there is no revocation set to look it up in (see the
stateless-verify bullet below). The `iris-peer` row is a **reserved** plane:
finelog's conformance vectors exercise it as a cross-plane rejection, but iris does
not mint peer tokens today — controller-to-controller federation uses the ordinary
rigging client-credential path (`credentials_for`).

The `aud` names the recipient **plane** — a bounded, static set — not a per-resource
value (RFC 8725). So an endpoint token's `aud` is the fixed `iris-proxy`, and the
specific endpoint goes in an `endpoint` claim the `/proxy` gate matches against the
resolved route (`dashboard.py:117`). This lets the iris control-plane verifier carry
a *fixed* `expected_audiences = {"iris", "iris-proxy"}` — enumerating the dynamic set
of endpoint names would be impossible — while still rejecting a `finelog` /
`iris-peer` token replayed at the RPC surface (the load-bearing cross-service guard).

- **Verification is stateless; there is no revocation (CHANGED FROM PLAN).** The
  planned `api_keys` jti revocation set is gone — dropped with the `api_keys` table.
  `JwtTokenManager.verify()` is a pure function of the token (signature, expiry, and
  the `aud`↔scope binding) with **no DB read**; `jti` survives only as a
  log-correlation id. Control-plane (`aud="iris"`) JWTs are still verified **only by
  the issuing controller**, and remote verifiers (finelog) only ever see short-TTL,
  plane-scoped tokens (the relay's 1h delegation token, `finelog_relay.py`). An IAP
  user holds no minted token — its role is resolved per request from the assertion —
  so deprovisioning takes effect on the next request, with no TTL to wait out.
- **iris's verify rejects an unexpected `aud`/`scope` — no fail-open.** The
  control-plane `JwksVerifier` carries `expected_audiences = {"iris", "iris-proxy"}`
  and **rejects** any token whose `aud` is outside that set; `JwtTokenManager.verify`
  then enforces the `aud`↔scope binding (an `iris-proxy` token must carry
  `scope="proxy"` + an `endpoint` claim, an `iris` token must not), so no unknown
  scope passes through as a full identity.

### 0.2 Generic mechanism in `rigging` (`lib/rigging/src/rigging/token_authority.py`, new)

```python
# generic; carries NO service policy (no role/aud semantics — the service supplies those)
@dataclass(frozen=True)
class SigningKey:                 # sourced from a SecretSpec (§2), never generated into a DB
    kid: str                      # RFC 7638 JWK thumbprint of the public key (stable, collision-free)
    ed25519_private_pem: str

class JwtSigner:
    def __init__(self, key: SigningKey, *, issuer: str): ...
    def mint(self, claims: dict, *, audience: str, ttl_seconds: int) -> str: ...  # EdDSA; header kid; iss/aud/iat/exp
    def public_jwks(self, *, also: "Sequence[PublicKey]" = ()) -> dict: ...        # current + retained-previous keys

class VerifiedClaims:             # returned by the verifier — raw verified claims, NOT an identity
    sub: str; iss: str; aud: str; scope: str | None; claims: dict

class JwksVerifier:
    def __init__(self, *, issuers: "Mapping[str, PublicKeySet]",   # iss -> trusted keys (inline or a pinned jwksUri)
                 expected_audiences: frozenset[str],               # this surface's allowed aud set (fail-closed)
                 algorithms: tuple[str, ...] = ("EdDSA", "ES256", "RS256")): ...
    def verify(self, token: str) -> VerifiedClaims: ...            # resolve key by iss+kid; check sig/exp/aud
```

`JwksVerifier` returns *claims*, not `VerifiedIdentity` — mapping `sub`/`role`/`aud`
to an identity is service policy. The `issuers` map is a **configured allowlist**; a
`jwksUri` is only ever a pinned config value, never derived from the token's `iss`
(SSRF). `expected_audiences` is required and fail-closed.

### 0.3 iris policy wrapper (thin, over the rigging primitives)

`JwtTokenManager` wraps `JwtSigner`/`JwksVerifier`: it owns role/claim semantics,
endpoint-token minting, and the `expected_aud` per surface plus the aud↔scope
binding. CHANGED FROM PLAN: it holds **no revocation source** — the `api_keys` jti
set is gone and verification never reads a DB. It also mints **no user token**: the
controller signer produces only the worker identity token (§0.1), endpoint `/proxy`
tokens (§6 of `design.md`), the relay delegation token (§0.6), and — dev-only — the
`aud="iris"` session token `LocalCluster` uses for its in-process auto-login. Human
and machine users authenticate to the controller through IAP (§3): the GCLB validates
an OIDC edge token and forwards a signed `X-Goog-IAP-JWT-Assertion` the controller
verifies per request, resolving the asserted email to a role via the config
`RolePolicy`. There is no `Login` RPC and no token-exchange endpoint.

### 0.4 Key sourcing, provisioning, rotation

- **Private key from a `SecretSpec`, not the DB.** `AuthConfig.signing_key:
  SecretRefSpec`. For the crown-jewel key the recommended path is `file:` (CSI) /
  `gcp-secret://` — `env:` is dev-only. `controller_secrets` no longer stores a
  signing key. **Ephemeral fallback:** when no `signing_key` is configured,
  `_build_jwt_token_manager` mints an **ephemeral in-process keypair** (warning that
  tokens will not survive a restart) — so a laptop / CI / a CoreWeave cluster with no
  Secret Manager still works. Persistence is a correctness requirement **only for a
  relay cluster**: `require_persistent_signing_key(relay_address, signing_key_pem)`
  (called on the serve path, `main.py`) fails fast when `finelog.relay_address` is set
  and no key is configured, because the relay's delegation token is the one token an
  external verifier (the shared finelog) pins to this controller's published key.
- **`iris cluster init-keys`** generates the Ed25519 keypair and writes the private
  half to a `SecretSpec` **destination** — `--out-file` (a `file:` reference) and/or
  `--gcp-secret projects/<p>/secrets/<name>` (uploads the key as a new Secret Manager
  version, creating the secret container when the caller has permission, and returns
  the pinned `gcp-secret://…/versions/<n>` reference) with `--accessor <principal>`
  (also grants that principal — e.g. the controller SA — `roles/secretmanager.secretAccessor`
  on the secret so it can read the key at startup) — printing the public key + `kid`
  for the trust config.
- **Rotation** needs the *previous* public key retained. `AuthConfig` carries
  `previous_public_keys: list[str]` (public, inline-safe); JWKS serves current +
  previous; verifiers (incl. finelog's `Vec<JwtKey>` per issuer, already supported)
  accept both. **Overlap window ≥ the max outstanding token TTL** (30d, set by the
  worker token) or rotation is a mass logout. Runbook (into `authed-service.md`):
  add new public key everywhere → switch the signer → wait ≥ max TTL → drop the old
  public key.

### 0.5 Public keys & JWKS

New `@public` route `GET /.well-known/jwks.json` (`dashboard.py`) serving
`signer.public_jwks(also=previous_public_keys)`.

### 0.6 finelog verifies against a public key (Rust)

- `JwtKeyEntry{cluster, secret}` → **`{cluster, public_keys: list}`** in
  `finelog/deploy/config.py` (inline public keys per issuer; a list for rotation
  overlap). `assert_inlineable_auth` is removed — a jwt layer is inline-safe by
  construction (public keys), so there is nothing to guard against inlining.
  `_validate_finelog_relay`'s 16-byte HS256 minimum
  (`config.py:1012-1020`) is removed.
- The Rust `JwtVerifier` switches `hmac` → `jsonwebtoken` EdDSA, **adds `aud` to
  `JwtClaims` and requires `aud="finelog"`** (`auth.rs:376-379,226-239`);
  `MIN_SECRET_BYTES` and the raw-ASCII-HMAC path go away. finelog holds the public
  key **inline** (no JWKS-fetch dependency on controller availability); an `iss`
  map covers multi-issuer federation.
- `finelog_relay.py`'s `_DelegationTokenProvider` mints via the controller signer
  with `aud="finelog"` (not its own `JwtTokenManager(delegation_key)`).

### 0.7 Consequence for #6873

`ClusterFinelogConfig.delegation_key`, `peers.static_token`, and
`finelog.static_token` are **removed** (peers/finelog verify via the issuer's inline
public key). No shared symmetric secret anywhere on the verify path.

### 0.8 Dependencies

PyJWT EdDSA needs the `cryptography` extra (iris pins bare `PyJWT>=2.12.0` — add
it); finelog's Rust gains `jsonwebtoken` (today it hand-rolls `hmac` JWS parsing).

## 1. Declarative auth-stack schema — `lib/rigging/src/rigging/auth_config.py` (new)

### 1.1 Scope and wire format

This schema models the **request chain** only — the authenticators that decide an
already-credentialed request. There are no login-exchange verifiers to model: pure
IAP means a user never exchanges anything for a controller token, so every request
carries its own credential and the chain is the whole surface. The `jwt` layer's
verifier is always the service JWT manager, and the `iap_assertion` layer holds the
`IapAssertionVerifier`; both are constructed in code
(`iris/cluster/controller/auth.py`). No RPC is exempt from the policy
(`_UNAUTHENTICATED_RPCS` is empty).

`AuthStackConfig` serializes to an **ordered JSON list of internally-tagged
layer objects** (`{"type": <layer>, ...}`), matching finelog's existing
`FINELOG_AUTH_POLICY` shape (`lib/finelog/src/finelog/deploy/config.py:125-128`)
and following Istio's `RequestAuthentication`/`AuthorizationPolicy` field vocabulary
(research §7.1). Order is evaluation order: first `AUTHENTICATED`/`Allow` admits,
first `REJECTED`/`Reject` denies, all-absent falls to the deny terminal — a
deliberate **first-match** model (unlike Istio's union-with-DENY-precedence), which
the `authed-service.md` runbook states explicitly so nobody assumes Cedar/Istio
semantics. Cross-impl consistency is enforced by a **shared conformance
test-vector suite** (§1.4), not by a second hand-maintained parser (the Casbin
lesson, research §7.1).

```json
[
  {"type": "jwt"},
  {"type": "iap_assertion"},
  {"type": "cidr", "cidrs": ["10.0.0.0/8", "127.0.0.0/8", "::1/128"]},
  {"type": "loopback"}
]
```

### 1.2 Layer catalog (request chain)

| `type` | Fields | Verifier the service injects | Semantics |
|---|---|---|---|
| `jwt` | `optional: bool = false` | the service's **policy** `TokenVerifier` (iris: `JwtTokenManager` wrapping `JwksVerifier` + `expected_aud` + the aud↔scope binding) — **never** the bare `JwksVerifier`, which skips the scope policy | present+valid ⇒ AUTHENTICATED; absent ⇒ ABSENT. present+invalid ⇒ REJECTED when `optional=false`, else ABSENT (the `BestEffortJwtAuthenticator` case that makes a null-auth chain attribute a valid worker JWT but never reject) |
| `iap_assertion` | — | `IapAssertionVerifier` (via `iap_assertion_verifier=`) | verifies `X-Goog-IAP-JWT-Assertion`; forged ⇒ REJECTED; absent ⇒ ABSENT |
| `cidr` | `cidrs: list[str]` | — | direct socket peer in a CIDR ⇒ `ANONYMOUS_ADMIN`; `X-Forwarded-For`/port-0 ⇒ ABSENT |
| `loopback` | — | — | genuine loopback socket peer ⇒ `ANONYMOUS_ADMIN` |
| `anonymous` | — | — | terminal: admit as `ANONYMOUS_ADMIN` (the permissive / `optional` tail) |

Rules: an **empty list raises** at parse time (total lockout — a service passes
an explicit default stack rather than relying on omission). A stack whose last
layer is not `anonymous` is default-deny (all-absent ⇒ raise ⇒ `UNAUTHENTICATED`).
The five above are the only layer types; there are no login-exchange layers (§1.1).
A `jwt`/`iap_assertion` layer whose verifier was not supplied is a build-time
`ValueError`.

### 1.3 Python API and the no-behavior-change contract

```python
@dataclass(frozen=True)
class AuthStackConfig:
    """An ordered, declarative request-auth-layer stack (see §1.1 wire format)."""
    layers: tuple[AuthLayerSpec, ...]

    @classmethod
    def from_json(cls, data: str | list[dict]) -> "AuthStackConfig":
        """Parse the wire list; raise ValueError on an empty list or unknown type."""
    def to_json(self) -> list[dict]: ...

# AuthLayerSpec is a StrEnum-tagged frozen dataclass union:
#   JwtLayer(optional: bool = False) | IapAssertionLayer() |
#   CidrLayer(cidrs: tuple[str, ...]) | LoopbackLayer() | AnonymousLayer()

# On RequestAuthPolicy (rigging/server_auth.py), replacing bespoke enforcing():
@classmethod
def from_config(
    cls,
    stack: AuthStackConfig,
    *,
    jwt_verifier: "TokenVerifier | None" = None,
    iap_assertion_verifier: "IapAssertionVerifier | None" = None,
) -> "RequestAuthPolicy":
    """Compile a declarative stack into the authenticator chain.

    A `jwt` layer binds `jwt_verifier` (as JwtAuthenticator, or
    BestEffortJwtAuthenticator when `optional=True`); an `iap_assertion` layer
    binds `iap_assertion_verifier`. Raises ValueError if a layer names a verifier
    that was not supplied, or if `stack` is empty. `enforcing()`/`permissive()`
    are reimplemented as thin wrappers that build a stack and call this.
    """
```

**No-behavior-change contract.** Every current `ControllerAuth` state
(`request_auth_policy`, `iris/cluster/controller/auth.py:340-353`) compiles to a
stack that produces the *identical* authenticator chain it builds today:

| State | Compiled stack | Notes |
|---|---|---|
| null-auth (no provider) | `[jwt(optional=true), anonymous]` | best-effort JWT attributes workers; anonymous terminal = `permissive()`. **Stays open** — a null-auth dev cluster admits every request. |
| `iap` | `[jwt, iap_assertion, cidr(trusted_cidrs)?, loopback] (+ anonymous if optional)` | the `iap_assertion` layer verifies the IAP edge assertion; the `jwt` layer verifies worker / endpoint tokens |
| `cidr`-only (`trusted_cidrs`, no provider) | `[cidr(trusted_cidrs), loopback] (+ anonymous if optional)` | |

The migration is mechanical: `request_auth_policy` builds the matching
`AuthStackConfig` and calls `from_config`. `permissive()` keeps its exact
current semantics via the `jwt(optional=true)` layer. No cluster's admit/deny
outcome changes; the round-trip test (design §Testing) is the gate.

### 1.4 Cross-impl conformance vectors

A shared, language-neutral test-vector file (e.g.
`lib/rigging/src/rigging/auth_vectors.json`) is the single source of truth for
evaluator behavior, run by both the Python (`rigging`) and Rust (`finelog`)
evaluators in CI. Each vector pins an input and the expected outcome:

```json
{
  "stack": [{"type": "cidr", "cidrs": ["10.0.0.0/8"]}, {"type": "jwt"}, {"type": "loopback"}],
  "request": {"peer": "10.1.2.3:44100", "headers": {}, "token": null},
  "expect": {"verdict": "allow", "matched": "cidr"}
}
```

Vectors cover the behavior both engines share (default posture, cidr-vs-jwt
ordering, empty-list lockout, allow-localhost fallback); `X-Forwarded-For` refusal
is iris-controller-specific — finelog trusts its in-VPC proxy, so it matches the
transport peer regardless — and is covered by rigging's own `test_server_auth.py`,
not the shared vectors. This is
the drift gate; it replaces "two parsers that happen to agree" with "one contract
both must pass" (the Casbin lesson, research §7.1). The `jwt` layer's verifier is
mocked per-language (Python injects a `TokenVerifier`; the Rust engine, if built
in Phase 2, verifies natively), so vectors assert the *walk + cidr + posture*,
which is exactly the shared surface.

## 2. Secret supply — `lib/rigging/src/rigging/secrets.py` (new)

A secret field is a `SecretSpec`: an **ordered list of references**, resolved
first-present-wins. A bare string is sugar for a one-element list.

```python
class SecretSource(Protocol):
    scheme: str                         # "env" | "file" | "gcp-secret"
    def fetch(self, locator: str) -> str | None: ...   # None ⇒ ABSENT here; raise ⇒ FAILED here

SecretSpec = tuple[str, ...]            # ordered references; a bare str normalizes to a 1-tuple

@dataclass(frozen=True)
class ResolvedSecret:
    value: str
    source: str                         # the reference that produced it (logged)

def is_secret_reference(value: str) -> bool:
    """True if `value` starts with a known scheme (env: / file: / gcp-secret://)."""

def resolve_secret_spec(spec: SecretSpec) -> ResolvedSecret:
    """Resolve an ordered secret path, first PRESENT source wins.

    Per source, dispatched on scheme prefix:
      - `env:NAME`                                           → os.environ.get(NAME)
      - `file:/abs/path`                                     → file contents (trimmed)
      - `gcp-secret://projects/<p>/secrets/<n>/versions/<v>` → Secret Manager (version REQUIRED)

    ABSENT here (env unset / file missing / secret|version NOT_FOUND) ⇒ try the
    next source. FAILED here (denied IAM / unreachable / unreadable / malformed)
    ⇒ raise SecretResolutionError immediately — NEVER fall through to a
    staler/weaker source (mirrors the auth chain's REJECTED-halts rule). A
    scheme-shaped ref (^[a-z0-9+-]+:) with an unknown scheme raises. A bare
    literal is dev-only and rejected by the render guard. Exhausting the path
    with all-ABSENT raises. Logs the resolving source (and, for gcp-secret, the
    resolved version). The GCP path imports google-cloud-secret-manager lazily
    (optional extra `marin-rigging[secrets]`).
    """

def default_secret_spec(field_name: str) -> SecretSpec:
    """The conventional path for a field with no explicit spec:
    (env:IRIS_<FIELD>, file:/etc/iris/secrets/<field>). gcp-secret:// is NOT in the
    default — its version segment is mandatory and can't be conventional — so a
    Secret-Manager source is always explicit config. Lets a service inherit a
    secret home (env/file) without per-field config."""
```

No `k8s-secret://` scheme: the k8s-native path is a Secret + `envFrom` → `env:`
(or a CSI-mounted volume → `file:`). A runtime `k8s-secret://` read would require
`secrets: get` on the controller ClusterRole, which grants none today
(`iris/cluster/platforms/k8s/controller.py:678-728`) and would invert the
documented posture that the controller "never has these secrets"
(`platforms/k8s/controller.py:397-399`). `gcp-secret://` mirrors GCP's resource
name, so the version segment is **mandatory** (pin `versions/<n>` in prod, not
`latest`; research §7.2).

### 2.1 Config-side contract (iris)

- The secret-bearing fields are typed `SecretRefSpec` (accepts a bare ref or an
  ordered list) and marked with an explicit annotation, **not** a name heuristic
  (`rigging.redaction.is_sensitive_key_name` misses `delegation_key` and matches
  the whole non-secret `auth` block):

```python
SecretRefSpec = Annotated[str | tuple[str, ...], "secret-ref"]   # bare ref or ordered path
```

  Marked fields, as built (exactly two remain): `AuthConfig.signing_key` (the
  private key, §0) and `IapAuthConfig.oauth_client_secret`. **Removed** (now
  asymmetric — verify via the issuer's inline public key, no secret):
  `ClusterFinelogConfig.delegation_key` / `.static_token` and
  `PeerConfig.static_token`. The `StaticAuthConfig.tokens` field is gone with the
  whole static-token login provider, and so is the `gcp` login provider — `AuthConfig`
  now selects a single arm, `iap` (`_ONEOF_ARMS = ("iap",)`).
  **Not** marked: `WorkerConfig.auth_token` — minted on the controller at runtime,
  always empty in an authored config, so never a reference and never guarded.
  (`previous_public_keys` are public, not secrets.)
- **Resolve boundary is the controller runtime, not the loader.** `load_config`
  (`config.py:1287-1315`) parses only. The controller `serve` entrypoint
  (`main.py:337`) calls `resolve_config_secrets(config)` after `load_config`,
  replacing each marked field with `resolve_secret_spec(spec_or_default).value`
  (falling back to `default_secret_spec(field)` when the field is unset), before
  consumers read it (`finelog_relay.py:80-84`, `federation/peer.py:106`). The
  deploy CLI (`iris cluster start`) never resolves — it renders references verbatim.

```python
def resolve_config_secrets(config: IrisClusterConfig) -> IrisClusterConfig:
    """Return a copy with every SecretRefSpec field resolved via resolve_secret_spec.
    Called once on the controller serve path; never on the deploy/render path."""
```

- **Producer guard at the render sites** (not inside generic `config_to_dict`,
  which tests and round-trips also call): `_config_json_for_configmap`
  (`iris/cluster/platforms/k8s/controller.py:1131`) and
  `build_controller_bootstrap_script_from_config`
  (`iris/cluster/platforms/gcp/controller_bootstrap.py:264`) call:

```python
def assert_no_inlined_secrets(config: IrisClusterConfig) -> None:
    """Raise ValueError if any SecretRefSpec-marked field holds a non-empty value
    where ANY entry in the path is not a secret reference (is_secret_reference is
    False) — i.e. a raw secret about to be serialized into a broadly-readable
    ConfigMap / GCE metadata. Empty ⇒ pass (unset; resolves via default path)."""
```

Per-service JWT **signing** keys are `SecretRefSpec` fields too (§0): the private
key resolves from Secret Manager at startup, never generated into the SQLite
`controller_secrets` table (which is node-local NVMe, commit `f691c03f2`, and
would lose the key on node loss).

## 3. Roles & onboarding (#6580, #6592) — SUPERSEDED

> **SUPERSEDED.** The plan below (a `SetUserRole` RPC, an `iris user grant` CLI,
> and `login`-time DB provisioning against a `users` table) was **not** built.
> Authorization is instead a **config-driven, in-memory `RolePolicy`**: at
> controller start, `create_controller_auth` builds a frozen map from `AuthConfig`
> (`auth.admin_users` → `admin`, the worker machine identity → `worker`, everyone
> else → the default role — the IAP `unprovisioned_role` on an iap cluster, `user`
> otherwise). `role_for(user_id)` answers with no DB and no reconciliation; the `users`
> and `api_keys` tables are dropped (migrations `0039_drop_users`,
> `0038_drop_api_keys`). There is no `login` RPC: a user reaches the controller
> through IAP, and `IapAssertionVerifier` resolves the asserted email to a role via
> `RolePolicy.role_for` on **every** request — no minted user token, no `login`-time
> write. Because that resolution is per request, deprovisioning takes effect on the
> next request. The **read-only default** the plan wanted is the policy default
> (`unprovisioned_role`, `dashboard`). The IAM half of onboarding — granting
> `roles/iap.httpsResourceAccessor` — still stands (see [`onboarding.md`](./onboarding.md));
> the RPC/CLI/DB half is gone. The rest of this section is retained only as a record
> of the abandoned plan.

### 3.1 (superseded) `SetUserRole` RPC — not built

The plan added an admin-only `SetUserRole` RPC (`AuthzAction.MANAGE_USER_ROLES`)
writing a `users`-table row. There is no such RPC or action; roles come from config.

### 3.2 (superseded) `iris user grant` CLI — not built

The plan added an `iris user grant` command orchestrating an IAM binding, a live
`SetUserRole` RPC, and a printed config edit. It was not built; granting a role is a
`auth.admin_users` config edit + reload, and the IAM binding is a manual/`gcloud`
step documented in [`onboarding.md`](./onboarding.md).

### 3.3 Provisioning change (#6592) — realized via the config default

The behavior the plan wanted holds, by a different mechanism: on an IAP cluster a
non-admin identity resolves to `IapAuthConfig.unprovisioned_role` (default read-only
`dashboard`) rather than a write-capable `"user"`, because that is the `RolePolicy`
default. The role is resolved per request from the verified IAP assertion — never
baked into a minted token — so a config change applies to the very next request, with
nothing to re-issue and no TTL to age out. Where IAP's own allowlist is meant to be
the sole gate, set `unprovisioned_role: admin` (as `marin.yaml` does): anyone Google
admits then acts as admin.

## 4. Files

| Path | Change |
|---|---|
| `lib/rigging/src/rigging/auth_config.py` | **new** — `AuthStackConfig`, layer specs, wire (de)serialization |
| `lib/rigging/src/rigging/auth_vectors.json` | **new** — shared cross-impl conformance vectors (§1.4) |
| `lib/rigging/src/rigging/server_auth.py` | `RequestAuthPolicy.from_config`; `enforcing`/`permissive` reimplemented on it |
| `lib/rigging/src/rigging/secrets.py` | **new** — `resolve_secret_spec`, `default_secret_spec`, `SecretSource`, `is_secret_reference` |
| `lib/rigging/pyproject.toml` | add `[secrets]` optional extra (`google-cloud-secret-manager`) |
| `lib/rigging/docs/authed-service.md` | **new** — the rollout runbook (server + client recipe; cidr-grants-admin caveat; first-match note) |
| `lib/iris/src/iris/cluster/controller/main.py` | resolve secrets on the `serve` path after `load_config` |
| `lib/iris/src/iris/cluster/platforms/k8s/controller.py` | guard at `_config_json_for_configmap` |
| `lib/iris/src/iris/cluster/platforms/gcp/controller_bootstrap.py` | guard at the render site |
| `lib/rigging/src/rigging/token_authority.py` | **new, §0** — `SigningKey`, `JwtSigner` (EdDSA mint w/ `aud`, `public_jwks`), `JwksVerifier` (returns `VerifiedClaims`; `iss` allowlist + required `expected_audiences`); verifies the EdDSA service-token plane (the IAP assertion verifies separately in `server_auth.py`) |
| `lib/rigging/src/rigging/auth.py` | `run_iap_desktop_login` — the browser IAP edge-login flow `iris login` drives; caches the edge refresh token (no cluster-token exchange). The deleted `IapIdTokenVerifier` / `GcpAccessTokenVerifier` login verifiers are gone |
| `lib/iris/src/iris/cluster/controller/auth.py` | **§0** `JwtTokenManager` = thin policy wrapper over `rigging.JwtSigner`/`JwksVerifier` (endpoint tokens, per-surface `expected_aud`, aud↔scope binding — **no** jti-revocation); the in-memory `RolePolicy` (config roles); `request_auth_policy` builds `AuthStackConfig`; `require_persistent_signing_key` (relay-scoped); **drop** DB signing-key storage, the user store, the fail-open scope path, and the `gcp` provider |
| `lib/iris/src/iris/cluster/controller/finelog_relay.py` | **§0** `_DelegationTokenProvider` mints via the controller signer with `aud="finelog"` (drop `JwtTokenManager(delegation_key)`) |
| `lib/iris/src/iris/cluster/config.py` | **§0** `AuthConfig.signing_key: SecretRefSpec` + `previous_public_keys`; **remove** `delegation_key` / `finelog.static_token` / `peers.static_token` (→ inline public keys); mark `SecretRefSpec` fields; `resolve_config_secrets`; `assert_no_inlined_secrets` |
| `lib/iris/src/iris/cluster/controller/dashboard.py` | **§0** `@public GET /.well-known/jwks.json` route |
| `lib/iris/src/iris/cli/cluster.py` | **§0** `iris cluster init-keys` (Ed25519 keypair → private to a `--out-file` / `--gcp-secret` dest, `--accessor` grants the reader SA `secretAccessor`). `iris user grant` NOT built (§3, superseded) |
| `lib/finelog/rust/src/server/auth.rs` | **§0** jwt layer verifies EdDSA via `jsonwebtoken`, **requires `aud="finelog"`** (add `aud` to `JwtClaims`), inline/`iss`-resolved public key, `Vec` for rotation; drop `hmac`/`MIN_SECRET_BYTES` |
| `lib/finelog/src/finelog/deploy/config.py` | **§0** `JwtKeyEntry{cluster, public_keys}`; `assert_inlineable_auth` removed (jwt inline-safe by construction — public keys carry no secret); share the `cidr`/walk convention + conformance vectors |
| `lib/iris/src/iris/cluster/controller/service.py` | no `login` / `GetAuthInfo` RPC — users authenticate via IAP per request. `SetUserRole` handler NOT built (§3) |
| `lib/iris/src/iris/cluster/controller/migrations/0038_drop_api_keys.py`, `0039_drop_users.py` | **new** — drop the `api_keys` and `users` tables (stateless verify; config roles) |
| `lib/iris/proto/…` | `SetUserRoleRequest` / `SetUserRoleResponse` NOT built (§3) |
| `lib/finelog/rust/src/server/auth.rs` (test) | `conformance_vectors_match_rigging` runs the shared `auth_vectors.json`; cidr matches the transport peer (finelog trusts its in-VPC proxy — no `X-Forwarded-For` refusal) |
| `lib/finelog/AGENTS.md` | stale "ships no auth" note fixed |
| `mkdocs.yml` | nav entry for `authed-service.md` (that runbook is not yet written) |

## 5. Out of scope (this spec = Phase 1)

- **The Phase-2 Rust *verify* engine** (design §Phasing, Open Question 1) — this
  spec pins the schema + conformance vectors + secrets + onboarding. If Phase 2 is
  built (gated on the vectors showing semantic drift), it generalizes finelog's
  `lib/finelog/rust/src/server/auth.rs` (cidr + EdDSA) with **ES256 IAP-assertion**
  verify (`jsonwebtoken` + a JWKS fetch/cache; the controller verifies no other
  non-EdDSA token now that users mint nothing), exposed to rigging via the existing
  pyo3/abi3 wheel pattern
  (`lib/finelog/rust/pyext/`); the engine returns `VerifiedIdentity{email,
  matched_layer}` and Python assigns the role. It is **verify-only** — client-side
  token *minting* stays Python `google-auth` (research §7.3). A separate design.
- **Biscuit tokens** — with §0 asymmetric JWTs as the standard core, Biscuit is an
  *optional* enhancement for **offline attenuation** of the scoped `/proxy`
  capability tokens (#6857) only, evaluated separately; it replaces the `jwt` layer,
  not the stack (design Open Questions).
- **A first-class runtime k8s Secret fetch** (`k8s-secret://`) — deliberately
  excluded (RBAC escalation, §2).
- Refresh for the dev-only `LocalCluster` session JWT (deferred in
  `20260312_iris_auth_design.md`) — a restart re-mints it. IAP edge tokens, by
  contrast, auto-refresh from the refresh token `iris login` cached, so a deployed
  user never re-runs login. (The 30-day worker token is a separate residual — see
  design "Resolved decisions".)
- Generalizing scoped tokens from a single `audience` to a `scopes` set
  (deferred in `2026-07-02_iris_per_endpoint_ingress_auth.md`).
- Unauthenticated bundle downloads (flagged in research §6; not committed here).
- The GCLB `public-proxy` / controller-redeploy ops steps of #6937 (operator
  actions, not code) — this doc only records the answered
  `MAX_ENDPOINT_TOKEN_TTL = 86400s` and the native `/proxy` BEARER pattern.
- Any change to the wire `EndpointAccess` proto or `MintEndpointToken` (shipped,
  PR #6857).
