# Spec — unified server auth & secret configuration

Contracts pinned by `design.md`. New code lives in `lib/rigging` (the shared
leaf); iris and finelog consume it. Signatures are the public surface reviewers
agree to; bodies are illustrative. Paths are under `main` at
`5a6f64cbeef5e1962ed367deb3aaf72956ddb4d1`.

## 0. Token format & signing authority (greenfield — Part 0)

Service tokens switch from HS256 to **asymmetric EdDSA (Ed25519)**. Nothing is
deployed, so this is a contained edit (mint sites `auth.py:224,248`; verify
`auth.py:262`; finelog `auth.rs`), not a migration.

- **`JwtTokenManager` holds an Ed25519 keypair.** `controller_secrets` stores the
  **private** key (still generated on the controller, INSERT-OR-IGNORE, unchanged
  storage path). Minting: `jwt.encode(payload, private_key, algorithm="EdDSA")`.
  Claims add a stable **`kid`** (in the JWT header) and **`iss`** (= cluster name)
  so verifiers can resolve the right key across issuers.
- **Public keys are published as JWKS.** New route
  `GET /.well-known/jwks.json` (`@public`) serves the controller's current +
  previous public keys (an overlap window for rotation). Signature:

```python
# iris/cluster/controller/auth.py
class JwtTokenManager:
    def public_jwks(self) -> dict: ...          # {"keys": [{"kty":"OKP","crv":"Ed25519","kid":...,"x":...}, ...]}
    def create_token(self, user_id, role, ttl_seconds) -> str: ...   # EdDSA, header kid+iss
    def verify(self, token: str) -> VerifiedIdentity: ...            # EdDSA against local public key(s)
```

- **Verifiers hold only public keys.** rigging gains a `JwksVerifier`
  (`TokenVerifier`) that verifies EdDSA/ES256/RS256 against a fetched+cached JWKS,
  keyed by `iss`→`jwksUri` and `kid` — the **same** machinery the IAP (ES256) and
  Google-ID-token (RS256) verifiers use. finelog's Rust `JwtVerifier` switches
  from `hmac` to `jsonwebtoken` EdDSA against the controller's published key
  (`iss`-resolved), so its `JwtKey.secret` (a shared HMAC secret) becomes a
  **public key** — no secret. `MIN_SECRET_BYTES` and the raw-ASCII-HMAC path go
  away.
- **Consequence for #6873:** `ClusterFinelogConfig.delegation_key` is **removed**;
  finelog's auth layer references the controller's JWKS/public key (inline-safe).
  Federation: a `/proxy` gate or a peer resolves the minting cluster's public key
  by `iss`. No shared symmetric secret anywhere on the verify path.

## 1. Declarative auth-stack schema — `lib/rigging/src/rigging/auth_config.py` (new)

### 1.1 Scope and wire format

This schema models the **request chain** only — the authenticators that decide
an already-authenticated request. It does **not** model login-exchange verifiers
(`static`/`gcp`/`iap_id_token`): those run inside the `Login` RPC (which the
policy skips via `unauthenticated_methods`) and are constructed in code
(`iris/cluster/controller/auth.py:447-470`). The request chain's verifier is
always the service JWT manager, so the schema never carries a login verifier.

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
| `jwt` | `optional: bool = false` | the service JWT `TokenVerifier` — a `JwksVerifier` (EdDSA, §0) via `jwt_verifier=` | present+valid ⇒ AUTHENTICATED; absent ⇒ ABSENT. present+invalid ⇒ REJECTED when `optional=false`, else ABSENT (the `BestEffortJwtAuthenticator` case that makes a null-auth chain attribute a valid worker JWT but never reject) |
| `iap_assertion` | — | `IapAssertionVerifier` (via `iap_assertion_verifier=`) | verifies `X-Goog-IAP-JWT-Assertion`; forged ⇒ REJECTED; absent ⇒ ABSENT |
| `cidr` | `cidrs: list[str]` | — | direct socket peer in a CIDR ⇒ `ANONYMOUS_ADMIN`; `X-Forwarded-For`/port-0 ⇒ ABSENT |
| `loopback` | — | — | genuine loopback socket peer ⇒ `ANONYMOUS_ADMIN` |
| `anonymous` | — | — | terminal: admit as `ANONYMOUS_ADMIN` (the permissive / `optional` tail) |

Rules: an **empty list raises** at parse time (total lockout — a service passes
an explicit default stack rather than relying on omission). A stack whose last
layer is not `anonymous` is default-deny (all-absent ⇒ raise ⇒ `UNAUTHENTICATED`).
`static`, `gcp`, and `iap_id_token` are **not** request-chain layers (they are
login-exchange verifiers — see §1.1). A `jwt`/`iap_assertion` layer whose
verifier was not supplied is a build-time `ValueError`.

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

| Current state | Compiled stack | Notes |
|---|---|---|
| null-auth (no provider) | `[jwt(optional=true), anonymous]` | best-effort JWT attributes workers; anonymous terminal = today's `permissive()`. **Stays open** — a null-auth dev cluster is unchanged. |
| `gcp` / `static` | `[jwt, cidr(trusted_cidrs)?, loopback] (+ anonymous if optional)` | request verifier is the JWT manager; the gcp/static login verifier stays in the `Login` RPC |
| `iap` | `[jwt, iap_assertion, cidr(trusted_cidrs)?, loopback] (+ anonymous if optional)` | same chain `enforcing()` builds today |
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

Vectors cover the divergences #6861 names (default posture, cidr-vs-jwt ordering,
`X-Forwarded-For` refusal, empty-list lockout, allow-localhost fallback). This is
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
    (env:IRIS_<FIELD>, file:/etc/iris/secrets/<field>, gcp-secret://…/iris-<field>/versions/<v>).
    Lets a service inherit a secret home without per-field config."""
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

  Marked fields (the residue after Part 0 retires `delegation_key`):
  `StaticAuthConfig.tokens` values (`config.py:506`, dev/CI static bearer);
  `IapAuthConfig.oauth_client_secret` (`config.py:512`); and `PeerConfig.static_token`
  (`config.py:661-664`) *if* the symmetric peer path survives (open question — a
  peer can instead verify via the minting cluster's JWKS). **Removed:**
  `ClusterFinelogConfig.delegation_key` — now the controller's public key, not a
  secret (§0). **Not** marked: `WorkerConfig.auth_token` (`config.py:402`) — minted
  on the controller at runtime (`local_cluster.py:207`), always empty in an
  authored config, so never a reference and never guarded.
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
    ConfigMap / GCE metadata. Empty ⇒ pass (unset; resolves via default path).
    Mirrors finelog's assert_inlineable_auth (finelog/deploy/config.py:131-142)."""
```

Per-service JWT **signing** keys are out of config entirely — minted on the
controller into `controller_secrets` (unchanged, `auth.py:157-181`).

## 3. Role-grant RPC + onboarding CLI (#6580, #6592)

### 3.1 `SetUserRole` RPC (new, on `ControllerService`)

Today there is no role-change RPC (`set_user_role` runs only at startup). Add:

```proto
message SetUserRoleRequest { string user_id = 1; string role = 2; }  // role ∈ {dashboard, user, admin}
message SetUserRoleResponse { string user_id = 1; string role = 2; }
```

Handler (`iris/cluster/controller/service.py`) is admin-only via a new
`AuthzAction.MANAGE_USER_ROLES` (`iris/rpc/auth.py`, empty allowed-set = admin
only, like `MANAGE_OTHER_KEYS`); it `ensure_user` + `set_user_role`
(`writes.py:742-745`). Rejects reserved `system:` user ids.

### 3.2 `iris user grant` CLI

```
iris user grant \
    --cluster <name> \
    --user <email|sa-email> \
    --role {dashboard|user|admin}      # default: dashboard (read-only)
    [--headless]                        # a service account: also wire keyless WIF (no SA key)

# Orchestrates three distinct planes (only 1 and 3 are applied live):
#  1. [IAM]      gcloud iap web add-iam-policy-binding … roles/iap.httpsResourceAccessor  (idempotent)
#                with --headless, also: roles/iam.workloadIdentityUser +
#                roles/iam.serviceAccountTokenCreator on the iap-caller SA (keyless WIF)
#  3. [RPC]      SetUserRole(user, role) against the live controller               (idempotent)
#  2. [CONFIG]   if the IAP client id is absent from auth.iap.audiences, PRINT the
#                required config edit + `iris cluster start` redeploy step         (NOT applied live)
#  →  prints:    iap+https://<host>/proxy/<name>?audience=<iap-client-id>
```

Recommended headless auth is **Workload Identity Federation — no downloaded SA
key** (research §7.2): CI mints an OIDC token → WIF → impersonates the `iap-caller`
SA → ID token with `id_token_audience = <IAP client id>`; GCE/GKE cron uses its
attached SA → `generateIdToken(audience=<IAP client id>)`. The custom-audience ID
token needs the impersonation path, so one keyless `iap-caller` SA stays the
impersonation target. The command is idempotent for planes 1 and 3; plane 2 is a
config edit it cannot apply to a running controller, so it prints the diff and the
redeploy command rather than pretending to mutate live state.

### 3.3 `login` provisioning change (#6592)

Scoped to **IAP clusters only**: the `login` RPC (`service.py:2584-2620`)
provisions a new IAP identity at `IapAuthConfig.unprovisioned_role` (default
read-only `dashboard`) instead of the write-capable default `"user"`.
`gcp`/`static` login provisioning is unchanged — there the login verifier is
already the allowlist (`GcpAccessTokenVerifier`'s project check,
`server_auth.py:189-196`). Claims trap (runbook): role is baked into the 30-day
JWT and `verify()` never reads the user store, so a grant applies only after the
user re-runs `iris login`.

## 4. Files

| Path | Change |
|---|---|
| `lib/rigging/src/rigging/auth_config.py` | **new** — `AuthStackConfig`, layer specs, wire (de)serialization |
| `lib/rigging/src/rigging/auth_vectors.json` | **new** — shared cross-impl conformance vectors (§1.4) |
| `lib/rigging/src/rigging/server_auth.py` | `RequestAuthPolicy.from_config`; `enforcing`/`permissive` reimplemented on it |
| `lib/rigging/src/rigging/secrets.py` | **new** — `resolve_secret_spec`, `default_secret_spec`, `SecretSource`, `is_secret_reference` |
| `lib/rigging/pyproject.toml` | add `[secrets]` optional extra (`google-cloud-secret-manager`) |
| `lib/rigging/docs/authed-service.md` | **new** — the rollout runbook (server + client recipe; cidr-grants-admin caveat; first-match note) |
| `lib/iris/src/iris/cluster/config.py` | mark `SecretRefSpec` fields; `resolve_config_secrets`; `assert_no_inlined_secrets` |
| `lib/iris/src/iris/cluster/controller/main.py` | resolve secrets on the `serve` path after `load_config` |
| `lib/iris/src/iris/cluster/platforms/k8s/controller.py` | guard at `_config_json_for_configmap` |
| `lib/iris/src/iris/cluster/platforms/gcp/controller_bootstrap.py` | guard at the render site |
| `lib/iris/src/iris/cluster/controller/auth.py` | **§0** EdDSA keypair in `JwtTokenManager` (`kid`/`iss`, `public_jwks()`); `request_auth_policy` builds `AuthStackConfig`; IAP `login` provisions at `unprovisioned_role` |
| `lib/iris/src/iris/cluster/controller/dashboard.py` | **§0** `@public GET /.well-known/jwks.json` route |
| `lib/rigging/src/rigging/server_auth.py` (verifier) | **§0** `JwksVerifier` (EdDSA/ES256/RS256, `iss`/`kid`-resolved, cached) shared by service + IAP/Google verify |
| `lib/finelog/rust/src/server/auth.rs` | **§0** jwt layer verifies EdDSA via `jsonwebtoken` against the issuer's public key (drop `hmac`/`MIN_SECRET_BYTES`); `delegation_key`→public key |
| `lib/iris/src/iris/cluster/config.py` (finelog) | **§0** remove `ClusterFinelogConfig.delegation_key`; finelog auth references controller JWKS/public key |
| `lib/iris/src/iris/cluster/controller/service.py` | `SetUserRole` handler |
| `lib/iris/src/iris/rpc/auth.py` | `AuthzAction.MANAGE_USER_ROLES` |
| `lib/iris/proto/…` | `SetUserRoleRequest` / `SetUserRoleResponse` |
| `lib/iris/src/iris/cli/…` | `iris user grant` |
| `lib/finelog/src/finelog/deploy/config.py` | share the `cidr`/walk wire convention + contract test; keep `assert_inlineable_auth` |
| `lib/finelog/AGENTS.md` | fix stale "ships no auth" note (`:29-31`) |
| `mkdocs.yml` | nav entry for `authed-service.md` |

## 5. Out of scope (this spec = Phase 1)

- **The Phase-2 Rust *verify* engine** (design §Phasing, Open Question 1) — this
  spec pins the schema + conformance vectors + secrets + onboarding. If Phase 2 is
  built (gated on the vectors showing semantic drift), it generalizes finelog's
  `lib/finelog/rust/src/server/auth.rs` (cidr + HS256) with **ES256 IAP-assertion**
  and **RS256 Google-ID-token** verify (`jsonwebtoken` + a JWKS fetch/cache),
  exposed to rigging via the existing pyo3/abi3 wheel pattern
  (`lib/finelog/rust/pyext/`); the engine returns `VerifiedIdentity{email,
  matched_layer}` and Python assigns the role. It is **verify-only** — client-side
  token *minting* stays Python `google-auth` (research §7.3). A separate design.
- **Biscuit tokens** — with §0 asymmetric JWTs as the standard core, Biscuit is an
  *optional* enhancement for **offline attenuation** of the scoped `/proxy`
  capability tokens (#6857) only, evaluated separately; it replaces the `jwt` layer,
  not the stack (design Open Questions).
- **A first-class runtime k8s Secret fetch** (`k8s-secret://`) — deliberately
  excluded (RBAC escalation, §2).
- Token refresh / rotation for the 30-day iris JWT (deferred in
  `20260312_iris_auth_design.md`).
- Generalizing scoped tokens from a single `audience` to a `scopes` set
  (deferred in `2026-07-02_iris_per_endpoint_ingress_auth.md`).
- Unauthenticated bundle downloads (flagged in research §6; not committed here).
- The GCLB `public-proxy` / controller-redeploy ops steps of #6937 (operator
  actions, not code) — this doc only records the answered
  `MAX_ENDPOINT_TOKEN_TTL = 86400s` and the native `/proxy` BEARER pattern.
- Any change to the wire `EndpointAccess` proto or `MintEndpointToken` (shipped,
  PR #6857).
