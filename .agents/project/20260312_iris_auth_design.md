# Iris Authentication & Authorization

## Problem

Iris controllers are open by default — any client with the controller URL can submit jobs, kill jobs, register workers, and view the dashboard. In shared or production deployments this is unacceptable: we need identity verification, per-user job ownership, and worker authentication.

## Design

### Auth modes

Authentication is configured via the `auth` block in `IrisClusterConfig`. Three modes:

| Mode | Config | Behavior |
|------|--------|----------|
| **Null-auth** | No `auth` block | All requests pass as `anonymous` (admin). Workers still get a JWT. |
| **Static** | `auth.static.tokens: {token: username}` | Pre-shared tokens exchanged for JWTs via Login RPC. Good for local dev and testing. |
| **GCP** | `auth.gcp.project_id: <id>` | Users log in with a GCP OAuth2 access token, exchanged for a JWT via Login RPC. |

### Token lifecycle

All tokens are **JWTs signed with HMAC-SHA256**. The signing key is persisted in the `controller_secrets` table so tokens survive controller restarts.

JWT claims: `sub` (user_id), `role`, `jti` (key_id), `iat`, `exp`.

1. On controller start, `create_controller_auth()` reads the config proto and:
   - Loads (or creates) the persistent JWT signing key from `controller_secrets`.
   - Creates a `system:worker` user with a fresh worker JWT (all modes, including null-auth).
   - For static auth: preloads config tokens into `api_keys` for audit; sets up `StaticTokenVerifier` as the login verifier.
   - For GCP auth: instantiates a `GcpAccessTokenVerifier` as the login verifier.
   - Loads revoked key_ids into an in-memory revocation set.
2. On `Login` RPC: the controller verifies the identity token (GCP access token or raw static token), creates/ensures the user, revokes old login keys, mints a new JWT, and returns it.
3. All subsequent RPCs are authenticated by **verifying the JWT signature** and checking the in-memory revocation set. No database hit on the hot path.
4. The `api_keys` table is retained for audit, key management RPCs, and revocation tracking.

### Interceptor chain

```
Request → SelectiveAuthInterceptor → Service handler
            │
            ├─ Login, GetAuthInfo  →  skip auth (unauthenticated RPCs)
            └─ everything else     →  AuthInterceptor.verify()
                                        │
                                        ├─ Authorization: Bearer <JWT>
                                        └─ Cookie: iris_session=<JWT>
```

In null-auth mode, `NullAuthInterceptor` replaces `AuthInterceptor`: JWTs are verified if present (workers), but missing tokens fall through as `anonymous` admin.

The verified identity (user_id + role) is stored in a `ContextVar` (`_verified_identity`) and read by service code via `get_verified_identity()`. Role checks read directly from the JWT claims — no database lookup.

### Authorization model

Three roles: `admin`, `user`, `worker`.

| Action | Required role |
|--------|-------------|
| Register worker | `worker` |
| Submit root-level job | any authenticated user (user segment is overwritten with verified identity) |
| Submit child job / terminate job | must own the parent job hierarchy |
| Create API key for self | any authenticated |
| Create API key for another user | `admin` |
| List/revoke another user's keys | `admin` |

In null-auth mode, job ownership enforcement is skipped entirely.

### Client-side auth

- **CLI**: `iris login` exchanges an identity token (GCP access token or raw static token) for a JWT via the Login RPC, stored in `~/.iris/tokens.json` keyed by cluster name.
- **Workers**: receive `auth_token` (a JWT) via `WorkerConfig` proto. The autoscaler passes it through from controller config.
- **Dashboard**: session cookie (`iris_session`) set via `/auth/session` POST or `?session_token=` query param redirect. The frontend shows a login page when `/auth/config` reports `auth_enabled: true` and no valid session exists.

### Schema

```sql
-- migration 0004
CREATE TABLE api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,   -- "jwt:<key_id>" for JWT tokens
    key_prefix TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    name TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    last_used_at_ms INTEGER,
    expires_at_ms INTEGER,
    revoked_at_ms INTEGER
);

ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'
    CHECK (role IN ('admin', 'user', 'worker'));

-- migration 0006
CREATE TABLE controller_secrets (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);
```

### RPCs

Added to `ControllerService`:

- `GetAuthInfo` — unauthenticated; returns provider name and GCP project ID.
- `Login` — unauthenticated; exchanges an identity token for a JWT.
- `CreateApiKey` / `RevokeApiKey` / `ListApiKeys` — API key management (returns JWTs).
- `GetCurrentUser` — returns the authenticated user's identity and role (from JWT claims).

### Known limitations

- **Bundle downloads** are unauthenticated. Bundle IDs are SHA-256 hashes (256 bits of entropy) acting as capability URLs. Workers and K8s init-containers fetch bundles via stdlib `urlopen` which doesn't support auth headers.
- **No token refresh**: JWTs have a 30-day TTL by default. Re-run `iris login` to get a new one.
- **Single-role model**: a user has exactly one role. No per-job or per-resource ACLs.
- **Revocation is in-memory**: revoked JTIs are loaded from the DB at startup and updated on revocation RPCs. A controller restart reloads the full revocation set.
