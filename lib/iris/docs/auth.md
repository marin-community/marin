# Iris Authentication & Authorization

## Problem

Iris controllers are open by default — any client with the controller URL can submit jobs, kill jobs, register workers, and view the dashboard. In shared or production deployments this is unacceptable: we need identity verification, per-user job ownership, and worker authentication.

## Design

### Auth modes

Authentication is configured via the `auth` block in `IrisClusterConfig`. Three modes:

| Mode | Config | Behavior |
|------|--------|----------|
| **Null-auth** | No `auth` block | All requests pass as `anonymous` (admin). Workers still get an internal bearer token. |
| **Static** | `auth.static.tokens: {token: username}` | Pre-shared tokens mapped to usernames. Good for local dev and testing. |
| **GCP** | `auth.gcp.project_id: <id>` | Users log in with a GCP OAuth2 access token. The controller verifies it against Google's tokeninfo endpoint and checks project access via Cloud Resource Manager. |

### Token lifecycle

All auth modes converge on the same runtime path: **API keys stored in SQLite**.

1. On controller start, `create_controller_auth()` reads the config proto and:
   - Creates a `system:worker` user with a fresh bearer token (all modes, including null-auth).
   - For static auth: preloads config tokens into the `api_keys` table.
   - For GCP auth: instantiates a `GcpAccessTokenVerifier` as the *login verifier*.
2. On `Login` RPC (GCP mode): the controller verifies the GCP access token, creates/ensures the user, revokes old login keys, mints a new API key, and returns it.
3. All subsequent RPCs are authenticated by hashing the bearer token (SHA-256) and looking it up in `api_keys`. Expired and revoked keys are rejected.
4. `last_used_at` is throttled to one DB write per 60s per key.

### Interceptor chain

```
Request → SelectiveAuthInterceptor → Service handler
            │
            ├─ Login, GetAuthInfo  →  skip auth (unauthenticated RPCs)
            └─ everything else     →  AuthInterceptor.verify()
                                        │
                                        ├─ Authorization: Bearer <token>
                                        └─ Cookie: iris_session=<token>
```

In null-auth mode, `NullAuthInterceptor` replaces `AuthInterceptor`: tokens are verified if present (workers), but missing tokens fall through as `anonymous`.

The verified user identity is stored in a `ContextVar` (`_verified_user`) and read by service code via `get_verified_user()`.

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

- **CLI**: `iris login` exchanges a GCP access token (or picks the first static token) for an API key, stored in `~/.iris/tokens.json` keyed by cluster name.
- **Workers**: receive `auth_token` via `WorkerConfig` proto. The autoscaler passes it through from controller config.
- **Dashboard**: session cookie (`iris_session`) set via `/auth/session` POST or `?session_token=` query param redirect. The frontend shows a login page when `/auth/config` reports `auth_enabled: true` and no valid session exists.

### Schema (migration 0004)

```sql
CREATE TABLE api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,   -- SHA-256 of raw token
    key_prefix TEXT NOT NULL,        -- first 8 chars for display
    user_id TEXT NOT NULL REFERENCES users(user_id),
    name TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    last_used_at_ms INTEGER,
    expires_at_ms INTEGER,
    revoked_at_ms INTEGER
);

ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'
    CHECK (role IN ('admin', 'user', 'worker'));
```

### New RPCs

Added to `ControllerService`:

- `GetAuthInfo` — unauthenticated; returns provider name and GCP project ID.
- `Login` — unauthenticated; exchanges an identity token for an API key.
- `CreateApiKey` / `RevokeApiKey` / `ListApiKeys` — API key management.
- `GetCurrentUser` — returns the authenticated user's identity and role.

### Known limitations

- **Bundle downloads** are unauthenticated. Bundle IDs are SHA-256 hashes (256 bits of entropy) acting as capability URLs. Workers and K8s init-containers fetch bundles via stdlib `urlopen` which doesn't support auth headers.
- **No token refresh**: API keys don't auto-refresh. Login keys from GCP auth are one-shot; re-run `iris login` to get a new one.
- **Single-role model**: a user has exactly one role. No per-job or per-resource ACLs.
