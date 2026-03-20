# Iris Security Fixes — Implementation Plan

Based on the audit at `lib/iris/docs/security-audit-report.md` and user review.

## Fixes to implement

| ID | Finding | Severity | Action |
|----|---------|----------|--------|
| F-1 | `/auth/session` sets cookie without verifying token | CRITICAL | Fix |
| F-2 | No auth on Starlette HTTP routes | HIGH | Fix (default-deny middleware) |
| F-5 | Full tracebacks in RPC errors | HIGH | Fix (strip in production) |
| F-6 | Raw SQL can read JWT signing key | MEDIUM | Separate auth DB via ATTACH |
| F-8 | No CSRF protection | MEDIUM | Fix |
| F-15 | No rate limiting | MEDIUM | Comment only (Cloudflare WAF) |

Not fixing: F-4 (cloudpickle, accepted risk), F-9 (session token in URL, accepted), F-11 (NullAuthInterceptor, intentional).

---

## Fix 1: Verify token before setting session cookie (F-1)

**Problem:** `dashboard.py:143-158` — `_auth_session` sets the `iris_session` cookie
from `body["token"]` without verifying it. Any string becomes a valid cookie.

**Fix:** Verify the token against `self._auth_verifier` before setting the cookie.
Skip verification only when auth is disabled (no verifier).

```python
# dashboard.py — _auth_session
async def _auth_session(self, request: Request) -> JSONResponse:
    body = await request.json()
    token = body.get("token", "").strip()
    if not token:
        return JSONResponse({"error": "token required"}, status_code=400)
    if self._auth_verifier is not None:
        try:
            self._auth_verifier.verify(token)
        except ValueError:
            return JSONResponse({"error": "invalid token"}, status_code=401)
    response = JSONResponse({"ok": True})
    response.set_cookie(
        SESSION_COOKIE, token, httponly=True, samesite="strict",
        secure=request.url.scheme == "https", path="/",
    )
    return response
```

**Files:** `lib/iris/src/iris/cluster/controller/dashboard.py:143-158`

---

## Fix 2: Default-deny auth middleware for Starlette routes (F-2)

**Problem:** `dashboard.py:91-103` — all Starlette HTTP routes have no authentication
middleware. Only the Connect RPC mount goes through the interceptor chain. New routes
added by developers will be unauthenticated by default.

**Approach:** Add a Starlette ASGI middleware that verifies the `iris_session` cookie
(or `Authorization` header) for all routes, with an explicit allowlist of public paths.
Reuse the existing `TokenVerifier` protocol and `_extract_bearer_token` logic from
`rpc/auth.py`.

The middleware should be a decorator-style `require_auth` that wraps individual route
handlers, rather than a global middleware. This avoids complexity with the RPC mount
(which has its own interceptor chain) and makes the auth requirement explicit per-route.

However, to achieve **default-deny**, we need new routes to fail closed. Two options:

1. **Global middleware + allowlist** — all routes require auth unless allowlisted.
2. **`require_auth` decorator + lint rule** — routes must be decorated.

Option 1 is safer (default-deny). The middleware skips:
- `/health`
- `/auth/config`
- `/auth/session` (POST only — token is verified by Fix 1)
- `/auth/logout` (POST only — clears cookie)
- `/static/*`
- `/iris.cluster.ControllerService/*` (has its own interceptor chain)

When auth is disabled (`_auth_provider is None`), the middleware is not installed.

```python
# dashboard.py — new middleware class

_PUBLIC_PATH_PREFIXES = (
    "/health",
    "/auth/",
    "/static/",
    "/iris.cluster.ControllerService/",
)

class _AuthMiddleware:
    """ASGI middleware enforcing session cookie auth on all non-public routes."""

    def __init__(self, app: ASGIApp, verifier: TokenVerifier):
        self._app = app
        self._verifier = verifier

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self._app(scope, receive, send)

        path = scope.get("path", "")
        if any(path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES):
            return await self._app(scope, receive, send)

        # Extract token from cookie or Authorization header
        headers = dict(scope.get("headers", []))
        token = _extract_token_from_scope(scope)
        if token is None:
            response = JSONResponse({"error": "authentication required"}, status_code=401)
            return await response(scope, receive, send)

        try:
            identity = self._verifier.verify(token)
        except ValueError:
            response = JSONResponse({"error": "invalid session"}, status_code=401)
            return await response(scope, receive, send)

        # Store identity in scope for handlers to access
        scope["auth_identity"] = identity
        return await self._app(scope, receive, send)
```

Token extraction from ASGI scope needs a helper that reads raw headers:

```python
def _extract_token_from_scope(scope: dict) -> str | None:
    headers = {k.decode(): v.decode() for k, v in scope.get("headers", [])}
    return _extract_bearer_token(headers)
```

Install in `_create_app`:

```python
def _create_app(self) -> Starlette:
    # ... build routes and app as before ...
    app = Starlette(routes=routes)
    if self._auth_verifier is not None and self._auth_provider is not None:
        app = _AuthMiddleware(app, self._auth_verifier)
    return app
```

Note: `_AuthMiddleware` wraps the Starlette app, so `self._app` must store
the wrapped version. Adjust the property to return the middleware-wrapped app.

**Files:**
- `lib/iris/src/iris/cluster/controller/dashboard.py` — add `_AuthMiddleware`, update `_create_app`
- `lib/iris/src/iris/rpc/auth.py` — export `_extract_bearer_token` (rename to `extract_bearer_token`)

---

## Fix 3: Strip tracebacks from RPC error responses (F-5)

**Problem:** `interceptors.py:39-41` calls `connect_error_with_traceback()` which
embeds full Python tracebacks in `ErrorDetails.traceback` sent to clients. This leaks
server paths, function names, and variable values.

**Approach:** Add a `debug_mode` flag to `RequestTimingInterceptor`. When `False`
(default in production), the traceback field in `ErrorDetails` is cleared before
raising. The traceback is always logged server-side regardless.

The traceback is already logged at `interceptors.py:40` via `exc_info=True`. The
client-facing error only needs the message and exception type.

```python
# errors.py — add production-safe variant

def connect_error_sanitized(
    code: Code,
    message: str,
    exc: Exception | None = None,
) -> ConnectError:
    """Create a ConnectError WITHOUT traceback details. For production use."""
    details = errors_pb2.ErrorDetails(message=message)
    details.timestamp.CopyFrom(Timestamp.now().to_proto())
    if exc is not None:
        details.exception_type = f"{type(exc).__module__}.{type(exc).__name__}"
    # No traceback field — it stays empty
    return ConnectError(code, message, details=[details])
```

```python
# interceptors.py — parameterize traceback exposure

class RequestTimingInterceptor:
    def __init__(self, include_traceback: bool = False):
        self._include_traceback = include_traceback

    def intercept_unary_sync(self, call_next, request, ctx):
        # ... existing timing logic ...
        except Exception as e:
            logger.warning("RPC %s failed after %dms: %s", method, timer.elapsed_ms(), e, exc_info=True)
            if self._include_traceback:
                raise connect_error_with_traceback(Code.INTERNAL, f"RPC {method}: {e}", exc=e) from e
            raise connect_error_sanitized(Code.INTERNAL, f"RPC {method}: {e}", exc=e) from e
```

Wire up in `dashboard.py:83`:
```python
interceptors = [RequestTimingInterceptor(include_traceback=bool(os.environ.get("IRIS_DEBUG")))]
```

This way tracebacks are opt-in via `IRIS_DEBUG=1` and never leak in production.

**Files:**
- `lib/iris/src/iris/rpc/errors.py` — add `connect_error_sanitized`
- `lib/iris/src/iris/rpc/interceptors.py` — add `include_traceback` param to `RequestTimingInterceptor`
- `lib/iris/src/iris/cluster/controller/dashboard.py:83` — pass `include_traceback` based on env var
- `lib/iris/src/iris/cluster/worker/dashboard.py` — same for worker interceptor if present

---

## Fix 4: Separate auth DB via SQLite ATTACH (F-6)

**Problem:** The `controller_secrets` table (containing the JWT signing key) and
`api_keys`/`users` tables live in the main controller DB. The raw SQL query endpoint
(`query.py:42-61`) can read them:

```sql
SELECT value FROM controller_secrets WHERE key = 'jwt_signing_key'
```

**Approach:** Move auth-related tables (`controller_secrets`, `api_keys`, `users`) to
a separate SQLite file (`auth.sqlite3`). Use `ATTACH DATABASE` to make them accessible
from the main connection under the `auth` namespace. The query browser operates on
`read_snapshot()` which uses pooled read-only connections — these connections will NOT
attach the auth DB.

### Schema changes

Auth tables move to `auth.sqlite3`:
- `controller_secrets`
- `api_keys`
- `users`

The main DB retains all operational tables (jobs, tasks, workers, etc.) and references
`auth.users` via the attached namespace for foreign key resolution.

### Connection setup

The main write connection and the pooled read connections behave differently:

```python
# In ControllerDB.__init__ or similar setup:
# Main connection — attach auth DB for full access
self._conn.execute(f"ATTACH DATABASE ? AS auth", (str(auth_db_path),))

# Read pool connections — do NOT attach auth DB
# (already the case — read_snapshot creates connections without ATTACH)
```

All auth-related SQL in `auth.py` and `db.py` updates table references:
- `controller_secrets` → `auth.controller_secrets`
- `api_keys` → `auth.api_keys`
- `users` → `auth.users`

The query browser uses `read_snapshot()` (`db.py:1078`), which pulls from the read
pool. Since read pool connections don't attach the auth DB, queries against
`controller_secrets`, `api_keys`, or `users` will fail with "no such table" — which
is exactly what we want.

### Foreign keys

Jobs, tasks, etc. reference `users.user_id`. With ATTACH, foreign keys can reference
`auth.users(user_id)`. However, SQLite does not enforce foreign keys across attached
databases. Since the FK is only for referential documentation (not enforced at runtime
in practice), this is acceptable. The alternative is to keep a `users` view in the
main DB, but that adds complexity.

A simpler approach: keep the `user_id` columns in jobs/tasks as plain TEXT with no FK.
The `users` table is only used for role lookups and display names — not for cascading
deletes.

### Migration

New migration `0012_separate_auth_db.py`:
1. Check if auth tables exist in main DB
2. If so, copy rows to the attached auth DB
3. Drop the tables from the main DB (or rename to `_old_*` for safety)

The migration needs access to the auth DB path. Pass it via a module-level variable or
environment variable set before migration runs.

### Checkpoint/restore

Currently `checkpoint.py` backs up the main DB via `db.backup_to()`. With a separate
auth DB, checkpoint must also back up `auth.sqlite3`.

```python
# checkpoint.py changes
def write_checkpoint(db: ControllerDB, remote_state_dir: str) -> ...:
    prefix = remote_state_dir.rstrip("/") + "/controller-state"
    # Back up main DB (existing)
    _backup_and_upload(db, prefix, "checkpoint")
    # Back up auth DB
    if db.auth_db_path is not None:
        _backup_and_upload_file(db.auth_db_path, prefix, "auth-checkpoint")

def download_checkpoint_to_local(remote_state_dir, local_db_path, ...) -> bool:
    # Download main DB (existing)
    ...
    # Download auth DB
    auth_path = local_db_path.with_name("auth.sqlite3")
    _download_if_exists(prefix + "/auth-latest.sqlite3", auth_path)
```

### ControllerDB changes

Add `auth_db_path` property and `attach_auth_db()` method:

```python
class ControllerDB:
    def __init__(self, db_path: Path, auth_db_path: Path | None = None):
        self._auth_db_path = auth_db_path
        # ... existing init ...
        if auth_db_path:
            self._conn.execute("ATTACH DATABASE ? AS auth", (str(auth_db_path),))

    @property
    def auth_db_path(self) -> Path | None:
        return self._auth_db_path
```

**Files:**
- `lib/iris/src/iris/cluster/controller/db.py` — add `auth_db_path`, attach in init, update `ensure_user`/`set_user_role`/`get_user_role` to use `auth.users`
- `lib/iris/src/iris/cluster/controller/auth.py` — update all SQL to use `auth.` prefix for `controller_secrets`, `api_keys`, `users`
- `lib/iris/src/iris/cluster/controller/checkpoint.py` — backup/restore auth DB separately
- `lib/iris/src/iris/cluster/controller/migrations/0012_separate_auth_db.py` — new migration
- `lib/iris/src/iris/cluster/controller/main.py` — pass auth DB path to ControllerDB
- `lib/iris/src/iris/cluster/controller/service.py` — update any direct SQL referencing auth tables
- `lib/iris/src/iris/cluster/controller/query.py` — no changes needed (read pool won't see auth tables)

### Complexity note

This is the largest change. Foreign key removal from `jobs.user_id → users.user_id`
needs careful review — ensure no CASCADE behavior is relied upon. Search for
`REFERENCES users` in migrations to identify all FKs.

---

## Fix 5: CSRF protection (F-8)

**Problem:** `dashboard.py:143-158,160-164` — `/auth/session` and `/auth/logout`
POST endpoints have no CSRF protection. The `SameSite=strict` cookie flag provides
partial protection but is conditional on HTTPS.

**Approach:** Two complementary fixes:

1. **Always set `SameSite=strict`** regardless of scheme. Currently the `samesite`
   kwarg is always `"strict"` but the `secure` flag is conditional. `SameSite=strict`
   works without `secure` — it just means the cookie is sent only with same-site
   requests. Keep `secure` conditional on HTTPS (it controls whether the cookie is
   sent over HTTP at all, which we need for SSH tunnel use).

2. **Add `Origin` / `Referer` header check** for mutating POST endpoints. This is
   simpler than a CSRF token system and works well for API-style endpoints:

```python
def _check_csrf(request: Request) -> bool:
    """Verify Origin header matches the request host for CSRF protection."""
    origin = request.headers.get("origin")
    if origin is None:
        # No Origin header — check Referer as fallback
        referer = request.headers.get("referer")
        if referer is None:
            # No origin info — reject (strict mode)
            return False
        from urllib.parse import urlparse
        origin = f"{urlparse(referer).scheme}://{urlparse(referer).netloc}"

    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    return origin == expected_origin
```

Apply to `_auth_session` and `_auth_logout`:

```python
async def _auth_session(self, request: Request) -> JSONResponse:
    if not _check_csrf(request):
        return JSONResponse({"error": "CSRF check failed"}, status_code=403)
    # ... rest of handler
```

**Files:**
- `lib/iris/src/iris/cluster/controller/dashboard.py` — add `_check_csrf`, apply to `_auth_session` and `_auth_logout`

---

## Fix 6: Rate limiting comment (F-15)

**Problem:** No rate limiting on auth endpoints.

**Action:** Add a comment referencing the Cloudflare WAF approach.

```python
# dashboard.py — near _auth_session
# Rate limiting is handled at the infrastructure layer via Cloudflare WAF rules.
# See: https://developers.cloudflare.com/waf/rate-limiting-rules/
```

**Files:** `lib/iris/src/iris/cluster/controller/dashboard.py`

---

## Implementation Tasks (parallelizable)

### Task 1: Session cookie verification + CSRF + rate-limit comment

**Scope:** Fixes F-1, F-8, F-15 — all in `dashboard.py`

**Files to modify:**
- `lib/iris/src/iris/cluster/controller/dashboard.py:143-164` — verify token before setting cookie, add CSRF check to POST handlers, add rate-limit comment

**Tests to write/update:**
- Test that `_auth_session` returns 401 for invalid tokens when auth is enabled
- Test that `_auth_session` returns 200 for valid tokens
- Test that `_auth_session` works without verification when auth is disabled
- Test CSRF rejection when Origin header mismatches
- Test CSRF pass when Origin matches
- Add tests in `lib/iris/tests/cluster/controller/test_dashboard.py` or create a new `test_dashboard_auth.py`

**Dependencies:** None (standalone)

---

### Task 2: Default-deny Starlette auth middleware

**Scope:** Fix F-2

**Files to modify:**
- `lib/iris/src/iris/cluster/controller/dashboard.py` — add `_AuthMiddleware` class, update `_create_app` to wrap app
- `lib/iris/src/iris/rpc/auth.py` — rename `_extract_bearer_token` to `extract_bearer_token` (public API)

**Tests to write:**
- Test that unauthenticated requests to `/job/{id}`, `/worker/{id}`, `/bundles/{id}.zip` return 401 when auth is enabled
- Test that `/health`, `/auth/config`, `/auth/session`, `/static/*` remain accessible without auth
- Test that auth middleware is not installed when auth is disabled
- Test that RPC routes still work (go through their own interceptor chain)
- Add tests in `lib/iris/tests/cluster/controller/test_dashboard_auth.py`

**Dependencies:** None (standalone, but coordinates with Task 1 since both modify `dashboard.py`)

**Merge note:** Tasks 1 and 2 both modify `dashboard.py`. They touch different
sections (Task 1: handler methods; Task 2: `_create_app` + new class) so merge
conflicts should be minimal, but the implementer should be aware.

---

### Task 3: Strip tracebacks from production error responses

**Scope:** Fix F-5

**Files to modify:**
- `lib/iris/src/iris/rpc/errors.py:50-74` — add `connect_error_sanitized()` function
- `lib/iris/src/iris/rpc/interceptors.py:17-41` — add `include_traceback` parameter to `RequestTimingInterceptor.__init__`, use `connect_error_sanitized` when disabled
- `lib/iris/src/iris/cluster/controller/dashboard.py:83` — pass `include_traceback` from env var
- Check `lib/iris/src/iris/cluster/worker/dashboard.py` for similar interceptor usage

**Tests to write:**
- Test that `connect_error_sanitized` produces `ErrorDetails` with empty traceback
- Test that `RequestTimingInterceptor` with `include_traceback=False` does not include traceback in error
- Test that `RequestTimingInterceptor` with `include_traceback=True` preserves existing behavior
- Add tests in `lib/iris/tests/rpc/test_errors.py` (create if needed)

**Dependencies:** None (standalone)

---

### Task 4: Separate auth DB via SQLite ATTACH

**Scope:** Fix F-6

This is the most complex task. Consider splitting further if needed.

**Files to modify:**
- `lib/iris/src/iris/cluster/controller/db.py:1000-1070` — add `auth_db_path` parameter to `ControllerDB.__init__`, ATTACH auth DB on main connection only, update `ensure_user`/`set_user_role`/`get_user_role` to prefix with `auth.`
- `lib/iris/src/iris/cluster/controller/auth.py:116-142,42-108` — update all SQL to use `auth.controller_secrets`, `auth.api_keys`, `auth.users`
- `lib/iris/src/iris/cluster/controller/migrations/0012_separate_auth_db.py` — new migration that:
  1. Creates `controller_secrets`, `api_keys`, `users` in the attached `auth` DB
  2. Copies existing rows from main DB tables to auth DB tables
  3. Drops the old tables from main DB
- `lib/iris/src/iris/cluster/controller/checkpoint.py:55-93,96-121` — backup/restore auth DB alongside main DB
- `lib/iris/src/iris/cluster/controller/main.py` — construct auth DB path, pass to ControllerDB
- `lib/iris/src/iris/cluster/controller/service.py` — update any direct references to `users` table (search for `FROM users`, `JOIN users`, `REFERENCES users`)

**Migration details:**

The migration must handle the ATTACH carefully. Since migrations run on the main
connection which will have the auth DB attached, the migration can:

```python
def migrate(conn: sqlite3.Connection) -> None:
    # Tables already exist in auth DB (created by ControllerDB init)
    # Copy data from main DB to auth DB
    conn.execute("INSERT OR IGNORE INTO auth.users SELECT * FROM users")
    conn.execute("INSERT OR IGNORE INTO auth.api_keys SELECT * FROM api_keys")
    conn.execute("INSERT OR IGNORE INTO auth.controller_secrets SELECT * FROM controller_secrets")
    # Drop from main DB
    conn.execute("DROP TABLE IF EXISTS api_keys")
    conn.execute("DROP TABLE IF EXISTS controller_secrets")
    # Keep users in main DB as a view? Or drop?
    # Drop — queries will use auth.users
    conn.execute("DROP TABLE IF EXISTS users")
```

However, `users` has foreign keys from `jobs.user_id REFERENCES users(user_id)`. Dropping
`users` would break this constraint. Options:
- SQLite doesn't enforce FKs by default (needs `PRAGMA foreign_keys = ON`), and the init
  migration uses it but runtime likely doesn't. Check if FK enforcement is on.
- Safest: leave `users` table in main DB as a non-authoritative copy, and keep the
  authoritative version in `auth.users`. OR just remove the FK constraint in a migration.

**FK analysis:** `0001_init.py:29` has `user_id TEXT NOT NULL REFERENCES users(user_id)`.
However, `ControllerDB.__init__` likely doesn't set `PRAGMA foreign_keys = ON` at
runtime (only the migration does). Verify this. If FKs are not enforced at runtime,
dropping `users` from main DB is safe. If they are, replace `users` with a view:

```sql
CREATE VIEW users AS SELECT * FROM auth.users;
```

This view would only be visible on the main connection (which has auth attached), not
on read pool connections. But read pool connections currently don't query `users`
directly — they query jobs/tasks which have `user_id` as a plain TEXT column.

**Tests to write:**
- Test that raw SQL query endpoint cannot access `controller_secrets`, `api_keys`, `users`
- Test that auth operations (login, token creation, key management) still work
- Test checkpoint/restore round-trip with separate auth DB
- Test migration from existing single-DB to split-DB
- Add tests in `lib/iris/tests/cluster/controller/test_auth_db.py` (new file) and update `lib/iris/tests/cluster/controller/test_checkpoint.py`

**Dependencies:** None technically, but this is the riskiest change. Implement and merge last.

---

## Task dependency graph

```
Task 1 (cookie verify + CSRF) ─────┐
Task 2 (auth middleware) ───────────┤
Task 3 (traceback stripping) ───────┼──> merge + integration test
Task 4 (separate auth DB) ──────────┘
```

All four tasks are independent. Tasks 1 and 2 both touch `dashboard.py` but different
sections. Merge conflicts should be straightforward text conflicts.

**Recommended merge order:** 3 → 1 → 2 → 4 (least to most risk).

---

## Risks and open questions

1. **Auth middleware + RPC mount interaction:** The `_AuthMiddleware` wraps the whole
   Starlette app including the RPC mount. The RPC mount has its own auth interceptor.
   The middleware must skip `/iris.cluster.ControllerService/*` to avoid double-auth.
   Verify that the path prefix matching works correctly with Starlette's mount routing.

2. **ATTACH and migrations:** The migration to separate auth DB needs the auth DB path
   at migration time. The current migration system (`db.py:1105-1158`) runs migrations
   via `module.migrate(conn)` with just the connection. We may need to pass the auth DB
   path as an environment variable or attach the auth DB before running migrations.

3. **Read pool and ATTACH:** Verify that read pool connections (used by `read_snapshot`)
   do NOT have the auth DB attached. This is the core security property. The current
   code creates read pool connections in `_create_read_connection` — confirm it doesn't
   call ATTACH.

4. **Foreign key enforcement:** Check whether `PRAGMA foreign_keys = ON` is set at
   runtime in `ControllerDB.__init__`. If so, dropping `users` from main DB requires
   either removing the FK or creating a view.

5. **Worker dashboard:** The worker dashboard (`worker/dashboard.py`) also uses
   `RequestTimingInterceptor`. Task 3 should update it too if it exists.

6. **Existing test coverage:** Auth-related tests exist in `tests/cluster/controller/test_api_keys.py`
   and `tests/rpc/test_auth.py`. New tests should extend these, not duplicate.
