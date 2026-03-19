# Iris Security Audit Report

**Date:** 2026-03-19
**Scope:** `lib/iris/src/iris/` and `lib/iris/dashboard/`
**Threat model:** Controller exposed via cloudflared tunnel with TLS. Workers have private IPs.

## 1. Executive Summary

Iris has a **reasonable security architecture** with JWT-based auth, centralized authorization policy, and per-RPC interceptors. However, several findings require attention before public exposure:

- **CRITICAL:** The `/auth/session` endpoint accepts any string as a session cookie without verification, bypassing authentication when auth is enabled.
- **HIGH:** The `_SelectiveAuthInterceptor` uses an allowlist for unauthenticated RPCs, but all Starlette HTTP routes (dashboard pages, bundle downloads, `/auth/*`) bypass the RPC interceptor entirely — they have no authentication middleware.
- **HIGH:** `cloudpickle.loads()` deserializes untrusted data in the actor system, enabling arbitrary code execution.
- **HIGH:** Full Python tracebacks are sent to RPC clients in error responses, leaking internal paths and code structure.
- **MEDIUM:** Raw SQL query endpoint allows admin-role users to read all database contents including `controller_secrets` (JWT signing key).

The system is **not default-deny**. New Starlette HTTP routes are unauthenticated by default; only Connect RPC routes go through the interceptor chain.

## 2. Architecture Overview

### Authentication Flow

1. **Auth providers:** GCP OAuth2 (`GcpAccessTokenVerifier`), static tokens (`StaticTokenVerifier`), or null-auth (no provider configured).
2. **Login exchange:** Client sends identity token → `Login` RPC verifies via `login_verifier` → returns HMAC-SHA256 JWT.
3. **JWT verification:** All subsequent RPCs carry the JWT as `Authorization: Bearer <token>` or `iris_session` cookie. The `AuthInterceptor` verifies the JWT signature, checks expiry and in-memory revocation set, then stores `VerifiedIdentity` in a `ContextVar`.
4. **Authorization:** Service handlers call `require_identity()`, `authorize(action)`, or `authorize_resource_owner(owner)` from `iris/rpc/auth.py`. Policy is centralized in the `POLICY` dict.
5. **Null-auth mode:** When no auth provider is configured, `NullAuthInterceptor` sets identity to `anonymous:admin` for all requests.

### Route Architecture

- **Connect RPC routes** (`/iris.cluster.ControllerService/*`): Served via WSGI middleware with interceptor chain (`_SelectiveAuthInterceptor` → `RequestTimingInterceptor`).
- **Starlette HTTP routes** (`/`, `/health`, `/auth/*`, `/bundles/*`, `/job/*`, `/worker/*`): Direct Python handlers on the Starlette app. **No auth middleware.**
- **Worker dashboard**: Completely unauthenticated (no interceptor chain). Expected to be on private network.

## 3. Findings

### F-1: `/auth/session` Sets Cookie Without Token Verification

**Severity:** CRITICAL

**Description:** The `_auth_session` handler at `lib/iris/src/iris/cluster/controller/dashboard.py:143-158` accepts a POST with `{"token": "..."}` and sets it as the `iris_session` cookie without verifying the token against the auth verifier. An attacker can set any arbitrary string as the session cookie.

The cookie is then sent with subsequent requests and extracted by `_extract_bearer_token()` in `iris/rpc/auth.py:52-58`. The JWT verifier will reject invalid tokens at the RPC layer, so this doesn't directly bypass RPC auth — but it means the cookie-setting endpoint has no server-side validation, and the session cookie will be set regardless.

However, in **null-auth mode with a JWT verifier** (the default when no auth provider is configured, `controller/auth.py:270-281`), `NullAuthInterceptor` at `rpc/auth.py:275-290` catches `ValueError` from invalid tokens and silently falls through to the anonymous admin identity. This means: any token (even garbage) → cookie set → RPC calls use cookie → `NullAuthInterceptor` catches verify failure → falls through as anonymous admin.

**Affected files:**
- `lib/iris/src/iris/cluster/controller/dashboard.py:143-158`
- `lib/iris/src/iris/rpc/auth.py:275-290` (NullAuthInterceptor swallows verify errors)

**Recommended fix:** Verify the token before setting the cookie:
```python
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
    response.set_cookie(...)
    return response
```

---

### F-2: No Authentication on Starlette HTTP Routes

**Severity:** HIGH

**Description:** The Starlette app routes (`/`, `/auth/config`, `/auth/session`, `/auth/logout`, `/job/*`, `/worker/*`, `/bundles/*.zip`, `/health`) are defined directly on the Starlette app without any authentication middleware. Only the Connect RPC mount (`/iris.cluster.ControllerService/*`) passes through the interceptor chain.

This means:
- Dashboard HTML pages are served without auth (acceptable — they're static shells)
- `/auth/config` leaks whether auth is enabled, the provider type, and provider kind (LOW risk)
- `/bundles/{bundle_id}.zip` serves bundle ZIP files without authentication (see F-3)
- `/auth/session` sets cookies without verification (see F-1)

The Starlette app has no global auth middleware. If a developer adds a new route that returns sensitive data, it will be unauthenticated by default.

**Affected files:**
- `lib/iris/src/iris/cluster/controller/dashboard.py:91-103` (route definitions)

**Recommended fix:** Add a Starlette middleware that verifies the session cookie for all routes except explicit allowlist (`/health`, `/auth/config`, `/auth/session`, `/auth/logout`, `/static/*`, `/iris.cluster.ControllerService/*`). Or, add `Depends`-style auth to each route.

---

### F-3: Bundle Download Endpoint Lacks Authentication

**Severity:** MEDIUM

**Description:** The `/bundles/{bundle_id}.zip` endpoint at `dashboard.py:170-180` serves bundle ZIP files without authentication. The comment at line 172 acknowledges this: "TODO(#3291): Add bearer token auth once Kubernetes init-containers support Authorization headers."

Bundle IDs are SHA-256 hashes of the content (256 bits of entropy), serving as capability URLs. This is defense-in-depth — an attacker would need to guess or obtain a bundle ID. However:
- Bundle IDs may appear in logs, error messages, or the dashboard UI
- If an attacker can observe any job submission, they learn the bundle ID
- Bundles contain user source code and potentially secrets in `workdir_files`

**Affected files:**
- `lib/iris/src/iris/cluster/controller/dashboard.py:170-180`
- `lib/iris/src/iris/cluster/bundle.py:28-30` (bundle_id = SHA-256 of content)

**Recommended fix:** Add bearer token verification to the bundle endpoint, or implement a time-limited signed URL scheme.

---

### F-4: Cloudpickle Deserialization of Untrusted Data

**Severity:** HIGH

**Description:** The actor system uses `cloudpickle.loads()` to deserialize method arguments received over the network:
- `lib/iris/src/iris/actor/server.py:230-231`: `cloudpickle.loads(request.serialized_args)`, `cloudpickle.loads(request.serialized_kwargs)`
- `lib/iris/src/iris/client/worker_pool.py:132-134`: Same pattern in `_PickleExecutor`

Pickle deserialization can execute arbitrary code. If an attacker can send crafted protobuf messages to the actor service, they achieve remote code execution on the worker.

The actor service is hosted on worker nodes (private IPs), so the attack surface is limited to:
1. A compromised controller sending malicious heartbeats with `tasks_to_run`
2. A compromised peer worker
3. Network-level attacks on the worker network

The worker→controller communication is authenticated via JWT, but the controller→worker heartbeat flow (`WorkerService/Heartbeat`) is received by the worker and contains task definitions that may include pickled data.

**Affected files:**
- `lib/iris/src/iris/actor/server.py:230-231`
- `lib/iris/src/iris/client/worker_pool.py:132-134`
- `lib/iris/src/iris/cluster/types.py:502,754` (pickle deserialization of task callables)

**Recommended fix:** This is an accepted risk in most cluster computing systems (Ray, Dask, etc. all use pickle). Document the trust model: workers must trust the controller, and the controller must trust authenticated job submitters. Consider adding HMAC signing to pickled payloads.

---

### F-5: Full Tracebacks Exposed in RPC Error Responses

**Severity:** HIGH

**Description:** The `RequestTimingInterceptor` at `interceptors.py:39-41` wraps all unhandled exceptions with `connect_error_with_traceback()`, which includes the full Python traceback in the response:

```python
# errors.py:68-69
details.exception_type = f"{type(exc).__module__}.{type(exc).__name__}"
details.traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
```

The traceback is embedded in the `ErrorDetails` protobuf attached to the `ConnectError`. Any RPC client receives:
- Full file paths on the server (e.g. `/app/lib/iris/src/iris/cluster/controller/service.py`)
- Internal function names and call chains
- Variable values from exception context
- Dependency versions visible in paths

This aids attacker reconnaissance.

**Affected files:**
- `lib/iris/src/iris/rpc/interceptors.py:39-41,57-59`
- `lib/iris/src/iris/rpc/errors.py:50-74`
- `lib/iris/src/iris/rpc/errors.proto:26` (traceback field in ErrorDetails)

**Recommended fix:** In production, log the traceback server-side but return only a sanitized error message and a correlation ID to the client. The `ErrorDetails.traceback` field should be empty or contain only the error message.

---

### F-6: Raw SQL Query Endpoint — Admin Can Read JWT Signing Key

**Severity:** MEDIUM

**Description:** The `execute_raw_query` RPC at `service.py:1827-1839` allows admin-role users to execute arbitrary SELECT statements. The keyword blocklist (`query.py:20-33`) prevents mutations but allows:

```sql
SELECT value FROM controller_secrets WHERE key = 'jwt_signing_key'
```

This reveals the HMAC signing key used for all JWTs. With this key, an attacker with admin access can:
- Forge JWTs for any user, including `system:worker`
- Create tokens that bypass the revocation system (custom JTIs not in the revocation set)

The SELECT-only restriction is enforced via string prefix check (`query.py:48`) and keyword scan (`query.py:52-53`). This is bypassable with certain SQL constructs (e.g., `SELECT ... FROM (SELECT ... REPLACE ...)` — though SQLite may not support all bypass patterns). The keyword blocklist uses `\b` word boundaries which is solid.

**Affected files:**
- `lib/iris/src/iris/cluster/controller/query.py:42-61`
- `lib/iris/src/iris/cluster/controller/service.py:1827-1839`
- `lib/iris/src/iris/cluster/controller/auth.py:116-142` (signing key in controller_secrets)

**Recommended fix:** Either exclude `controller_secrets` from the query endpoint (deny access to the table) or move the signing key out of SQLite (e.g., to an environment variable or file-based secret). Additionally, consider restricting the queryable tables to an allowlist.

---

### F-7: Worker Dashboard Has No Authentication

**Severity:** LOW (workers on private network)

**Description:** The worker dashboard at `worker/dashboard.py:39-51` defines routes with no authentication interceptor. All RPC endpoints (`WorkerService/*`) and HTTP routes are fully open.

If the assumption that workers have private IPs is violated (e.g., misconfigured firewall, lateral movement), any actor on the worker network can:
- List and inspect all tasks
- Fetch task logs
- Profile running processes (including reading memory via py-spy/memray)

**Affected files:**
- `lib/iris/src/iris/cluster/worker/dashboard.py:39-51`
- `lib/iris/src/iris/cluster/worker/service.py` (all RPC methods have no auth checks)

**Recommended fix:** Add worker-to-controller token verification. The controller already sends a JWT to workers; the worker service could require this token for incoming requests from the controller's heartbeat proxy.

---

### F-8: No CSRF Protection on State-Mutating HTTP Endpoints

**Severity:** MEDIUM

**Description:** The `/auth/session` and `/auth/logout` POST endpoints have no CSRF protection. The `SameSite=strict` cookie flag provides some protection, but:
- `SameSite=strict` is not set when `request.url.scheme != "https"` (falls through to browser default)
- The `secure` flag is conditional on HTTPS (`dashboard.py:116`)

When accessed over HTTP (e.g., via SSH tunnel to localhost), the cookie has `SameSite=strict` but no `secure` flag. A site that can make requests to the tunnel endpoint could potentially set or clear the session.

**Affected files:**
- `lib/iris/src/iris/cluster/controller/dashboard.py:111-118,143-158,160-164`

**Recommended fix:** Always set `SameSite=strict`. Consider adding a CSRF token check for mutating endpoints.

---

### F-9: Session Token in URL Query Parameter

**Severity:** MEDIUM

**Description:** The dashboard handler at `dashboard.py:106-121` accepts a `session_token` query parameter in GET requests to `/`. If the token is valid, it sets a session cookie and redirects. This means:
- The JWT appears in URL history, server logs, and referrer headers
- Shared links may contain valid authentication tokens

**Affected files:**
- `lib/iris/src/iris/cluster/controller/dashboard.py:106-121`

**Recommended fix:** Remove the query parameter flow. Use only the POST `/auth/session` endpoint for setting cookies.

---

### F-10: ProxyControllerDashboard Forwards Without Authentication

**Severity:** LOW (local use only)

**Description:** `ProxyControllerDashboard` at `dashboard.py:183-265` proxies all RPC calls to an upstream controller without authentication. It forwards request bodies and content-type headers verbatim. This is designed for local use (viewing a remote controller), but:
- No auth headers are forwarded from the browser to the upstream
- No verification that the upstream URL is trusted

**Affected files:**
- `lib/iris/src/iris/cluster/controller/dashboard.py:246-258`

---

### F-11: NullAuthInterceptor Swallows Verification Failures

**Severity:** MEDIUM

**Description:** In null-auth mode (no auth provider configured), `NullAuthInterceptor` at `rpc/auth.py:278-284` attempts to verify tokens but swallows `ValueError` silently, falling through to the anonymous admin identity:

```python
try:
    identity = self._verifier.verify(token)
except ValueError:
    pass  # fall through to default identity (anonymous:admin)
```

This means:
- Revoked tokens still work in null-auth mode (revocation raises ValueError, which is caught)
- Expired tokens still work (same reason)
- Any garbage string works as a token

**Affected files:**
- `lib/iris/src/iris/rpc/auth.py:275-290`

**Recommended fix:** In null-auth mode, if a token is present and verification fails, log a warning. The current behavior is intentional for null-auth, but the silent swallowing makes it easy to miss configuration errors.

---

### F-12: Env Vars (WANDB_API_KEY, HF_TOKEN) Forwarded to Workers

**Severity:** INFO

**Description:** The CLI auto-forwards `WANDB_API_KEY` and `HF_TOKEN` from the user's environment to task containers (`cli/job.py:137`). These appear in the `LaunchJobRequest.env_vars` protobuf field, which is stored in the controller database and visible via:
- `GetJobStatus` RPC (returns the full `LaunchJobRequest` including env vars)
- Raw SQL query endpoint
- Controller logs (at DEBUG level)

**Affected files:**
- `lib/iris/src/iris/cli/job.py:137`
- `lib/iris/src/iris/cluster/controller/service.py:857-860` (returns `job.request` in response)

**Recommended fix:** Redact env vars containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD` in API responses. Store them separately from the job request proto if they need to be forwarded to workers.

---

### F-13: Subprocess Calls Use Hardcoded Commands (No Injection Risk)

**Severity:** INFO (no vulnerability)

**Description:** All `subprocess.run()` calls in `remote_exec.py` and `gcp.py` use list-form arguments (not shell=True), preventing command injection. The commands are constructed from controlled inputs (VM IDs, project IDs, zone names) that come from the cluster config or GCP API responses, not from user HTTP input.

The `shlex.quote()` usage in `entrypoint.py:33,41` properly escapes user-provided package names and extras.

**Affected files:**
- `lib/iris/src/iris/cluster/platform/remote_exec.py` (all `_build_cmd` methods)
- `lib/iris/src/iris/cluster/platform/gcp.py`
- `lib/iris/src/iris/cluster/runtime/entrypoint.py:33,41`

---

### F-14: DirectSshRemoteExec Disables Host Key Checking

**Severity:** LOW

**Description:** `DirectSshRemoteExec` at `remote_exec.py:196-216` disables SSH host key checking (`StrictHostKeyChecking=no`, `UserKnownHostsFile=/dev/null`). This enables MITM attacks on SSH connections between the controller and workers.

This is common in cloud environments where VMs are ephemeral, but in a public-facing deployment, controller→worker SSH should use known host keys.

**Affected files:**
- `lib/iris/src/iris/cluster/platform/remote_exec.py:201-205`

---

### F-15: No Rate Limiting on Authentication Endpoints

**Severity:** MEDIUM

**Description:** The `Login` RPC and `/auth/session` endpoint have no rate limiting. An attacker can brute-force:
- Static tokens (if static auth is configured)
- The `/auth/session` endpoint (no verification, so this is moot for cookie setting)

The `Login` RPC verifies tokens against the login_verifier, which for GCP involves an external API call (`oauth2.googleapis.com/tokeninfo`). High-volume login attempts could cause rate limiting on the Google API side.

**Affected files:**
- `lib/iris/src/iris/cluster/controller/service.py:1685-1731` (Login RPC)
- `lib/iris/src/iris/cluster/controller/dashboard.py:143-158` (/auth/session)

---

### F-16: Worker Registration Allows Arbitrary Worker ID and Address

**Severity:** MEDIUM

**Description:** The `register` RPC at `service.py:1060-1092` accepts an arbitrary `worker_id` and `address` from the caller. In auth mode, it checks `authorize(AuthzAction.REGISTER_WORKER)` — only the `worker` role is allowed. But the worker token is a shared secret given to all workers.

A compromised worker (or anyone with the worker JWT) can:
- Register with a spoofed worker_id, hijacking an existing worker's identity
- Register with any address, redirecting task dispatches to a malicious host
- Re-register repeatedly, causing the controller to update the address for an existing worker

**Affected files:**
- `lib/iris/src/iris/cluster/controller/service.py:1060-1092`
- `lib/iris/src/iris/cluster/controller/transitions.py` (register_or_refresh_worker)

**Recommended fix:** Bind worker identity to the JWT (e.g., embed worker_id in the JWT claims). Reject registrations where the JWT's embedded worker_id doesn't match the request.

## 4. Recommendations

### Systemic Improvements

1. **Default-deny HTTP middleware:** Add Starlette middleware that checks the session cookie for all routes except an explicit allowlist. This prevents future routes from being accidentally unauthenticated.

2. **Separate auth concerns from business logic:** The current pattern of calling `require_identity()` / `authorize()` inside each RPC handler is error-prone. A new RPC handler added without these calls is silently open (in auth mode, the interceptor sets the identity but doesn't enforce access — that's left to the handler). Consider:
   - A decorator/annotation system that declares required roles per RPC
   - Or move authorization into the interceptor with a method→required-role mapping

3. **Sanitize error responses:** Strip tracebacks from RPC errors in production. Return a request ID that correlates to the server-side log.

4. **Protect the JWT signing key:** Move it out of the queryable database. Store in an environment variable, file, or KMS. The raw SQL endpoint should not be able to read it.

5. **Rate limit auth endpoints:** Add rate limiting to `Login` and `/auth/session`, at minimum per-IP.

6. **Audit logging:** Log all authentication events (login success/failure, token creation, revocation) with enough context for incident response. Some of this exists (`logger.info` calls) but should be formalized.

7. **Worker identity binding:** Per-worker JWTs with embedded worker_id would prevent worker spoofing. Generate a unique JWT per worker at provisioning time instead of sharing a single worker token.

### Route/RPC Auth Status Summary

| Route/RPC | Auth Status | Notes |
|---|---|---|
| `GET /` | **No auth** | Static HTML shell |
| `GET /auth/config` | **No auth** | Leaks auth provider info |
| `POST /auth/session` | **No auth** | Sets cookie without verifying token |
| `POST /auth/logout` | **No auth** | Clears cookie |
| `GET /job/{id}` | **No auth** | Static HTML shell |
| `GET /worker/{id}` | **No auth** | Static HTML shell |
| `GET /bundles/{id}.zip` | **No auth** | Serves bundle ZIPs (capability URL) |
| `GET /health` | **No auth** | Returns `{"status": "ok"}` |
| `GET /static/*` | **No auth** | JS/CSS assets |
| `Login` RPC | **No auth** (allowlisted) | Exchanges identity token for JWT |
| `GetAuthInfo` RPC | **No auth** (allowlisted) | Returns auth provider config |
| All other RPCs | **JWT required** | Via `_SelectiveAuthInterceptor` |
| Worker RPCs | **No auth** | Worker expected on private network |

### Open Questions

1. Is the controller always behind cloudflared, or can it be directly exposed? The security model changes significantly.
2. Are bundle IDs ever logged or shown in UIs where non-admin users can see them?
3. Is the static auth provider intended for production use, or only testing? Static tokens in config files are a supply-chain risk.
4. What is the intended trust boundary for the worker network? If workers are compromised, can they forge arbitrary job results?
