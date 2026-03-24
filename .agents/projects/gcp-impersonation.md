# GCP Service Account Impersonation for Iris

## Problem

Tasks run as the worker VM's service account identity. All users' tasks share the same
GCP permissions, regardless of who submitted them. We want per-user credential scoping
via [GCP service account impersonation](https://cloud.google.com/docs/authentication/use-service-account-impersonation).

## Design

### Core Concept

When a user logs in via GCP OAuth, we already know their email (from tokeninfo).
We store this as `gcp_email` on the users table. When dispatching tasks for non-admin
users, the controller includes the user's GCP email in the `RunTaskRequest`. The
worker sets `GOOGLE_IMPERSONATE_SERVICE_ACCOUNT` or uses `gcloud` impersonation flags
so that all GCP API calls from the container use short-lived credentials scoped to
that user's permissions.

**Admin bypass**: Admin users skip impersonation — tasks run with the worker's
native SA identity (full access).

### GCP SA Impersonation Mechanism

Google's impersonated credentials work by having the calling identity (the worker VM's
service account) generate short-lived tokens on behalf of a target principal. Two approaches:

1. **Environment variable**: Set `CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT=<email>` —
   works for gcloud CLI and libraries that respect it.
2. **google-auth library**: Use `google.auth.impersonated_credentials.Credentials` to
   create impersonated credentials programmatically. This is the proper SDK approach.

For container tasks, option 1 is simplest: inject the env var and the gcloud/google-auth
SDK inside the container will automatically impersonate. But this only works if the
target is a *service account*. For user accounts (emails), we need a mapping from
user email → service account to impersonate.

**Chosen approach**: Add a `gcp_service_account` field to the users table. Admins can
set this per-user. When dispatching, if the user has a `gcp_service_account` and is
not an admin, inject `CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT` into the task env.

### Changes Required

#### 1. Proto Changes (`config.proto`, `cluster.proto`)

**config.proto** — Add impersonation config to `GcpAuthConfig`:
```protobuf
message GcpAuthConfig {
  string project_id = 1;
  // When true, non-admin users' tasks run with impersonated credentials
  // scoped to their gcp_service_account. The worker VM's SA must have
  // roles/iam.serviceAccountTokenCreator on each target SA.
  bool enable_task_impersonation = 2;
}
```

**cluster.proto** — Add user GCP identity fields:
```protobuf
message Worker.RunTaskRequest {
  // ... existing fields ...
  // GCP service account to impersonate for this task (empty = no impersonation)
  string impersonate_service_account = 11;
}

message GetCurrentUserResponse {
  string user_id = 1;
  string role = 2;
  string display_name = 3;
  string gcp_email = 4;              // GCP email from OAuth login
  string gcp_service_account = 5;    // SA to impersonate for tasks
}

// New RPC for admin to set user's SA
message SetUserServiceAccountRequest {
  string user_id = 1;
  string gcp_service_account = 2;    // Empty string to clear
}
message SetUserServiceAccountResponse {}
```

#### 2. DB Migration (new migration `0013_user_gcp_identity.py`)

```python
def migrate(conn):
    conn.executescript("""
        ALTER TABLE users ADD COLUMN gcp_email TEXT;
        ALTER TABLE users ADD COLUMN gcp_service_account TEXT;
    """)
```

#### 3. Controller DB Layer (`db.py`)

Add methods:
- `set_user_gcp_email(user_id, email)` — called during GCP login
- `get_user_gcp_email(user_id) -> str | None`
- `set_user_service_account(user_id, sa_email)` — admin-only
- `get_user_service_account(user_id) -> str | None`

#### 4. Login Flow (`service.py`)

In the `login()` handler, when auth provider is GCP:
- Store the verified GCP email: `db.set_user_gcp_email(username, username)`
  (username IS the email for GCP auth — see `GcpAccessTokenVerifier.verify()`)

#### 5. Task Dispatch (`transitions.py`)

In `_apply_assignments()`, when building `RunTaskRequest`:
- Look up the job's `user_id` from the jobs table
- Look up user's role and `gcp_service_account`
- If role != "admin" and `gcp_service_account` is set and config has
  `enable_task_impersonation`:
  - Set `run_request.impersonate_service_account = gcp_service_account`

#### 6. Task Environment (`task_attempt.py` + `env.py`)

In `build_common_iris_env()` or `build_iris_env()`:
- Accept optional `impersonate_service_account` parameter
- If set, add to env:
  ```python
  env["CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT"] = sa_email
  ```

In `TaskAttempt` / worker code that reads `RunTaskRequest`:
- Pass `request.impersonate_service_account` through to the env builder

#### 7. K8s Provider (`tasks.py`)

Same env var injection in the pod manifest for K8s tasks.

#### 8. Admin RPC (`service.py`)

New `set_user_service_account()` RPC handler:
- Requires admin role
- Sets `gcp_service_account` on the users table
- Returns success

Update `get_current_user()` to include `gcp_email` and `gcp_service_account`.

#### 9. Tests

- Unit test: verify env var is set when `impersonate_service_account` is in RunTaskRequest
- Unit test: verify env var is NOT set for admin users
- Unit test: verify login stores gcp_email
- Unit test: verify SetUserServiceAccount RPC requires admin
- Integration test: submit job as non-admin user with SA configured, verify task
  container env includes `CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT`

## File Change Summary

| File | Change |
|------|--------|
| `lib/iris/src/iris/rpc/config.proto` | Add `enable_task_impersonation` to `GcpAuthConfig` |
| `lib/iris/src/iris/rpc/cluster.proto` | Add `impersonate_service_account` to `RunTaskRequest`, add fields to `GetCurrentUserResponse`, add `SetUserServiceAccount` RPC |
| `lib/iris/src/iris/cluster/controller/migrations/0013_user_gcp_identity.py` | New migration: `gcp_email`, `gcp_service_account` columns |
| `lib/iris/src/iris/cluster/controller/db.py` | Add GCP identity getters/setters |
| `lib/iris/src/iris/cluster/controller/service.py` | Update login(), get_current_user(), add set_user_service_account() |
| `lib/iris/src/iris/cluster/controller/transitions.py` | Pass impersonation SA into RunTaskRequest during dispatch |
| `lib/iris/src/iris/cluster/runtime/env.py` | Accept + inject impersonation env var |
| `lib/iris/src/iris/cluster/worker/task_attempt.py` | Pass impersonation from request to env builder |
| `lib/iris/src/iris/cluster/providers/k8s/tasks.py` | Pass impersonation env var to pod manifest |
| `lib/iris/tests/test_auth.py` | Integration tests for impersonation flow |
| `lib/iris/tests/rpc/test_auth.py` | Unit tests for new auth fields |

## Decomposition

### Sub-task 1: Proto + DB changes (foundation)
- Proto changes to `config.proto` and `cluster.proto`
- New migration `0013_user_gcp_identity.py`
- DB layer methods in `db.py`
- Regenerate proto stubs

### Sub-task 2: Controller logic (depends on 1)
- Login flow update in `service.py`
- Task dispatch impersonation in `transitions.py`
- New `set_user_service_account` RPC
- Update `get_current_user` response

### Sub-task 3: Runtime env injection (depends on 1)
- `env.py` changes
- `task_attempt.py` changes
- K8s provider changes

### Sub-task 4: Tests (depends on 2, 3)
- All test files
