# GCP Impersonation for Iris (based on logged-in user)

## Problem

Tasks run as the worker VM's service account identity. All users' tasks share the same
GCP permissions, regardless of who submitted them. We want per-user credential scoping
via [GCP service account impersonation](https://cloud.google.com/docs/authentication/use-service-account-impersonation).

## Design

### Core Concept

When a user logs in via GCP OAuth, we know their email (from tokeninfo).
We store this as `gcp_email` on the users table. When dispatching tasks for non-admin
users, the controller sets `impersonate_service_account` on the `RunTaskRequest` to the
user's GCP email. The worker injects `CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT` into
the task container env so all GCP API calls use impersonated credentials.

**Admin bypass**: Admin users skip impersonation — tasks run with the worker's
native SA identity (full access).

**No separate SA mapping**: The impersonation target IS the user's GCP login email.
No admin-managed service account assignment is needed.

**No K8s impersonation**: K8s tasks do not get the impersonation env var injected.

### Changes

| File | Change |
|------|--------|
| `config.proto` | `enable_task_impersonation` in `GcpAuthConfig` |
| `cluster.proto` | `impersonate_service_account` on `RunTaskRequest`, `gcp_email` on `GetCurrentUserResponse` |
| `migrations/0013_user_gcp_identity.py` | `gcp_email` column on users table |
| `db.py` | `set_user_gcp_email`, `get_user_gcp_email` |
| `service.py` | Store `gcp_email` on login, return it in `get_current_user` |
| `transitions.py` | Set `impersonate_service_account = gcp_email` for non-admin users |
| `env.py` | Inject `CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT` when set |
| `task_attempt.py` | Wire `impersonate_service_account` from request to env builder |
