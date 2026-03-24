# GCP Impersonation Research for PR #4080

## Key Finding: You Cannot Impersonate User Accounts

GCP service account impersonation **only works with service accounts** as the target.
The IAM Credentials API (`generateAccessToken`) rejects regular user emails.

PR #4080 sets `CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT` to the user's GCP login
email (e.g., `alice@example.com`). **This will fail at runtime** because user
accounts are not valid impersonation targets.

## `CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT` Scope

This env var only affects `gcloud` CLI commands. It does **not** affect Python
client libraries (google-cloud-storage, etc.) that use Application Default
Credentials (ADC). Most Marin tasks use Python clients, so even with a valid SA
email the impersonation wouldn't propagate to library calls.

## Options for Entering an Impersonated Environment

### Option A: gcloud CLI only (current PR approach, limited)
```bash
export CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT=sa@project.iam.gserviceaccount.com
```
- Only gcloud commands are affected
- Target must be a service account email

### Option B: ADC with impersonation (gcloud + Python/Go/Java/Node)
```bash
gcloud auth application-default login \
  --impersonate-service-account=sa@project.iam.gserviceaccount.com
```
- Creates ADC file that Python client libraries auto-discover
- Supported in Python, Go, Java, Node.js
- Requires interactive login flow — not directly usable in containers

### Option C: Programmatic impersonation in Python
```python
from google.auth import impersonated_credentials
import google.auth

source_credentials, _ = google.auth.default()
target_credentials = impersonated_credentials.Credentials(
    source_credentials=source_credentials,
    target_principal="sa@project.iam.gserviceaccount.com",
    target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
```
- Works with all Python GCP clients
- Short-lived tokens, auto-refreshed
- Must be wired into task startup code

### Option D: Generate short-lived token, write ADC JSON
```bash
TOKEN=$(gcloud auth print-access-token \
  --impersonate-service-account=sa@project.iam.gserviceaccount.com)
```
- Write a temporary credential file, set GOOGLE_APPLICATION_CREDENTIALS
- Token expires in 1 hour — only for short tasks

## Recommended Design for Per-User Scoping

1. **Create per-user service accounts** (e.g., `alice@hai-gcp-models.iam.gserviceaccount.com`)
   with limited IAM bindings per user
2. **Map user email → service account** in the Iris users table
   (not using the email directly as impersonation target)
3. **Use programmatic impersonation** (Option C) in the task wrapper
   to generate credentials that work with Python client libraries
4. Worker VM's SA needs `roles/iam.serviceAccountTokenCreator` on each per-user SA

## Can an Admin `gcloud auth login` as Another User?

No. `gcloud auth login` requires interactive OAuth — the user must authenticate
themselves in a browser. There is no mechanism to programmatically log in as
another user from a server process.

**Domain-wide delegation** lets a service account act as a Workspace user for
Workspace APIs (Gmail, Drive), but this doesn't apply to GCP resource APIs
(Storage, Compute, etc.).

## Sources

- https://cloud.google.com/docs/authentication/use-service-account-impersonation
- https://cloud.google.com/iam/docs/service-account-impersonation
- https://cloud.google.com/docs/authentication/application-default-credentials
- https://cloud.google.com/iam/docs/best-practices-for-managing-service-account-keys
- https://cloud.google.com/sdk/docs/authorizing
