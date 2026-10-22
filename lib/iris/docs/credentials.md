# Task credentials

The credential broker gives a Kubernetes task temporary controller and S3
credentials. The controller holds the provider's minting credential. Tasks
receive an attempt-bound refresh capability in a mounted Kubernetes Secret;
the capability never enters the task environment, persisted job specification,
bundle or output archive.

This deployment path currently supports CoreWeave Kubernetes controllers.
Worker-fleet and VM controllers do not support `credential_broker`.

## Authority and lifetime

A controller access token lasts 15 minutes. Its subject is the job's verified
`submitting_user`, which children inherit from their root. A separate claim
grants access to the job owner's namespace. The token also names the job,
root, exact attempt UID and session generation. It grants task access within
that root; it does not grant the submitter's admin role. Endpoint tokens minted
by a task inherit its session and generation, so disabling or rolling over the
session also closes those capability URLs. Endpoint aliases may be absolute;
authorization follows the registered endpoint's owning task.

The Python RPC handlers and native proxy check current session state. The
native JWT signature cache does not cache the authorization decision. A retry
or reschedule gets a different attempt UID and capability. An old attempt
cannot renew after it is superseded. Jobs recorded with an anonymous or
`local_admin` submitter cannot obtain broker credentials; resubmit them through
an authenticated client during cutover.

During output finalization, the session can renew storage credentials, resolve
the log service and read cluster status. It cannot submit children, mutate jobs or proxy to
task services. This window is bounded by `task_outputs.finalization_timeout`,
recorded in the controller DB. Completing the attempt closes its session.
Disabling a session grants no finalization grace.

CoreWeave API-token exchange produces temporary S3 credentials for the API
token's **service principal**. It does not impersonate the submitting user at
CoreWeave. The broker uses the provider's returned expiry, normally 30 minutes.
The optional OIDC exchange asserts the verified submitting user and requests
1800 seconds. The broker rejects provider credentials lasting less than 30
minutes, allowing 60 seconds for exchange latency. Both modes use separately provisioned storage permissions;
neither provides dynamic per-prefix downscoping.

Disabling a session immediately blocks new controller requests and future
credential exchanges once the committed decision reaches the native listener.
Existing proxy response streams close within one second of a decision update
or token expiry. Already authorized mutations cannot be undone. S3 credentials
already released to a task remain valid until their provider expiry, even if
the task session is disabled. An exchange that races a disable is checked
again before its result is returned.
Closing an active response can leave an HTTP 200 with an incomplete body.
Clients must treat short reads or a stream ending before its application-level
completion message as failures. Bundle downloads also verify their content hash.

## Configuration

Configure persistent signing and capability keys and select at least one
storage profile. Each broker secret reference must include a distinct `env:`
fallback that names its projected file. The deploy operator resolves the source
into `iris-controller-secrets`; the controller mounts that Secret as private
files. Its ConfigMap contains only `file:` references to those paths.

```yaml
auth:
  required: true
  signing_key:
    - file:/secure/iris-signing-key.pem
    - env:IRIS_SIGNING_KEY
  trusted_cidrs: []

credential_broker:
  capability_key:
    - file:/secure/iris-task-capability-key
    - env:IRIS_TASK_CAPABILITY_KEY
  tls_secret: iris-controller-tls
  profiles:
    training:
      api_token:
        - file:/secure/coreweave-api-token
        - env:IRIS_CW_API_TOKEN
      allowed_principals: ["*@example.org"]
      endpoint: https://cwobject.com
      region: US-EAST-02A
      buckets: [training-data]

kubernetes_provider:
  controller_address: https://iris-controller-svc.iris.svc.cluster.local:10000
```

The capability key must contain at least 32 random bytes. Keep it separate
from the Ed25519 JWT signing key: ordinary signing-key rotation can retain
previous public keys while attempts keep their refresh authority. Rotating
the capability key invalidates existing refresh capabilities and requires a
coordinated task restart. Never put either key, a provider API token, or shared
S3 access keys in `defaults.task_env`.

`allowed_principals` accepts exact identities, domain patterns and `*`, using
the same matching rules as federation admission. Principal-directory
references are resolved on the controller. Profile names may contain letters,
digits, underscores and hyphens. Bucket routing selects a granted profile;
the bucket list itself is a client-routing declaration. Configure the actual
access boundary in CoreWeave storage policy.

For node-local LOTA, explicitly set `endpoint: http://cwlota.com`. This
preserves the existing storage network path. SigV4 authenticates requests,
but HTTP exposes their session-token headers and data to the network. The
controller broker and provider credential exchange always use HTTPS. Using
`https://cwobject.com` selects the public object-store endpoint and may change
throughput; it is not an automatic replacement for LOTA.

Controller checkpoint storage and node storage-health probes use the separate
system storage identity. Its credential files are kept out of task Secrets.
`iris-controller-secrets` contains controller secrets; `iris-storage-secrets` is
available only to the controller's storage path and node health service.

### OIDC profiles

Replace `api_token` with:

```yaml
oidc:
  issuer: https://credentials.example.org/iris
  audience: coreweave-storage
  organization: organization-id
  signing_key:
    - file:/secure/iris-storage-oidc.pem
    - env:IRIS_STORAGE_OIDC_KEY
```

Publish OIDC discovery and the signing key's public JWKS at that **separate
public HTTPS origin**. Register the issuer, audience and subject-based storage
roles with CoreWeave. The controller's private Service and federation hostname
are not the public issuer. Keep the issuer private key on the controller and
publish only public keys. Retain previous public keys through the lifetime of
issued credentials during rotation.

CoreWeave's [OIDC exchange API](https://docs.coreweave.com/products/storage/object-storage/reference/object-storage-api-ref/cwobject/create-access-key-from-oidc)
documents the request and expiry response. Acceptance of this deployment's
self-hosted Ed25519 issuer must be proved with a short-lived credential probe
before enabling an OIDC profile. This change does not provision organization
trust or bucket policies. API-token exchange is usable independently of OIDC
and Iris federation.

## Transport and task delivery

Pulumi provisions a cert-manager CA issuer and a controller leaf certificate.
The leaf covers the Service DNS names and loopback, so a Kubernetes
port-forward verifies against this cluster's CA. A TLS sidecar exposes the
Service port and reloads renewed certificates. Native and Python listeners
remain on the controller Pod's loopback interface. Traefik verifies its TLS
connection to the Service with the same CA. Monitor the CA's expiry and plan
CA replacement separately; cert-manager's CA issuer does not manage that
operator procedure.

The controller publishes the public CA in `iris-controller-ca`. Init, task,
log shipper and uploader containers mount the CA and their attempt's Secret.
Kubernetes collects the Secret through its Pod owner reference. The controller
has a namespaced Secret-management Role; node agents use their separate
ServiceAccount and do not receive the controller's signing or capability keys.
This Role can read every Secret in the controller namespace, including the CA
private key. Treat controller-Pod exec and namespace Secret access as privileged
operator authority. Attempt files are readable by all users inside their Pod
so custom non-root images can use them; isolation is between Pods.
Secrets left behind by a failed Pod create are collected after ten minutes.

Sandbox task containers receive neither the capability nor the credential
cache. Their trusted staging init container can authenticate bundle downloads.
For broker deployments, the sandbox egress policy permits the controller TLS
port for this staging path; the sandbox task has no ambient controller
credential. Finelog and other private destinations remain blocked.
Renewal requests are limited by the TLS sidecar to 20 per second per peer,
with a burst of 100. The public routes expose the dashboard shell, public
verification keys, auth/backend configuration and database health/checkpoint
timestamp; they expose no job contents or credentials.

Inside a regular task, Iris clients renew controller access automatically.
`IRIS_TASK_CREDENTIAL_FILE` names the mounted file. A rollover rejection triggers
one access-token renewal and retry; the access token alone cannot renew itself.

S3 clients use AWS `credential_process` profiles. Iris renders the AWS config
into the attempt Secret and mounts it before any setup or task command runs.
The mounted helper uses only
Python's standard library, so the user environment need not install Iris.
Both helper and broker refresh when credentials have 15 minutes remaining,
matching the standard SDK advisory refresh window. They cache temporary S3 keys in
memory-backed storage with private files and process locking. Filesystems
created through Rigging route each configured bucket to its profile without
freezing a key pair in the filesystem instance. Existing boto3 and s3fs clients
can refresh credentials for long jobs. Raw SDK callers using a non-default
profile must select that named AWS profile themselves.

## Operations and cutover

Broker activation requires matching controller, task and
sidecar code together, apply the namespace RBAC and certificate resources with
Pulumi, and wait for the TLS Secret before starting the broker deployment.
Remove CIDR and optional anonymous authentication and all shared task S3 keys.
Resubmit tasks that need the new identity model; no compatibility bridge is
provided for tasks dispatched without sessions. Do not disable the broker or restore
shared task keys as an incident workaround.

Authenticate an operator through existing Kubernetes Pod-exec authority:

```bash
uv run iris --cluster CLUSTER operator-login --principal OPERATOR_IDENTITY
```

This command executes inside the controller container, records a mint event
in the credential audit table, and caches a one-hour admin JWT plus the public
CA in the normal private client credential store. It does not print the token.
The supplied identity labels the audit event; Kubernetes RBAC grants the
authority. This admin token expires after one hour and can be invalidated by
signing-key rotation. Task-session controls do not revoke operator tokens.
The supplied principal is a self-asserted audit label; Kubernetes' Pod-exec
audit record identifies the operator. Credential audit events are retained for
30 days; disable blocks persist until explicitly removed.

For browser access, run `iris --cluster CLUSTER cluster dashboard` and trust
the cluster's public CA in your browser. Open the printed HTTPS URL and paste
the `token` value from your private
`~/.config/marin/credentials/CLUSTER.json` record into the existing login form.
The form installs a session cookie; it does not put the token in a URL. Refresh
operator login after one hour. The CLI does not install a browser cookie or
modify the operating system's certificate trust store.

Rotate an expired or revoked provider root credential in its configured secret
source and redeploy the controller's Secret projection. Storage refresh fails
if the provider rejects that credential; existing issued keys retain their
provider expiry. Keep root-credential monitoring separate from task-session
monitoring.

Select an exact attempt, a job subtree, or a submitting principal:

```bash
uv run iris --cluster CLUSTER task-sessions --attempt ATTEMPT_UID --action rollover
uv run iris --cluster CLUSTER task-sessions --job /OWNER/JOB --action disable
uv run iris --cluster CLUSTER task-sessions --principal PRINCIPAL --action disable
uv run iris --cluster CLUSTER task-sessions --principal PRINCIPAL --action enable
```

Rollover changes the access-token generation while preserving the task's
refresh capability. Disable also prevents renewal; a job or principal block
applies to future attempts and descendants. Enable removes that exact block
and advances matching sessions again so old access tokens do not revive.
Other applicable blocks still take precedence. These commands require admin
access, persist an audit event, and acknowledge only after publishing the
committed decision to the native verifier.

Before restoring a controller DB checkpoint, rotate the external capability
key and JWT signing key and remove previous JWT public keys. A checkpoint
may predate a disable decision. Reusing the old keys would revive that
authority. Treat restore as a cutover: existing task sessions lose controller
access and storage renewal, and already issued S3 credentials expire on their
provider schedule. Restart affected jobs under fresh sessions. An ordinary
controller restart reuses the current DB and keys and preserves blocks.
