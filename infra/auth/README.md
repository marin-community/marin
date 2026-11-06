# Marin auth management service

## Application adapters and local operation

Applications select their authentication backend at startup. Python services use
`rigging.shared_grants.SharedGrantClient`; Iris native ingress and Finelog share
the `marin-grants` Rust crate. Both implementations use the signed grant contract
and the same conformance fixtures. They verify pinned keys, issuer, audience,
expiry and token scope locally, then request a current decision from
`POST /v1/authorize`. The receiving service supplies the action and actual resource.
Token scope never supplies group memberships or native administrator privileges.

The shared trust configuration contains public material:

```json
{
  "issuer": "https://auth.example",
  "public_keys": ["-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"],
  "authority_url": "https://auth.example",
  "ca_bundle": "/etc/marin-auth/ca.pem",
  "timeout": 5
}
```

Outside isolated loopback tests the authority requires HTTPS. The receiving
service returns 403 for a denial and 503 when a current decision is missing or
malformed. No outage changes the
backend to native or local authentication. Decisions are not cached by these
adapters, so relationship removal applies to the next protected operation.

| Service | Configuration | Shared operations and resource boundary |
| --- | --- | --- |
| Iris | `auth.shared_grants` in cluster YAML | `job.submit`, `job.read`, `job.cancel` against `/clusters/NAME/jobs/namespace/job/` |
| Marina | `MarinaConfig.shared_grants`, or a JSON file named by `MARINA_SHARED_GRANTS_CONFIG` | `app.read`, `app.release` against `/apps/UUID/` |
| Finelog | A `shared_grants` entry in `auth` / `--auth-policy` | `logs.query` against the server's configured `/instances/NAME/` |
| Grafana alert bridge | A JSON file named by `LOOM_ALERT_AUTH_CONFIG` | Authenticate through OpenBao, obtain one bounded grant per alert, then launch a pinned Loom profile |

Iris derives `NAME` from the controller's configured cluster name, so a grant
cannot be replayed into the same job namespace on another cluster. It accepts
shared callers only on launch, individual/batch job-status and
cancellation RPCs. It denies other RPCs and endpoint proxies. Shared launches
require the sandbox container profile, which withholds native Iris task
credentials. Shared callers cannot submit a job handed off from another cluster.
Removing a grant stops
subsequent operations, but does not itself terminate an already submitted job.
An operator can still cancel that job using native authority. The initial Iris
adapter does not persist the grant ID on the job or bind task
credentials to it. Shared submission of ordinary task profiles and delegation
from a running Loom session need that lifecycle integration before rollout.

Marina resolves named routes to stable app UUIDs and filters its app directory
through the same checks. Release grants can update a specified UUID, including
its first revision, but cannot create an unbounded random app, change visibility,
archive apps, manage hosts, invoke write APIs, or use MCP. Native IAP and local
operation retain their existing permissions. Applications proxied through Marina receive the
verified identity without the incoming bearer or browser credentials.

Finelog grants cover **all data in one instance**, including SQL over multiple
clusters. They are not row or cluster filters. Shared callers cannot ingest,
register/drop tables, report relay status, or use diagnostic/admin HTTP routes.
Native replication JWTs retain their separate origin-cluster binding. For example:

```yaml
auth:
  - type: shared_grants
    instance: research-logs
    trust:
      issuer: https://auth.example
      public_keys: ["<public Ed25519 PEM>"]
      authority_url: https://auth.example
      timeout: 5
```

Grafana's shared configuration names `authority_url`, `bao_address`,
`bao_token_file`, `actor`, optional `ca_bundle`, and `grant_ttl` (300 seconds by
default). The rotating OpenBao token file needs only permission to obtain that
workload's `marin-auth` identity token. The actor must match that authenticated
principal. Alert labels and text cannot select permissions, repositories, or
launch overrides. Grant creation persists its idempotency key per authenticated
principal; retries reuse the original grant without extending its expiry, and
revoked/expired grants cannot be recreated under the same key.

Local development requires neither OpenBao nor OpenFGA. Keep the existing native
backend and bind it to loopback: Iris's local controller creates an ephemeral
signing key; Marina without IAP admits loopback; Finelog's default is loopback
only; Loom supports its native local mode. Installing shared trust selects a
different backend rather than modifying these local paths. Loom's existing
shared-mode latch also prevents removing shared trust from reopening implicit
local administration on that database.

`marin-auth` deploys OpenBao, OpenFGA, and the Marin credential service in a dedicated management
namespace. OpenFGA stores standing relationships in external PostgreSQL; the
credential service retains the runtime grant ledger. It owns no training resources. The three OpenBao replicas use separate retained
PVCs, verified TLS, pod anti-affinity, a disruption budget, and restricted
service accounts. The public ingress exposes exactly the identity discovery and
JWKS endpoints. The API remains a ClusterIP service.

This is a deployment program, not evidence that a staging service exists.
`Pulumi.staging.yaml` records only the namespace, capacity and shared KMS secrets
provider. An operator must supply the environment-specific inputs below before
preview. No service, WIF registration, or live storage exchange has been applied
or verified by adding this program.

## Deployment prerequisites

Use the repository environment with `marin-iac[deploy]` installed. State uses
the shared `gs://marin-iac-state/` backend and repository `.venv`, as described in
[Pulumi ownership](../pulumi.md).

Choose an explicit Kubernetes context for a management cluster with three
schedulable hosts, Traefik's `traefik` ingress class and
`traefik.io/v1alpha1/ServersTransport` CRD, and a CNI that enforces NetworkPolicy.
Set the actual ingress controller namespace and pod labels. Only those pods and
the explicitly selected auth broker pods can contact OpenBao on port 8200.
Raft traffic stays inside the OpenBao pod set. Egress allows DNS only to
explicit resolver host CIDRs and HTTPS to configured destination CIDRs.
Read the selected cluster's pod resolver configuration and DNS services before
setting `dns_cidrs`; each entry must be a `/32` or `/128` resolver address.
Read-only inspection of the current CoreWeave cluster found resolver service
`10.16.0.10`, upstream service `10.16.0.95`, and node-local service `10.16.0.30`,
with `node-local-dns` and `coredns` selectors. These observations are inputs to
deployment review, not portable defaults. Include the resolver addresses
actually reached by the selected cluster's pods.

Use a dedicated management cluster before enrolling live identities. Existing
Iris cluster roles can manage pods across namespaces; a namespace and
NetworkPolicy alone do not isolate signing authority from those roles. Reuse
of a training cluster requires a separately audited RBAC, admission and node
isolation design. This stack does not change Iris roles or create that design.

Select storage whose provider documents the synchronous writes, locking and
durability required by [OpenBao Integrated Storage](https://openbao.org/docs/configuration/storage/raft/).
Prefer durable block storage. The observed CoreWeave cluster currently exposes
only `shared-vast`; that observation does **not** establish that this class is
safe for Raft. The stack deliberately has no storage-class default. Certify a
suitable class before applying. Each pod receives its own `ReadWriteOnce` PVC;
PVCs remain after scale-down and deletion. Disable host swap because the
unprivileged server uses `disable_mlock=true`.

Supply a digest-pinned OpenBao image that supports the configured APIs and runs
as UID/GID 1000. Test image and storage compatibility on staging before adoption.

Provision two TLS Secrets outside Pulumi in the management namespace:

- `listener_tls_secret`: `tls.crt`, `tls.key`, and `ca.crt`. The server
  certificate must contain `openbao.NAMESPACE.svc`, the configured
  `listener_tls_server_name`, and `localhost` for verified operator
  port-forwarding. `ca.crt` must contain its issuing trust chain. The same
  server name and CA authenticate Raft joins and Traefik's upstream connection.
- `ingress_tls_secret`: a publicly trusted certificate and private key for
  `public_hostname`. Point that DNS name at the ingress. Only the two exact
  discovery paths are routed to OpenBao; `/v1/sys/*`, authentication, token
  minting and signing APIs are not publicly routed.

Pulumi stores Secret **names** only. It does not create, fetch or export their
contents. The first approved deployment creates the namespace and leaves pods
pending until the external TLS Secrets are installed there. Service, ingress
and StatefulSet readiness are deliberately not awaited during this bootstrap
phase. If the namespace already exists, import it under the OpenBao component
using the [program-first import procedure](../pulumi/README.md) before adoption.

Select the `staging` stack in `infra/auth`, then set these non-secret config
keys with `pulumi config set --stack staging KEY VALUE` or
`pulumi config set --stack staging --path 'KEY[index]' VALUE`:

| Key | Required value |
| --- | --- |
| `kubernetes_context` | Explicit management-cluster kubeconfig context |
| `image` | OpenBao image with `@sha256:` digest |
| `public_hostname` | Public discovery DNS name |
| `raft_storage_class` | Reviewed storage class suitable for Raft |
| `listener_tls_secret` | Existing server TLS/CA Secret name |
| `listener_tls_server_name` | SAN used to verify internal TLS |
| `ingress_tls_secret` | Existing public edge certificate Secret name |
| `ingress_namespace` | Existing Traefik namespace |
| `ingress_pod_labels` | Object of exact ingress controller pod labels |
| `operator_groups` | Array of Kubernetes groups allowed to port-forward |
| `dns_cidrs` | Explicit resolver host CIDRs used by OpenBao and the broker |
| `https_egress_cidrs` | Explicit array; use `[]` if no external auth fetches are needed |

The operator Role permits pod discovery and port-forwarding. It grants no
Secret access, pod execution or cluster-wide rights. The OpenBao ServiceAccount
has no RoleBinding and no mounted Kubernetes API token. Kubernetes auth using
TokenReview is therefore not configured implicitly.

OpenBao API ingress is limited to the broker pods labeled
`app.kubernetes.io/name=marin-auth` in the management namespace, the explicit
Traefik selector and OpenBao peers. Operators use the Kubernetes port-forward
API rather than a publicly routed management endpoint.

Review without applying:

```bash
cd infra/auth
pulumi preview --stack staging --diff
```

## Initialize, unseal and configure

After an explicitly approved apply, servers start sealed. Pulumi deliberately
does not await StatefulSet readiness because first initialization is an
operator ceremony. Deployment completion therefore does not imply readiness.
Port-forward a specific pod on a trusted operator machine:

```bash
kubectl --context "$MANAGEMENT_CONTEXT" -n marin-auth-staging \
  port-forward --address 127.0.0.1 pod/openbao-0 18200:8200
```

In a separate terminal, set `BAO_ADDR=https://localhost:18200` and
`BAO_CACERT` to the issuing CA file. Run `bao operator init` **once** on pod 0,
with reviewed share/threshold and PGP recipient options, and store encrypted
unseal shares and the initial root token in the approved recovery system. Do
not send them to Pulumi, logs or the repository. Run interactive
`bao operator unseal` separately for enough key holders. The other pods
retry-join using verified TLS; port-forward and unseal each in turn. Never
initialize a second pod into an independent cluster.

Once a quorum is unsealed, supply an ephemeral operator token through
`BAO_TOKEN` and run from the repository root:

```bash
uv run --no-sync infra/auth/bootstrap.py \
  --issuer "https://$AUTH_HOSTNAME/v1/identity/oidc"
```

The bootstrap command creates the non-exportable Ed25519 Transit key
`marin-grants`; the rotating RS256 identity signing key `marin-coreweave`;
identity roles `marin-auth` (audience `marin-auth`) and `marin-coreweave`
(audience `https://coreweave.com/iam`, both with a ten-minute TTL); and three
separate ACL policies:

| ACL policy | Allowed API |
| --- | --- |
| `marin-auth-broker` | Sign with `transit/sign/marin-grants`; read its public key metadata |
| `marin-auth-identity` | Mint the authenticated entity's `marin-auth` identity JWT |
| `marin-coreweave-identity` | Mint the authenticated entity's `marin-coreweave` identity JWT |

Configure trusted auth methods, entity aliases and policy assignments through a
separately reviewed OpenBao operator change. The bootstrap command does not
invent an upstream issuer, enroll arbitrary callers, or issue runtime tokens.
Bind the broker ACL to the broker workload only. Bind each storage ACL to its
own predeclared entity and keep its token separate from the broker token.
OpenBao identity tokens use the authenticated entity UUID as `sub`; the caller
cannot choose another entity's subject.

Enable a reviewed audit device before enrolling callers. Verify that audit
redaction and log routing keep bearer tokens and private material out of logs.
Revoke the initial root token after recovery and administrative access are
verified. Unset `BAO_TOKEN` when finished.

## Grant broker deployment

Enable the broker only after OpenBao is initialized and the external broker
and storage identity tokens have been installed. Set `broker_image` to the
digest-pinned image built from `infra/auth/Dockerfile` with the `marin-auth`
runtime dependencies. It runs `python -m infra.auth.service` from `/app`.
With this key set, all broker configuration is required;
there is no generated identity or in-memory ledger fallback.

| Key | Required value |
| --- | --- |
| `broker_storage_class` | Storage class suitable for a persistent SQLite ledger |
| `broker_storage_size` | Explicit ledger volume capacity |
| `broker_tls_secret` | External TLS Secret valid for `marin-auth.NAMESPACE.svc` and operator `localhost` access |
| `broker_ca_secret` | Existing Secret with `ca.crt` combining internal OpenBao/Loom roots and public roots |
| `broker_token_secrets` | Array of external Secret names, each with a `token` key |
| `broker_client_namespace` | Namespace of isolated Loom sessions or other approved clients |
| `broker_client_pod_labels` | Exact approved client pod labels |
| `broker_https_egress_cidrs` | HTTPS egress destinations for public OpenBao discovery/JWKS |
| `broker_policy` | Object with exactly the seven fields below |

`broker_policy` supplies `broker_token_file`, `subjects`, `openfga`, `storage`
and `max_token_ttl`, plus explicit `loom` (`null` or a configured Loom binding)
and `signing_key_version` (the reviewed Transit key version);
their exact schema is
[`ServiceConfig`](service.py). Token files must be
`/var/run/marin-auth/SECRET-NAME/token`, matching `broker_token_secrets`. The
signing, storage identity and Loom observer tokens must use pairwise distinct
files and Secrets.
When `loom` is enabled, its `observer_token_file` must also correspond to a
listed external Secret mount containing a native Loom operator token. The
observer rejects OpenBao credentials before making a request to Loom.
`subjects` maps authenticated OpenBao entity UUIDs to stable principals.
`openfga` contains `api_url`, `store_id`, `model_id`, `token_file`, and trusted
`principals` (`principal`, `owner`, `max_ttl`, `delegation_depth`). The stack
sets `api_url` to its TLS Check gateway. The store and immutable model IDs must
come from operator publication, not a caller token. Mount a separate OpenFGA
API token alongside the other external token files. Standing group membership
and action/resource relationships live in OpenFGA, not in the broker ConfigMap.
Each storage binding still declares its complete native permissions and separate
OpenBao identity token; permission to issue that identity is checked in OpenFGA.

The auth stack exports `coreweave_identities`, keyed by storage binding name,
with only `subject` and `permissions`. It excludes token-file references and
roles. The bucket stack selects named entries from this output, so its standing
provider policies and the runtime's capability checks use the same declaration.

Pulumi manages the remaining `ServiceConfig` fields: the actual private
OpenBao service address and public issuer, `ca_bundle` pointing to a CA-only
projection from `broker_ca_secret`, the retained database path
`/var/lib/marin-auth/grants.db`, and issuer
`https://marin-auth.NAMESPACE.svc:8090`. It publishes this policy in a ConfigMap
and mounts external Secret files without reading their values. Runtime grant
creation, attenuation and revocation update only SQLite. They cannot alter the
ConfigMap, OpenBao ACLs or provider policies.

The combined CA bundle must verify the private OpenBao listener, public OpenBao
issuer/JWKS endpoint and private Loom TLS proxy. A listener-only private CA
cannot verify the public issuer certificate. The broker uses this same explicit
bundle for observer requests to Loom; no TLS verification is disabled.

The broker is a restricted, single-replica StatefulSet. Its PVC and ledger
survive a process restart; do not scale it above one or share this SQLite file
across broker processes. Configuration changes update a pod-template hash and
trigger a singleton `RollingUpdate`: Kubernetes stops the old process before
starting its replacement on the same PVC. Changing the pinned OpenFGA model ID,
provider mappings, or principal limits changes the revision. Standing relationship
changes are checked live and do not themselves change the revision. A rollout
with a new bundle invalidates **all existing grants**. The lifecycle reconciler
permanently revokes every tracked run's old grant and archives its Loom session,
even when that run's permissions would still be allowed. Rolling the bundle
back does not restore those revoked grants or archived sessions. Plan every
bundle update as a session-draining rollout: finish running work beforehand,
then obtain fresh grants under the new revision before launching replacement
sessions. A preview alone changes no policy. Rotate
external tokens before their expiry; the runtime reads
token files on demand. The namespace has no Docker socket, host mounts,
privileged workloads or general-purpose session runner.

Only approved client pods can reach the HTTPS broker service on port 8090.
There is no public broker ingress. An explicitly approved staging session
deployment or an operator port-forward is required for an end-to-end run.

From outside the cluster, fetch
`https://HOST/v1/identity/oidc/.well-known/openid-configuration`, check that its
issuer matches the stack output exactly, and fetch its `jwks_uri`. Confirm a
public request to `/v1/sys/health` receives no OpenBao route. Inspect readiness,
`bao operator raft list-peers` and each PVC before treating the service as ready.

## CoreWeave standing policies

[CoreWeave WIF](https://docs.coreweave.com/products/storage/object-storage/auth-access/workload-identity-federation/use-oidc-tokens)
requires a registered issuer/audience pair and publicly reachable discovery and
JWKS endpoints. Register the stack's `issuer` and audience
`https://coreweave.com/iam` in the CoreWeave Console's Workload Federation page.
Record the returned Config ID in the deployment review. Installed
`pulumi-coreweave` 1.0.2 exposes organization policies and bucket policies, but
has no WIF/OIDC registration resource. This program does not claim to register
an issuer or export a fabricated registration ID.

The existing `infra/buckets` project owns provider storage authorization.
Declare each complete mapping once under `broker_policy.storage` in the auth
stack, for example:

```yaml
- name: checkpoint-reader
  role: marin-coreweave
  subject: ENTITY-UUID-FROM-OPENBAO
  token_file: /var/run/marin-auth/checkpoint-reader/token
  permissions:
    - service: cw_s3
      actions: ["s3:GetObject"]
      resource: s3://YOUR-BUCKET/YOUR-PREFIX/
```

Replace the example subject and resource with audited values, and include the
external token Secret in `broker_token_secrets`. Each subject has one complete
mapping. Select the mapping in the bucket stack by name:

```yaml
marin-buckets:auth_stack: AUTH-STACK-REFERENCE
marin-buckets:auth_identity_names:
  - checkpoint-reader
```

Pulumi reads the issuer and selected `coreweave_identities` entries through
`StackReference`. Resource names and count come from the explicit selected
names; their subject and permissions come only from the auth output. The
organization policy allows that exact `role/ISSUER:SUBJECT` to exchange
identity JWTs and access only the declared S3 resources. Existing bucket
policies keep their current ownership and deletion protections.

Object actions become short-form `bucket/prefix/*` resources. Bucket actions
such as `s3:ListBucket` are supported only for a grant covering the bucket root,
because this provider's organization-policy resource has no prefix-condition
field. Prefix-scoped list access fails configuration rather than widening to a
bucket-wide permission. No wildcard actions, all-principal statements or global
S3 resource grants are generated.

Audit **all** organization policies for the chosen role before adoption. An
unrelated wildcard policy can grant additional rights; creating a scoped
policy does not revoke those other policies. The provider mapping must match
the entire standing authority of its OpenBao subject. Runtime grants do not
change organization or bucket policies. The runtime must refuse to exchange a
provider identity unless its complete mapping is covered by the active grant.

CoreWeave credentials have their own provider expiration; neither deleting a
runtime grant nor revoking an OpenBao login token retracts already exchanged
credentials. Record and enforce that expiration at the runtime boundary. A
successful local grant test is not proof of provider-side revocation, isolation
or a live WIF exchange.

## Shared grants and Loom

Standing authorization lives in OpenFGA. The native [model](policy/model.json)
defines nested groups and action-scoped resource ancestry. The reviewed
[relationships](policy/relationships.json) include a team nested in
`marin-research`; [principals](policy/principals.json) record ownership, lifetime,
and delegation limits. OpenBao supplies verified identity and signing. The
Marin service checks the active grant and asks OpenFGA for the standing decision;
Loom enforces its launch profile and repository locally.

For example, these relationships give a researcher transitive Iris access:

```text
principal:example-researcher member group:marin-team
group:marin-team#member member group:marin-research
group:marin-research#member accessor scope:iris/job.submit//clusters/research-cluster/jobs/analysis/
```

A scope object contains the service, action, and canonical resource path. Thus
`scope:iris/job.submit//clusters/research-cluster/jobs/analysis/` and `scope:loom/session.create//diagnostics/`
are different objects. The double slash separates the action from an absolute
resource path. The adapter's action space is `loom.session.create`,
`iris.job.submit`, `iris.job.read`, `iris.job.cancel`, `finelog.logs.query`, `finelog.logs.ingest`,
`marina.app.read`, `marina.app.release`, and `cw_s3.credentials.issue`.
Unknown actions and malformed resources deny access. The application adapter
table above lists the handlers currently protected by
shared grants. Other actions remain unavailable to shared callers.

The adapter derives resource parents from canonical server-selected paths and
supplies only those parent edges as contextual tuples. Callers cannot provide
groups, accessors, blocks, or arbitrary contextual tuples. OpenFGA resolves
standing memberships and accessors from its database. A `blocked` relationship
also excludes descendants, even if another ancestor permits access.
Each Check pins the model ID and requests `HIGHER_CONSISTENCY`, bypassing the
permission-result cache. Transport failures, redirects and malformed decisions
return HTTP 503 and cannot grant access. A temporary OpenFGA outage does not
permanently revoke a live grant; lifecycle reconciliation retries instead. No positive decision cache is added in the broker.

The production OpenFGA component uses external PostgreSQL and a TLS gateway
exposing only POST `/stores/STORE-ID/check`. The full HTTP and gRPC APIs bind to
pod loopback. Operators publish models and reconcile standing tuples through
Kubernetes port-forwarding; the broker has neither Kubernetes credentials nor
an exposed write route. OpenFGA's experimental built-in API access controls are
not required by this deployment.

A runtime grant is a bounded record approved by that standing policy. For
example, an approved bundle can contain:

```yaml
permissions:
  - service: loom
    actions: [session.create]
    resource: /diagnostics/
  - service: iris
    actions: [job.submit]
    resource: /clusters/research-cluster/jobs/analysis/
  - service: cw_s3
    actions: [s3:GetObject]
    resource: s3://example-diagnostics/incidents/
```

The trailing slash is part of a bound: `/jobs/analysis/` excludes `/jobs/analysis-other/`.
Encoded paths, traversal, wildcard actions and ambiguous paths are rejected.
The same `Permission` type describes runtime ceilings and the CoreWeave policy
compiler's input. It is a capability description, not a second standing-policy
language. Only complete approved native mappings can be issued for storage;
an unrepresentable narrower prefix is rejected. An OpenFGA decision on a prefix
node is not a proof about every descendant of that node.

Pulumi deploys OpenFGA and publishes the pinned store/model references, trusted
OpenBao principal mappings and provider mappings. The operator reconciles
reviewed standing relationships using the native OpenFGA API. `POST /v1/grants` authenticates its bearer as an OpenBao
identity and writes the approved runtime grant to SQLite. Owner comes from
trusted principal data. A child grant can reduce actions, prefixes, lifetime and
delegation depth and must still pass current policy. Live checks require the
entire ancestor chain to remain active. They query OpenFGA using the original
granting principal and enforce the current actor's grant ceiling separately;
an attribution owner does not confer that owner's permissions.

`POST /oauth/token` follows [OAuth Token Exchange](https://www.rfc-editor.org/rfc/rfc8693.html):
form fields name the subject token, token type, exact audience and grant ID.
The authority signs short-lived `at+jwt` tokens through OpenBao Transit with
mandatory integer `grant_version: 1`. Python and Rust parse the same signed
contract and run the same positive and negative fixtures.
Only `cw_s3` access tokens may refresh while unexpired, preserving their audience
and bounds. Loom and Iris require a fresh OpenBao identity exchange; an external
service receiving their bearer token cannot renew it. Workloads receive no
broker credential or signing key. Applications check
issuer, signature, audience, token type, format version and acting principal.
For an operation, the receiving handler constructs its action and resource and
calls `POST /v1/authorize` with the presented bearer. The broker checks token
bounds, live grant state, current OpenFGA permissions and revision. A successful
decision returns matching grant/actor/owner metadata; it grants no authority to
other operations. Iris, Finelog and Marina have schema entries but do not yet
enforce these grants in their existing handlers.

Enable Loom lifecycle integration with an explicit `loom` policy object
containing `base_url` and `observer_token_file`. Set `loom: null` to disable
Loom launch and runtime credential issuance.
`POST /v1/loom/launch` accepts only a Loom-audience grant token and the chosen
template, title and goal. It forwards the original shared token to native Loom.
Loom verifies that token, pins a restricted, strict, environment-cleared profile
and repository from its own configuration, and obtains runtime credentials from
`POST /v1/loom/credentials`. Credential injection is an internal launch step;
callers cannot supply environment values, repository paths or profile overrides.
Loom creates one run per grant, records its shared grant ID, and keeps bearer
credentials out of its stored request and profile. Shared-auth mode requires
credentials for loopback administration too.

Run/session IDs and launch intent live in a separate SQLite ledger on the same
retained volume. Repeated delivery uses one durable idempotency key. A five-second
reconciler checks actual Loom state and revokes completed grants. Native calls
register their intent before credential issuance, so bypassing the launch
front door does not bypass lifecycle tracking. An intent that never produces
a run becomes eligible for revocation after sixty seconds. Each reconciliation
checks local expiry, revocation and policy revision for all rows before any
network requests, so an earlier failed request cannot postpone authority already
due for retirement. An in-progress network cycle can delay the next local check.
Each ledger row is reconciled independently: a missing or malformed run cannot block
completion cleanup for later grants. Malformed or mismatched metadata revokes
the affected grant; cleanup uses only its previously verified session ID.
Revoking a grant immediately prevents further broker issuance; archiving its
Loom session and discovering a late run depend on successful reconciliation.
Unavailable requests are logged and retried, so a Loom outage can delay session
cleanup. SQLite targets a single broker instance; running replicas against
independent ledgers is unsupported.
A fatal background-task or ledger failure makes `/healthz` return 503,
allowing Kubernetes probes to restart the broker.

Storage issuance requires an OpenFGA permit on the complete configured provider
identity, such as `principal:service:grafana accessor
scope:cw_s3/credentials.issue/diagnostics` for the `diagnostics` binding. Its native permissions must also fit the live grant. OpenBao and
CoreWeave retain their own ACL and S3 enforcement. Relationship rules are not
compiled into cloud IAM; the adapter issues only predeclared, representable
provider capabilities.

Current restricted Loom profiles have no CoreWeave storage tool and cannot run
an arbitrary shell helper or attach an unreviewed MCP. The adapter establishes
the launch and credential boundary. A reviewed Loom storage tool is still needed
for normal restricted agents to read CoreWeave objects. The local scripted ACP
proof below consumes the injected credential without making a model call.

For an authorized workload that can run a command, `infra.auth.client` obtains
and rotates a private OIDC token file for the native SDK provider:

```bash
uv run --package marin-auth python -m infra.auth.client \
  --auth-url "$AUTH_URL" --binding diagnostics --organization "$CW_ORG_ID" \
  --token-file "$PRIVATE_RUNTIME_DIR/cw.jwt" -- your-command
```

The parent process reads `MARIN_GRANT_TOKEN`; the child receives the native
container-provider file configuration. Use a new private token-file path. The
fixed provider identity must fit entirely inside the grant, and its OIDC token
must expire before the grant. Issuance therefore stops before grant completion
when the remaining lifetime is shorter than the fixed OpenBao role TTL.

## Reproduce the local proof and verify staging

Build the deployable broker image from the repository root:

```bash
docker build -f infra/auth/Dockerfile -t marin-auth:staging .
docker run --rm --network none --read-only marin-auth:staging \
  python -m infra.auth.service --help
```

Publish only a reviewed image, then set its registry digest as `broker_image`.
The Docker build installs the `marin-auth` closure from the repository lock.

Run the repeatable local proof with OpenBao, OpenFGA, Loom and its sibling
Tapestry executable, plus the matching Loom checkout containing its scripted ACP fixture:

```bash
uv run --package marin-auth --group test python -m infra.auth.prove \
  --bao /path/to/bao --openfga /path/to/openfga --loom /path/to/loom --loom-source /path/to/loom-checkout
```

The proof initializes a temporary non-dev, TLS-enabled Raft server and a private
Loom home plus an authenticated temporary OpenFGA engine, then cleans up all three. It exercises actual OpenBao authentication,
Transit signing, OIDC issuance, key rotation, grant denials and revocation, plus
a real Loom session using a scripted ACP process. It makes no CoreWeave calls.
Its one-share initialization is confined to disposable test storage and is not
the staging recovery ceremony.

After registering WIF and applying reviewed provider policies, provision one
nonempty fixture inside the approved prefix and another outside it in a
dedicated **test** bucket. Supply a valid scoped `MARIN_GRANT_TOKEN` and run:

```bash
uv run --package marin-auth python -m infra.auth.verify_storage \
  --auth-url "$AUTH_URL" --binding diagnostics --organization "$CW_ORG_ID" \
  --endpoint "$CW_S3_ENDPOINT" --region "$CW_REGION" \
  --test-bucket "$TEST_BUCKET" --prefix incidents/ \
  --allowed-key incidents/allowed --denied-key outside/denied
```

The verifier uses only the native container credential provider, reads one byte
from each fixture, and requires explicit `AccessDenied` for the outside read,
a fresh sentinel write and a nonexistent sentinel delete. Authentication errors
and missing objects do not count as policy denials. An unexpectedly permitted
write attempts to remove only its own sentinel and reports the key if cleanup
is denied. No existing object is deleted. Add `--wait-up-to SECONDS` explicitly
to require the same client to read after its first provider credential expires;
the report includes observed expiration times and whether its key changed.
Retain that report with the deployment review. Until this check passes on the
deployed stack, live CoreWeave enforcement remains unverified.

## Recovery and upgrades

Keep encrypted unseal shares, CA recovery and Raft snapshots outside this stack.
Take and restore a staging snapshot before relying on this service. PVC
retention is not a backup. Protect the namespace and StatefulSet from accidental
Pulumi deletion; removing them requires a reviewed recovery or retirement plan.

OpenBao StatefulSet updates use `OnDelete`: config and image changes do not silently
restart sealed replicas. During an approved upgrade, replace and unseal one
replica at a time, verify quorum and readiness, then continue. Renew externally
managed certificates before expiry and restart or reload replicas through that
same reviewed procedure. Never disable TLS verification to bypass a failed
renewal or Raft join.

## OpenFGA publication

Provide these additional stack inputs: `openfga_image`, `openfga_proxy_image`
(both pinned by digest), `openfga_store_id`, `openfga_database_secret`
(key `uri`, a PostgreSQL URI with verified TLS), `openfga_token_secret`
(key `token`), `openfga_tls_secret` (`tls.crt`, `tls.key`), and
`openfga_database_cidrs` (only the database destinations on port 5432).
Secret values are provisioned outside Pulumi and never enter its arguments or
state. PostgreSQL needs independent backups and restoration checks.

For initial bootstrap, deploy without a broker using the reserved placeholder
`00000000000000000000000000` as `openfga_store_id` on the Check gateway. Port-forward the OpenFGA pod's loopback HTTP port 8080,
publish the model and relationships below, then configure the returned actual
store/model IDs in `openfga_store_id` and `broker_policy.openfga.store_id` /
`broker_policy.openfga.model_id`, then enable the broker. The Service exposes
only the TLS Check gateway; the operator full API remains bound to pod loopback.

```bash
uv run python -m infra.auth.fga_operator \
  --url http://127.0.0.1:18080 --token-file /private/openfga.token \
  --name marin-staging --relationships infra/auth/policy/relationships.json
```

The operator returns only store/model IDs and the model hash. For later standing
relationship updates, replace `--name` with `--store-id ID --model-id ID`.
This command authoritatively reconciles all tuples in its dedicated store;
review the complete relationship file before applying it. It does not modify
runtime grant records. Publish a new model to a new store and drain old grants
before switching model references.

## Bounded CoreWeave proof

[trial/](trial/Pulumi.yaml) runs the real OpenFGA, TLS/Raft OpenBao, broker, and
native Loom session proof as one bounded Kubernetes Job in `marin-auth-trial`
on `marin_US-WEST-04A`. It has no mounted service-account token or production
credentials, denies pod ingress and egress, requests no GPU, and uses only
scratch storage. Its OpenFGA memory datastore and single-node scratch Raft
storage are for the synthetic proof, not a production deployment.
The proof exercises transitive `marin-team` -> `marin-research` membership,
combined Loom/Iris grants, out-of-scope and blocked descendants, live group
removal against an existing JWT, credential expiry, key rotation, and Loom
completion revocation. The application proof submits and cancels a sandbox job
through native Iris ingress and persistence; its isolated backend has no workers,
so it does not execute a Kubernetes task. It also exercises UUID-scoped Marina
reads/releases with disposable PostgreSQL, whole-instance Finelog SQL over two
synthetic clusters, and a Grafana alert launch with a fake Slack HTTP endpoint.
Removing the group edge denies existing tokens at the receiving services. A failed
OpenFGA credential produces 503, recovery restores access, and native local
backends remain usable during the outage. No external messages are sent.
The scripted ACP agent obtains a native OpenBao storage identity;
CoreWeave S3/WIF enforcement still requires the separate live storage procedure.

Build the proof image with `trial/Dockerfile` and named BuildKit contexts:
`proof-tools` contains reviewed, container-compatible `bao`, `openfga`, `loom`,
matching `tapestry`, and `finelog-server` binaries; `loom-source` is the matching
Loom checkout; `proof-wheels` contains the built Iris native wheel. Use Finelog's
production Debian builder for its binary; a host binary can require a newer libc.
The image installs disposable PostgreSQL and the application dependency closure.
Run it locally without network first. Publish
and pin its digest before `pulumi preview` and `pulumi up` in `trial/`.
Set the trial's `kubernetes_context` and `image` config explicitly. Save the Job's
JSON evidence, then destroy the trial stack after inspection. The Job has a
10-minute deadline and finished pods expire after 24 hours.
