# CoreWeave Loom staging

This independent `marin-loom-cw` project deploys native Loom for restricted
shared-grant sessions. It does not modify the existing Loom VM or Iris cluster.
No live service or grant enrollment is implied by a successful preview.

The service uses a single retained PVC for `/home/app`, SQLite, runtime CLIs and
reviewed repositories under `/home/app/workspace`. It preserves the native
image's `tini` and `loom-entrypoint`, sets `LOOM_RUNNER=local`, disables service
account tokens, drops capabilities, and uses a read-only root filesystem with
writable home and temporary volumes. It mounts no host paths or Docker socket.
The image may contain Docker and sudo binaries; the pod grants them no host
daemon or privilege escalation.

Local sessions share Loom's operating-system identity. This deployment supports
only explicitly reviewed restricted ACP automation profiles. It does not offer
host shells, general coding sessions, custom MCP processes or per-session OS
isolation. Native tool filtering is a trust boundary; the PVC contains the
server's operator credentials and must not be exposed to unrestricted agents.
Use a dedicated isolated cluster or an independently audited RBAC, admission and
node isolation design. A namespace does not defeat existing cross-namespace
Iris pod-management roles. Keep this namespace separate from OpenBao's.

## Inputs and trust

Set these non-secret `marin-loom-cw` config keys for the `staging` stack:

| Key | Value |
| --- | --- |
| `auth_stack` | Auth stack with actual `grant_api_address`, `grant_public_keys` and `namespace` outputs |
| `kubernetes_context` | Explicit CoreWeave cluster context |
| `image` | Digest-pinned native Loom image containing shared-grant verification |
| `proxy_image` | Digest-pinned Caddy image |
| `storage_class` | Reviewed durable class supporting the SQLite ledger and runtime home |
| `hostname` | Public Loom DNS name pointing at Traefik |
| `ingress_tls_secret` | Existing public certificate Secret name |
| `listener_tls_secret` | Existing Caddy certificate Secret with `tls.crt`, `tls.key` and `ca.crt` |
| `listener_tls_server_name` | SAN used by Traefik to verify Caddy |
| `ca_secret` | Existing Secret with `ca.crt`, combining broker CA and public trust roots |
| `ingress_namespace`, `ingress_pod_labels` | Exact existing Traefik selector |
| `dns_cidrs` | Approved resolver `/32` or `/128` CIDRs |
| `https_egress_cidrs` | Reviewed destination CIDRs for CLI installation, model/OAuth APIs and repository access |
| `profiles` | Native reconcile profile declarations, each with explicit model and restricted policy |
| `profile_pins` | Reviewed native composition pins keyed by profile name |
| `env_secret_refs` | Optional approved provider/OAuth environment references `{ENV: {name: SECRET, key: KEY}}` |

The committed staging config has no cluster, image, host, credentials, public
keys or paid model defaults. Set actual inputs before preview. The native
entrypoint installs pinned agent CLIs into the PVC on first boot; permit their
reviewed HTTPS destinations or prepopulate a reviewed runtime home. The
unprivileged image's optional host cgroup/socket setup cannot gain privileges
here. Per-session cgroup delegation is unavailable; Kubernetes enforces the
pod's overall CPU/memory limits.

Secret contents never enter Pulumi. Install TLS, CA and optional provider/OAuth
Secrets out of band. Strict `env_clear` profiles do not automatically inherit
server model credentials; enroll the chosen runtime through its supported
credential flow before launching a paid model. No credential or model call is
performed by this stack.

The auth stack publishes audited public PEM keys through its explicit
`grant_public_keys` config/output after manual Transit initialization. The Loom
stack reads those keys and its authoritative issuer/endpoints through
`StackReference`; it does not copy addresses into config or invent keys.
`grant_api_address` currently names a Kubernetes service. It must resolve and
route from the Loom cluster with verified TLS. A separate management cluster
therefore needs an approved private DNS/routing design and a genuinely reachable
auth endpoint; this project does not provision cross-cluster connectivity.
For a shared reviewed cluster, configure the broker's permitted client namespace
and labels to match this stack's `app.kubernetes.io/name=loom-cw` pods.

Loom binds only `127.0.0.1:7878`. Caddy serves TLS on 8443 and overwrites
`X-Loom-Forwarded` so proxied requests cannot gain loopback trust. Traefik serves
the public certificate and verifies the Caddy certificate and server name via
`ServersTransport`. NetworkPolicy allows inbound traffic from the specified
Traefik pods and the exact management broker selector only in the final exposure
phase. Both sides allow the broker's observer requests to Loom on TLS port 8443.
Egress allows broker port 8090, the named resolver addresses and reviewed HTTPS
destinations. Native Loom's `auth.shared_grants.ca_file` explicitly selects
`/etc/loom-ca/ca.crt` for Rust TLS clients; environment variables alone do not
configure that verifier. This combined PEM bundle must contain the private
broker root and the public roots needed by the configured HTTPS endpoints.

## Reconcile before exposure

Pulumi publishes `/etc/loom-deployment/deployment.json`; native Loom owns the
database and applies it through `loom deployment apply`. A ConfigMap update or
pod readiness alone does not mean that database policy has been reconciled.
Use these phases in order:

1. Keep `exposure: bootstrap`. Preview and apply only after the deployment and
   external Secret inputs are approved. There is no ingress and no network
   ingress rule. The generated manifest disables loopback trust and shared
   grant authentication while declaring reviewed profiles.
2. Through a trusted operator connection inside the Loom container, run
   `loom deployment apply --file /etc/loom-deployment/deployment.json` with an
   actual operator credential. The first isolated bootstrap may use the native
   machine credential; after enrollment use a credential accepted by Loom's
   shared-deployment policy. Supply it through the normal private credential
   mechanism, never a command argument, ConfigMap, Pulumi output or log.
3. Materialize each reviewed repository under `/home/app/workspace`. Read the
   native profile composition preview and record its actual positive `revision`
   and `resolver_revision` in `profile_pins`, together with the exact `cwd`.
   A pin has shape `{cwd: /home/app/workspace/REPO, revision: INTEGER,
   resolver_revision: ACTUAL-FINGERPRINT}`. Never guess these values.
4. Set `exposure: enrolling`, preview and apply. There is still no ingress or
   inbound network access. Apply the mounted manifest through the authenticated
   native operator API. Read back effective settings and profiles. Verify
   `auth.trust_loopback=false`, the exact `auth.shared_grants` issuer, public
   keys, `/v1/authorize`, `/v1/loom/credentials` endpoints and every composition
   pin. Clear conflicting runtime overrides; Loom gives them precedence over
   deployment defaults. Verify a revoked grant fails authorization.
5. Only after that readback, set `exposure: serving`, review and apply. This
   adds the TLS ingress and its network allow rule. It leaves the enrolled
   manifest unchanged. From outside, verify the public certificate, grant-only
   access, denied shell/custom MCP operations and revocation cleanup before
   enrolling real workloads.

Every serving profile must explicitly set `class: automation`, `protocol: acp`,
`strict: true`, `env_clear: true`, `restricted: true`, an operator-selected
`model`, and `mcp_access: {mode: none, groups: []}`. Ambient allowlists and
runtime shell permissions must be empty. This project accepts no inline
profile environment or custom MCP declarations.

For later policy changes, first return to `enrolling`, removing ingress and
network access, then reconcile and verify the new manifest before restoring
`serving`. Revoking an active runtime grant is immediate through the authority;
changing these standing declarations still requires native reconciliation.

## Signing-key rotation

Create the new non-exportable Transit version without changing the broker's
pinned `signing_key_version`. Audit and publish both public PEMs in the auth
stack, update this stack's native trust declaration, reconcile and verify it,
then advance `broker_policy.signing_key_version` and complete the broker rollout.
Keep the previous public key trusted until its outstanding tokens expire.
Publishing a key or completing a pod rollout is not evidence of native Loom
database reconciliation.
