# IAP + GCLB ingress for the Iris controller

`marin-cluster admin iap` stands up an external HTTPS Load Balancer (GCLB)
with Identity-Aware Proxy (IAP) in front of a running Iris controller VM. GCLB
terminates TLS, IAP authenticates the caller against an IAM allowlist, and the
backend forwards plain HTTP to the controller on port `10000`. The controller
port is reachable **only** from Google's load-balancer ranges, so every request
arrives pre-authenticated by IAP.

```
client --HTTPS:443--> GCLB --(IAP gate)--> backend --HTTP:10000--> controller VM
```

One stack per cluster, fully config-driven: the project, zone, domain, resource
prefix, controller port, and discovery label all come from the cluster's
`provisioning.iap_gclb` block in `config/<cluster>.yaml` (no `--project`,
`--zone`, or `--domain` flags). The resource-name prefix (`iris-<cluster>-*`) and
the controller VM's GCE label / network tag come from that config, which the
command uses to find and firewall the VM. Every stage is an idempotent `gcloud`
create guarded by an existence probe, so the full `deploy` or any single stage is
safe to re-run.

This assumes the controller VM already exists (see `marin-cluster admin iam` for
the service-account / project-IAM bootstrap that precedes it).

## Prerequisite: two OAuth clients (one-time, by hand)

The IAP OAuth Admin API is being turned down, so the script does **not** create
OAuth clients. Create both under **APIs & Services → Credentials** in the Cloud
Console and pass their downloaded JSON secrets to `deploy`:

- **Web** client — IAP's anchor and the browser sign-in page. Add the redirect
  URI `https://iap.googleapis.com/v1/oauth/clientIds/<CLIENT_ID>:handleRedirect`.
- **Desktop** client — what the `iris` CLI drives for browser login. Its id is
  registered in the Web client's `programmaticClients`, so IAP admits the CLI's
  bearer ID token (whose `aud` is the desktop client id).

## Bootstrap

```bash
marin-cluster --cluster marin admin iap deploy \
    --web-client-secrets web.json \
    --desktop-client-secrets desktop.json \
    --member user:you@example.com
```

`deploy` runs the stages in dependency order — `address` (reserve static IP) →
`cert` (managed SSL) → `backend` (NEG → health check → backend service) → `iap`
(enable + bind clients) → `frontend` (URL map → HTTPS proxy → `:443` forwarding
rule) → `grant` (IAP allowlist) — and prints the reserved IP, the URL, and the
`auth.iap` block to paste (see below). The cluster (and thus the domain, project,
and zone) comes from the active cluster's config; pick it with `--cluster <name>`
or `marin-cluster config use <name>`. It finds the controller VM from the
configured discovery label; override its IP with `--controller-ip`. Add
`--dry-run` to trace every `gcloud` command without running it.

Two steps `deploy` does **not** do for you:

1. **DNS A record** — point the domain at the reserved static IP. The managed
   SSL cert stays `PROVISIONING` until that resolves.
2. **Firewall** — run the `firewall` stage (or pass `--with-firewall`) so the LB
   health check can reach the controller. Without it the backend stays
   `UNHEALTHY`. It is kept separate because its deny-public option is a footgun
   (see [Firewall](#firewall)).

Individual stages are subcommands, each runnable on its own and idempotent; run
`marin-cluster admin iap --help` for the full list. `status` reports what exists
(cert state, IP, JWT audience) and `teardown` deletes the LB stack (keeping the
static IP).

## Cluster config

Enable IAP on the cluster by pasting the printed block into its config and
setting `auth.iap.signed_header_audience` to the backend-service audience (also
printed by `status`):

```yaml
auth:
  iap:
    url: https://iris-marin.example.com
    oauth_client_id: <DESKTOP_CLIENT_ID>.apps.googleusercontent.com
    oauth_client_secret: <DESKTOP_CLIENT_SECRET>      # non-confidential, RFC 8252 §8.5
    audiences:
      - <DESKTOP_CLIENT_ID>.apps.googleusercontent.com
    signed_header_audience: /projects/<PROJECT_NUMBER>/global/backendServices/<BACKEND_ID>
  admin_users:
    - you@example.com
  optional: false   # tokenless calls that did NOT pass IAP are still rejected
```

A full template is in `lib/iris/config/iap-example.yaml`. The controller
verifies IAP's signed `X-Goog-IAP-JWT-Assertion` (its `aud` is
`signed_header_audience`) to map a request to an identity; leave that field
empty to disable the IAP path entirely. For how the controller turns an identity
into a role (`dashboard` read-only vs `user`/`admin` after `iris login`), see
`lib/iris/src/iris/rpc/auth.py` and `lib/iris/docs/auth-loopback-transition.md`.

## Access control

IAP admits a request only if the authenticated Google identity holds
`roles/iap.httpsResourceAccessor` on the backend service — that binding is the
allowlist. Grant principals with the `grant` stage:

```bash
marin-cluster --cluster marin admin iap grant --member group:team@example.com
```

- `user:alice@example.com` — one person
- `group:team@example.com` — **recommended**: manage access by group membership
- `domain:example.com` — a whole Workspace org

Authentication is not authorization: any Google account can sign in, but an
identity not on the allowlist is rejected by IAP with `403` before the request
reaches the controller.

## Firewall

`firewall` tags the controller VM and adds an **allow** rule so only the Google
front-end / health-check / IAP ranges (`130.211.0.0/22`, `35.191.0.0/16`) reach
the controller port. This is additive and required — the health check never
passes without it.

`firewall --deny-public` *also* adds a blanket `deny 0.0.0.0/0 → :10000`. This is
**not** additive: it sits above `default-allow-internal` and so also blocks
in-cluster traffic. On the marin cluster workers dial the controller's internal
IP on `:10000`, so a blanket deny would sever them — only enable it once internal
access is carved out (allow the RFC1918 ranges at a higher priority) or confirmed
unused. Without any deny rule the port is still not internet-reachable (no public
allow rule exists); the deny only makes that guarantee explicit.

## Verify

```bash
# Direct to the VM's external IP:10000 — connection times out (firewall drops it).
curl --connect-timeout 8 http://<CONTROLLER_EXTERNAL_IP>:10000/health

# Through the load balancer — IAP intercepts before the controller.
curl -i https://iris-marin.example.com/                 # → 302 to accounts.google.com

# The only allow rule for :10000 is the Google LB range.
gcloud compute firewall-rules list --filter='allowed.ports:10000' \
  --format='table(name, sourceRanges.list(), targetTags.list())'
```

A `302` (browser) or `401` (RPC) carrying `x-goog-iap-generated-response: true`
means IAP answered and the request never reached the controller; a timeout on the
direct `:10000` probe confirms the port is closed to the public internet.
