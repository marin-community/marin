# IAP + GCLB ingress for the Iris controllers

`lib/iris/scripts/iap_gclb.py` stands up an external HTTPS Load Balancer (GCLB)
with Identity-Aware Proxy (IAP) in front of the running Iris controller VMs.
GCLB terminates TLS, IAP authenticates the caller against an IAM allowlist, and
the backend forwards plain HTTP to the controller on port `10000`. The
controller port is reachable **only** from Google's load-balancer ranges, so
every request arrives pre-authenticated by IAP.

```
client --HTTPS:443--> GCLB --(IAP gate)--> backend --HTTP:10000--> controller VM
```

## Topology: one shared frontend, one backend per cluster

A single **shared frontend** carries every cluster:

- a global static IP (cluster domains' DNS A records all point here),
- a URL map that routes by `Host` header to per-cluster backends,
- an HTTPS proxy holding every cluster's managed cert,
- a `:443` forwarding rule.

The frontend is named after the cluster that first stood it up
(`SHARED_FRONTEND`, currently `marin`): its resources are `iris-marin-ip`,
`iris-marin-urlmap`, `iris-marin-https-proxy`, `iris-marin-fr`.

Each cluster contributes a **backend**: a zonal NEG to its controller VM, a
health check, an IAP-gated backend service (`iris-<cluster>-be`), a managed cert
for its domain, and a host rule in the shared URL map. The frontend-owning
cluster is the URL map's default service, so it needs no host rule; every other
cluster routes by domain:

| Host             | Backend service     | Cluster    |
| ---------------- | ------------------- | ---------- |
| `iris.oa.dev`    | `iris-marin-be`     | marin (default) |
| `iris-dev.oa.dev`| `iris-marin-dev-be` | marin-dev  |

The controller VM is found by its GCE label / network tag
(`iris-<cluster>-controller`), which the script uses to discover its IP and to
firewall the port. Every stage is an idempotent `gcloud` create guarded by an
existence probe, so the full `deploy` or any single stage is safe to re-run.

This assumes the controller VM already exists (see `setup_iam.py` for the
service-account / project-IAM bootstrap that precedes it).

## Prerequisite: two OAuth clients (one-time, by hand)

The IAP OAuth Admin API is being turned down, so the script does **not** create
OAuth clients. Create both under **APIs & Services → Credentials** in the Cloud
Console and pass their downloaded JSON secrets to `deploy`:

- **Web** client — IAP's anchor and the browser sign-in page. Add the redirect
  URI `https://iap.googleapis.com/v1/oauth/clientIds/<CLIENT_ID>:handleRedirect`.
- **Desktop** client — what the `iris` CLI drives for browser login. Its id is
  registered in the Web client's `programmaticClients`, so IAP admits the CLI's
  bearer ID token (whose `aud` is the desktop client id).

The **same** client pair can protect every cluster's backend service; reuse them
across clusters rather than minting one set per cluster.

## Bootstrap

Deploy the frontend-owning cluster first (it creates the shared frontend), then
each additional cluster (each adds its backend + host route):

```bash
# marin — owns the shared frontend; its backend is the URL map default.
uv run lib/iris/scripts/iap_gclb.py deploy marin \
    --domain iris.oa.dev \
    --web-client-secrets web.json \
    --desktop-client-secrets desktop.json \
    --member user:you@example.com

# marin-dev — adds a backend + a host rule (iris-dev.oa.dev) on the shared LB.
uv run lib/iris/scripts/iap_gclb.py deploy marin-dev \
    --domain iris-dev.oa.dev \
    --web-client-secrets web.json \
    --desktop-client-secrets desktop.json \
    --member user:you@example.com
```

`deploy` runs the stages in dependency order — `cert` (managed SSL) → `backend`
(NEG → health check → backend service) → `iap` (enable + bind clients) →
`address` (shared IP) → URL map + host `route` → HTTPS proxy (attach cert) →
`:443` forwarding rule — then prints the shared IP, the URL, and the `auth.iap`
block to paste (see below). It finds the controller VM from the
`iris-<cluster>-controller` label; override its IP with `--controller-ip`. Add
`--dry-run` to trace every `gcloud` command without running it.

Two steps `deploy` does **not** do for you:

1. **DNS A record** — point the cluster's domain at the shared static IP. The
   managed SSL cert stays `PROVISIONING` until that resolves.
2. **Firewall** — run the `firewall` stage (or pass `--with-firewall`) so the LB
   health check can reach the controller. Without it the backend stays
   `UNHEALTHY`. It is kept separate because its deny-public option is a footgun
   (see [Firewall](#firewall)).

Individual stages are subcommands, each runnable on its own and idempotent; run
`uv run lib/iris/scripts/iap_gclb.py --help` for the full list. `status <cluster>`
reports what exists for the shared frontend and that cluster's backend (including
its proxy certs and JWT audience); `teardown <cluster>` removes a cluster's
backend + route + cert, leaving the shared frontend for the others.

## Cluster config

Enable IAP on the cluster by pasting the printed block into its config and
setting `auth.iap.signed_header_audience` to the backend-service audience (also
printed by `status`):

```yaml
auth:
  iap:
    url: https://iris.oa.dev
    oauth_client_id: <DESKTOP_CLIENT_ID>.apps.googleusercontent.com
    oauth_client_secret: <DESKTOP_CLIENT_SECRET>      # non-confidential, RFC 8252 §8.5
    audiences:
      - <DESKTOP_CLIENT_ID>.apps.googleusercontent.com
    signed_header_audience: /projects/<PROJECT_NUMBER>/global/backendServices/<BACKEND_ID>
  admin_users:
    - you@example.com
  optional: false   # tokenless calls that did NOT pass IAP are still rejected
```

A full template is in `lib/iris/config/examples/iap.yaml`. The audience is
per-cluster — it names that cluster's backend service, not the shared frontend.
The controller verifies IAP's signed `X-Goog-IAP-JWT-Assertion` (its `aud` is
`signed_header_audience`) to map a request to an identity; leave that field empty
to disable the IAP path entirely. For how the controller turns an identity into a
role (`dashboard` read-only vs `user`/`admin` after `iris login`), see
`lib/iris/src/iris/rpc/auth.py` and `lib/iris/docs/auth-loopback-transition.md`.

## Access control

IAP admits a request only if the authenticated Google identity holds
`roles/iap.httpsResourceAccessor` on the **cluster's** backend service — that
binding is the allowlist, and it is granted per backend. Grant principals with
the `grant` stage:

```bash
uv run lib/iris/scripts/iap_gclb.py grant marin --member group:team@example.com
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
passes without it. It is per-cluster (it targets that cluster's controller tag).

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
curl -i https://iris.oa.dev/                 # → 302 to accounts.google.com

# The only allow rule for :10000 is the Google LB range.
gcloud compute firewall-rules list --filter='allowed.ports:10000' \
  --format='table(name, sourceRanges.list(), targetTags.list())'
```

A `302` (browser) or `401` (RPC) carrying `x-goog-iap-generated-response: true`
means IAP answered and the request never reached the controller; a timeout on the
direct `:10000` probe confirms the port is closed to the public internet.
