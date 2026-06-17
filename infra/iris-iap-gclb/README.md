# iris-iap-gclb

An external HTTPS Load Balancer (GCLB) fronting the Iris controller VM, with
Identity-Aware Proxy (IAP) enabled on the backend service. GCLB terminates
TLS, IAP authenticates the caller, and the backend forwards plain HTTP to the
controller VM on port `10000`.

```
                 HTTPS :443           HTTP :10000
  iris CLI / ───────────────▶ GCLB ───────────────▶ controller VM
  browser     Proxy-Auth +    (IAP)   X-Goog-IAP-*    (Connect-RPC,
              Authorization                            HTTP long-poll)
```

One LB stack per cluster (`iris-marin`, `iris-marin-dev`, …). The cluster name
is the resource-name prefix (`iris-<cluster>-*`) and the GCE discovery label
(`iris-<cluster>-controller=true`).

## When to use this vs the Cloud Run proxy

There are two ways to put IAP in front of the controller:

| | `iris-iap-gclb/` (this) | `../iris-iap-proxy/` (Cloud Run) |
|---|---|---|
| Path | GCLB → IAP → controller VM:10000 | Cloud Run (native IAP) → proxy → controller VM:10000 |
| Hops | one (direct to VM) | two (extra serverless hop) |
| Request timeout | LB backend timeout (tunable; set to 120s here) | Cloud Run caps requests at 300s |
| Setup | static IP + domain + DNS + managed cert | none (serverless URL) |
| Ops | a handful of LB resources to manage | one Cloud Run service |

Use **GCLB** when you want end-to-end HTTP straight to the controller with no
extra hop and no fixed serverless request cap on long-polls (the controller
holds HTTP long-poll requests up to ~120s). Use the **Cloud Run proxy** when
you want the simplest serverless setup and don't need a custom domain — it's
fewer moving parts, at the cost of an extra hop and the Cloud Run timeout.

## Prerequisites

- A **domain** with a **DNS A record** pointing at the reserved global static
  IP (`iris-<cluster>-ip`). A Google-managed SSL cert will not provision until
  DNS resolves to that IP.
- An **OAuth consent screen (brand)** and an **IAP OAuth client** in the
  project. The client id/secret are passed to `--iap` on the backend service.

Run `./deploy.sh <cluster> --setup` to print every one-time command (brand,
client, static IP, DNS note, managed cert, firewall, IAP IAM grants).

## One-time setup

```bash
./deploy.sh marin --setup     # prints commands; does NOT run them
```

That output covers: `gcloud iap oauth-brands create`, `gcloud iap
oauth-clients create`, reserving the global static IP, the DNS A-record
requirement, `gcloud compute ssl-certificates create`, the firewall
hardening rules, and granting `roles/iap.httpsResourceAccessor` to team
members on the backend service.

## Deploy / teardown

```bash
# fill in the reviewer placeholders (see below), then:
DOMAIN=iris-marin.yourdomain.com \
OAUTH_CLIENT_ID=... OAUTH_CLIENT_SECRET=... \
CONTROLLER_IP=10.x.x.x \
  ./deploy.sh marin

./teardown.sh marin           # deletes the LB stack; keeps IP + OAuth client
```

`deploy.sh` builds, in order: zonal NEG → health check (`/health` on
`:10000`) → backend service (HTTP, IAP enabled) → URL map → managed SSL cert
→ target HTTPS proxy → global forwarding rule (`:443`). It echoes the reserved
IP and `https://<domain>` at the end. The create commands are not idempotent;
re-runs may need `gcloud ... update` or a `teardown.sh` first.

## Firewall hardening

Lock the controller VM's port `10000` down: allow ingress only from the Google
front-end / health-check / IAP ranges and **deny** direct public ingress.

- Allow `tcp:10000` from `130.211.0.0/22` and `35.191.0.0/16`.
- Deny `tcp:10000` from `0.0.0.0/0` (lower-priority catch-all).

Both rules are printed by `--setup`. Without them, anyone who learns the VM's
IP could reach the controller directly, bypassing IAP.

## Auth: the two-token model

Two independent bearer tokens travel on each request:

- **IAP OIDC ID token** in `Proxy-Authorization: Bearer <id_token>` — proves
  *who the human/SA is* to IAP. IAP validates it before the request reaches the
  controller. (Sending it in `Proxy-Authorization` rather than `Authorization`
  is the IAP convention that frees `Authorization` for the app.)
- **Iris JWT** in `Authorization: Bearer <iris_jwt>` — the cluster-issued token
  from `iris login`, which the controller's own auth layer checks.

IAP passes identity to the controller via headers it injects after a
successful auth:

- `X-Goog-IAP-JWT-Assertion` — a signed JWT asserting the authenticated
  identity (verifiable against Google's public keys).
- `X-Goog-Authenticated-User-Email` — `accounts.google.com:user@example.com`.

## User flow: `iris login --cluster=<cluster>`

At a high level:

1. The client resolves the cluster's controller URL (`https://<domain>`).
2. It mints a Google OIDC **ID token** for the IAP OAuth client audience and
   attaches it as `Proxy-Authorization: Bearer …` so IAP lets the request
   through.
3. Through IAP, it calls the controller's `Login` RPC, exchanging its identity
   for a cluster **Iris JWT**, which is stored locally per cluster.
4. Subsequent `iris …` commands send both tokens: the IAP ID token in
   `Proxy-Authorization` and the stored Iris JWT in `Authorization`.

See `lib/iris/src/iris/cli/main.py` (`login`) for the JWT-exchange side of
this; the IAP ID-token side is the GCLB-specific addition this stack enables.
