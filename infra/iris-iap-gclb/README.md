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

One stack per cluster (`iris-marin`, `iris-marin-dev`, …). The cluster name is
the resource-name prefix (`iris-<cluster>-*`) and the controller VM's GCE label
/ network tag (`iris-<cluster>-controller`).

`iap_gclb.py` does the whole rollout idempotently — every resource is one
`gcloud` create guarded by a `describe` probe, so the full `deploy` or any single
stage is safe to re-run. Run `uv run infra/iris-iap-gclb/iap_gclb.py --help` for
the subcommands.

This supersedes the Cloud Run proxy in `../iris-iap-proxy/`: GCLB reaches the
controller VM directly (one hop instead of two) and has no fixed serverless
request cap, so the controller's long-poll requests are not truncated. As this
rolls out, the Cloud Run proxy is retired.

## Deploy

```bash
uv run infra/iris-iap-gclb/iap_gclb.py deploy marin \
    --domain iris-marin.example.com \
    --support-email you@example.com \
    --member user:you@example.com
```

`deploy` runs all stages in dependency order: ensure the IAP OAuth brand +
client → reserve the static IP → managed SSL cert → firewall hardening → backend
(NEG → controller endpoint → health check → backend service → IAP) → frontend
(URL map → HTTPS proxy → `:443` forwarding rule) → optional IAP IAM grant. It
discovers the controller VM's internal IP from the `iris-<cluster>-controller`
label (override with `--controller-ip`), finds or creates the OAuth client
(override with `--oauth-client-id/secret`), and prints the reserved IP, URL, and
OAuth client id at the end. Add `--dry-run` to trace every `gcloud` command
without running it.

The one inherently manual step: create a **DNS A record** for the domain
pointing at the reserved static IP. The Google-managed SSL cert stays
`PROVISIONING` until that resolves.

## Individual stages

Each stage is a subcommand, runnable on its own and idempotent:

```bash
uv run infra/iris-iap-gclb/iap_gclb.py oauth    marin --support-email you@example.com
uv run infra/iris-iap-gclb/iap_gclb.py address  marin           # reserve + print the static IP
uv run infra/iris-iap-gclb/iap_gclb.py cert     marin --domain iris-marin.example.com
uv run infra/iris-iap-gclb/iap_gclb.py firewall marin           # allow LB ranges, deny public
uv run infra/iris-iap-gclb/iap_gclb.py backend  marin           # NEG + health check + backend + IAP
uv run infra/iris-iap-gclb/iap_gclb.py frontend marin           # URL map + HTTPS proxy + forwarding rule
uv run infra/iris-iap-gclb/iap_gclb.py grant    marin --member user:teammate@example.com
uv run infra/iris-iap-gclb/iap_gclb.py status   marin           # what exists + cert state + reserved IP
uv run infra/iris-iap-gclb/iap_gclb.py teardown marin           # delete the LB stack (keeps IP + OAuth client)
```

## Firewall hardening

`firewall` locks the controller VM's port `10000` down: it allows ingress only
from the Google front-end / health-check / IAP ranges (`130.211.0.0/22`,
`35.191.0.0/16`) and **denies** direct public ingress (`0.0.0.0/0`, lower
priority). Without these, anyone who learns the VM's IP could reach the
controller directly, bypassing IAP. The rules target the
`iris-<cluster>-controller` network tag, so the controller VM must carry that
tag.

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
