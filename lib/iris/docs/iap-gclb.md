# IAP and GCLB ingress for GCE services

The `marin` Pulumi stack owns the external HTTPS load balancer in front of the
GCE-hosted Iris controllers and finelog. The declaration lives under
`provisioning.gcp.gclb` in `lib/iris/config/marin.yaml`; the implementation is
`iac.gcp.gclb.GcpGclbIap`.

```text
                                 ┌─ IAP backend ────────────────┐
client ── HTTPS:443 ── GCLB ─────┼─ capability backend ─────────┼─ controller VM:10000
                                 └─ Armor-restricted backend ───┴─ finelog VM:10001
```

One shared frontend carries every configured host:

- `iris-marin-ip`, the global static address;
- `iris-marin-urlmap`, with one host matcher per backend;
- `iris-marin-https-proxy`, with a managed certificate per host;
- `iris-marin-fr`, the global `:443` forwarding rule.

Each controller contributes a zonal NEG, endpoint, health check, IAP-enabled
backend service, certificate, firewall rule, and URL-map route. The optional
capability backend uses the same NEG and health check but leaves IAP disabled
only for `/proxy/t` and `/proxy/t/*`. Each finelog contributes an IAP-free
backend protected by Cloud Armor and its own firewall rules.

## Configuration sources

The aggregate `marin` stack owns the shared frontend, including the `marin-dev`
route. Pulumi reads values from their existing owners instead of copying VM
details into the GCLB block:

- Controller domain, zone, port, VM name, firewall tag, project number, and
  imported backend ID come from each Iris cluster config. The internal VM IP is
  read from GCE during the Pulumi program.
- Finelog VM name, zone, port, and firewall tag come from
  `lib/finelog/config/<cluster>.yaml`. The internal VM IP is read from GCE.
- `provisioning.gcp.gclb` selects the controller routes, finelog domains, IAP
  member exceptions, and finelog sender CIDRs.
- `rigging.auth.MARIN_DESKTOP_OAUTH_CLIENT` is registered as the programmatic
  IAP client.

Controller creation attaches the `iris-<cluster>-controller` network tag.
Finelog's GCE deployment config attaches `finelog-<cluster>-lb`. A replacement
VM therefore receives its firewall identity without a follow-up mutation.

DNS records and the Web OAuth client are external prerequisites. Existing Web
OAuth client fields are deliberately ignored by Pulumi so its secret never
enters Pulumi state. The Web client redirect URI is:

```text
https://iap.googleapis.com/v1/oauth/clientIds/<CLIENT_ID>:handleRedirect
```

## Preview, adoption, and updates

The controller and finelog VMs must exist before preview because their current
internal addresses become NEG endpoint inputs.

```bash
cd infra/pulumi
pulumi stack select marin
pulumi preview
```

For resources that predate the Pulumi stack, use the repository's Program-first
import flow from the repository root:

```bash
uv run --package marin-iac --extra deploy python infra/pulumi/import_resources.py \
  generate --stack marin --output /tmp/marin-iac-marin-import.json

uv run --package marin-iac --extra deploy python infra/pulumi/import_resources.py \
  apply --stack marin --file /tmp/marin-iac-marin-import.json
```

Inspect the generated transaction before applying it. Keep existing GCLB,
firewall, IAP settings, NEG endpoint, certificate, and Armor entries in the
import; remove entries that should be created. The final preview must not
replace or delete the shared IP, URL map, proxy, forwarding rule, certificates,
or backend services. Then reconcile the declaration normally:

```bash
cd infra/pulumi
pulumi up
pulumi stack output gclb_ip_address
```

Point every configured host's DNS A record at that output. Managed certificates
remain `PROVISIONING` until DNS resolves to the shared address and the proxy
serves them.

Changes to routes, firewall policy, IAP access, programmatic clients, sender
CIDRs, or certificates go through the same config-review, preview, and `pulumi
up` path. There are no per-stage `gcloud` commands.

## Controller access control

IAP authentication and authorization are separate checks:

- The edge token's `aud` must name the registered Web client or a configured
  programmatic client. Failure is normally `401`.
- The caller identity must hold `roles/iap.httpsResourceAccessor` on that
  backend service. Failure is normally `403`.

Add service-specific exceptions to a controller's `iap_members` list in
`provisioning.gcp.gclb.controllers`. Broader human and automation access remains
in the shared IAM declaration when appropriate. For example, the production
controller explicitly grants the infra-probes service account.

The three common caller paths are:

| Caller | Edge token audience | Identity |
| --- | --- | --- |
| Human after `iris login` | Marin desktop OAuth client | Signed-in user |
| Service account or CI | Configured programmatic audience, otherwise the desktop client | Service-account email |
| Direct VPC or loopback caller | None | Transport trusted by `auth.trusted_cidrs` |

IAP forwards a verified `X-Goog-IAP-JWT-Assertion`. The controller verifies its
audience against `auth.iap.signed_header_audience` and maps the asserted email
to an Iris role. The load balancer source ranges must never be added to
`trusted_cidrs`: all internet requests arrive from those ranges after the proxy.

For unattended access outside GCE, use service-account impersonation rather
than a downloaded key:

```bash
gcloud auth application-default login \
  --impersonate-service-account=iris-controller@hai-gcp-models.iam.gserviceaccount.com
iris --cluster=marin cluster status
```

The operator needs `roles/iam.serviceAccountTokenCreator` on the impersonated
account, and that account needs `roles/iap.httpsResourceAccessor` on the target
backend.

## Firewall boundaries

The controller allow rule admits port `10000` only from Google's frontend and
health-check ranges, `130.211.0.0/22` and `35.191.0.0/16`, and targets only the
controller tag. `deny_public: true` adds a lower-priority blanket deny. That deny
also overrides `default-allow-internal`, so leave it disabled while workers or
tasks dial the controller's internal address.

Finelog uses a separate tag and two rules:

- priority 900 allows its serving port from the VPC and Google frontend ranges;
- priority 1000 denies every other source.

The public finelog route has no IAP identity because federated finelogs
authenticate every push with their own JWT. Cloud Armor admits only the sender
CIDRs declared in `sender_source_ranges`, then finelog verifies the sender key.
CIDRs are a network boundary, not the caller identity.

## Capability URLs

When `token_proxy` is enabled, the URL map sends only `/proxy/t` and
`/proxy/t/*` to `iris-<cluster>-proxy-be`, where IAP is disabled. Every other
path uses the IAP backend.

A capability URL contains a scoped, expiring endpoint token:

```text
/proxy/t/<token>/<endpoint>/<sub_path>
```

The controller validates that token before forwarding. The firewall still
admits only GCLB traffic, so the alternate backend does not expose the VM
directly. Disable `token_proxy` in the controller route to remove this public
capability path.

## Verification

```bash
# Direct public access to the controller port should time out.
curl --connect-timeout 8 http://<CONTROLLER_EXTERNAL_IP>:10000/health

# Browser traffic should be intercepted by IAP.
curl -I https://iris.oa.dev/

# Review the declared ingress boundaries.
gcloud compute firewall-rules list \
  --filter='name~iris-.*-(allow-lb|deny-public)' \
  --format='table(name,priority,sourceRanges.list(),targetTags.list())'

gcloud compute security-policies describe finelog-marin-armor
```

A `302`, or a `401` carrying `x-goog-iap-generated-response: true`, means IAP
answered before the controller. A direct-port timeout confirms that no public
firewall rule bypasses the load balancer. For a persistent browser OAuth error,
test in a private window to exclude an existing IAP cookie, then inspect the
redirect trace and the live IAP settings.
