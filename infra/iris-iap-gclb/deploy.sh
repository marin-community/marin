#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Stand up an external HTTPS Load Balancer (GCLB) fronting the Iris
# controller VM, with Identity-Aware Proxy (IAP) enabled on the backend.
#
#   client --HTTPS:443--> GCLB --(IAP)--> backend service --HTTP:10000--> controller VM
#
# This is the alternative to the Cloud Run proxy in ../iris-iap-proxy/.
# GCLB talks straight to the controller VM (no extra serverless hop and no
# Cloud Run 300s request cap that would truncate long-poll requests).
#
# One LB stack per cluster: the cluster name maps to both the resource name
# prefix (iris-<cluster>-*) and the GCE label used for controller discovery
# (iris-<cluster>-controller=true).
#
# A Google-managed SSL certificate REQUIRES a real domain with a DNS A
# record pointed at the reserved static IP — set DOMAIN below (or via env).
#
# Usage:
#   ./deploy.sh marin              # build the iris-marin-* LB stack
#   ./deploy.sh marin-dev          # build the iris-marin-dev-* LB stack
#   ./deploy.sh marin --setup      # print one-time setup commands (does not run them)

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="hai-gcp-models"
REGION="us-central1"
ZONE="us-central1-a"
CONTROLLER_PORT="10000"

CLUSTER="${1:-}"
if [[ -z "${CLUSTER}" || "${CLUSTER}" == "--setup" ]]; then
    echo "Usage: $0 <cluster> [--setup]" >&2
    echo "  e.g.  $0 marin" >&2
    echo "        $0 marin-dev" >&2
    exit 1
fi

# REVIEWER: replace with the real domain whose DNS A record points at the
# reserved static IP (iris-<cluster>-ip). The managed cert will not provision
# until DNS resolves to that IP. Override with: DOMAIN=iris.foo.com ./deploy.sh ...
DOMAIN="${DOMAIN:-iris-${CLUSTER}.example.com}"

CONTROLLER_LABEL="iris-${CLUSTER}-controller"

# Derived resource names — one set per cluster.
NEG="iris-${CLUSTER}-neg"            # zonal NEG -> controller VM:10000
HC="iris-${CLUSTER}-hc"              # health check -> /health on :10000
BACKEND="iris-${CLUSTER}-be"         # backend service (HTTP, IAP enabled)
URLMAP="iris-${CLUSTER}-urlmap"      # url map -> backend
HTTPS_PROXY="iris-${CLUSTER}-https-proxy"
FORWARDING_RULE="iris-${CLUSTER}-fr" # global forwarding rule :443
STATIC_IP="iris-${CLUSTER}-ip"       # reserved global static IP
CERT="iris-${CLUSTER}-cert"          # Google-managed SSL cert for DOMAIN

if [[ "${2:-}" == "--setup" ]]; then
    cat <<SETUP
=== One-Time Setup (cluster=${CLUSTER}) ===

# 1. Configure the OAuth consent/brand for IAP (one-time per project).
#    A brand can also be created in the Cloud Console:
#    https://console.cloud.google.com/auth/branding?project=${PROJECT}
gcloud iap oauth-brands create \\
  --project=${PROJECT} \\
  --application_title="Iris Controller" \\
  --support_email="you@example.com"

#    List the brand to get its name (projects/<PROJECT_NUMBER>/brands/<BRAND_ID>):
gcloud iap oauth-brands list --project=${PROJECT}

# 2. Create an IAP OAuth client under that brand (one-time per project).
#    Capture the printed clientId / secret — they feed the --iap flag below.
gcloud iap oauth-clients create projects/<PROJECT_NUMBER>/brands/<BRAND_ID> \\
  --project=${PROJECT} \\
  --display_name="Iris GCLB IAP (${CLUSTER})"

# 3. Reserve a global static IP for the load balancer.
gcloud compute addresses create ${STATIC_IP} \\
  --project=${PROJECT} \\
  --global

#    Read the reserved address:
gcloud compute addresses describe ${STATIC_IP} \\
  --project=${PROJECT} --global --format='value(address)'

# 4. DNS: create an A record for ${DOMAIN} pointing at the reserved IP above.
#    The managed cert (step 5) stays PROVISIONING until this resolves.

# 5. Create a Google-managed SSL certificate for the domain.
#    (deploy.sh also creates this; shown here for the manual path.)
gcloud compute ssl-certificates create ${CERT} \\
  --project=${PROJECT} \\
  --domains=${DOMAIN} \\
  --global

# 6. Firewall: allow ingress to tcp:${CONTROLLER_PORT} on the controller VM
#    ONLY from the Google front-end / health-check / IAP ranges, and deny
#    direct public ingress to that port. Adjust --target-tags to match the
#    controller VM's network tag.
gcloud compute firewall-rules create iris-${CLUSTER}-allow-lb \\
  --project=${PROJECT} \\
  --network=default \\
  --direction=INGRESS \\
  --action=ALLOW \\
  --rules=tcp:${CONTROLLER_PORT} \\
  --source-ranges=130.211.0.0/22,35.191.0.0/16 \\
  --target-tags=iris-${CLUSTER}-controller \\
  --priority=900

gcloud compute firewall-rules create iris-${CLUSTER}-deny-public-10000 \\
  --project=${PROJECT} \\
  --network=default \\
  --direction=INGRESS \\
  --action=DENY \\
  --rules=tcp:${CONTROLLER_PORT} \\
  --source-ranges=0.0.0.0/0 \\
  --target-tags=iris-${CLUSTER}-controller \\
  --priority=1000

# 7. Grant team members access through IAP on the backend service.
#    (Run after deploy.sh has created the backend service ${BACKEND}.)
gcloud iap web add-iam-policy-binding \\
  --project=${PROJECT} \\
  --resource-type=backend-services \\
  --service=${BACKEND} \\
  --member="user:you@example.com" \\
  --role="roles/iap.httpsResourceAccessor"

# 8. Find your project number (referenced above and by IAP):
gcloud projects describe ${PROJECT} --format='value(projectNumber)'

SETUP
    exit 0
fi

# REVIEWER: supply the IAP OAuth client credentials from step 2 of --setup.
# These are required to enable IAP on the backend service.
OAUTH_CLIENT_ID="${OAUTH_CLIENT_ID:-<OAUTH_CLIENT_ID>}"
OAUTH_CLIENT_SECRET="${OAUTH_CLIENT_SECRET:-<OAUTH_CLIENT_SECRET>}"

# REVIEWER: the GCE instance group backing the controller VM. The controller
# is a single VM, so we attach it to a zonal NEG by internal IP:port. Set the
# controller VM's internal IP (discoverable via the GCE controller label).
#   gcloud compute instances list --project=${PROJECT} \
#     --filter="labels.${CONTROLLER_LABEL}=true"
CONTROLLER_IP="${CONTROLLER_IP:-<CONTROLLER_INTERNAL_IP>}"

echo "==> Building GCLB+IAP stack for cluster=${CLUSTER} (label=${CONTROLLER_LABEL}, domain=${DOMAIN})"
echo "    Re-runs may need 'gcloud ... update' or pre-deletion (see teardown.sh);"
echo "    the create commands below are not idempotent."

# 1. Zonal NEG of type GCE_VM_IP_PORT pointing at the controller VM:10000.
echo "==> [1/8] Creating zonal NEG ${NEG}"
gcloud compute network-endpoint-groups create "${NEG}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --network=default \
  --subnet=default \
  --network-endpoint-type=GCE_VM_IP_PORT \
  --default-port="${CONTROLLER_PORT}"

# Attach the controller VM endpoint (internal IP + port) to the NEG.
echo "==> [2/8] Attaching controller endpoint ${CONTROLLER_IP}:${CONTROLLER_PORT} to ${NEG}"
gcloud compute network-endpoint-groups update "${NEG}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --add-endpoint="ip=${CONTROLLER_IP},port=${CONTROLLER_PORT}"

# 2. Health check against /health on the controller port.
echo "==> [3/8] Creating health check ${HC} (HTTP /health :${CONTROLLER_PORT})"
gcloud compute health-checks create http "${HC}" \
  --project="${PROJECT}" \
  --global \
  --port="${CONTROLLER_PORT}" \
  --request-path="/health" \
  --check-interval=10s \
  --timeout=5s \
  --healthy-threshold=2 \
  --unhealthy-threshold=3

# 3. Backend service (HTTP), wiring in the NEG and health check.
echo "==> [4/8] Creating backend service ${BACKEND}"
gcloud compute backend-services create "${BACKEND}" \
  --project="${PROJECT}" \
  --global \
  --protocol=HTTP \
  --port-name=http \
  --health-checks="${HC}" \
  --timeout=120s \
  --load-balancing-scheme=EXTERNAL_MANAGED

echo "==> [4b/8] Adding NEG ${NEG} to backend service ${BACKEND}"
gcloud compute backend-services add-backend "${BACKEND}" \
  --project="${PROJECT}" \
  --global \
  --network-endpoint-group="${NEG}" \
  --network-endpoint-group-zone="${ZONE}" \
  --balancing-mode=RATE \
  --max-rate-per-endpoint=1000

# 4. Enable IAP on the backend service.
echo "==> [5/8] Enabling IAP on backend service ${BACKEND}"
gcloud compute backend-services update "${BACKEND}" \
  --project="${PROJECT}" \
  --global \
  --iap=enabled,oauth2-client-id="${OAUTH_CLIENT_ID}",oauth2-client-secret="${OAUTH_CLIENT_SECRET}"

# 5. URL map routing everything to the backend service.
echo "==> [6/8] Creating URL map ${URLMAP}"
gcloud compute url-maps create "${URLMAP}" \
  --project="${PROJECT}" \
  --global \
  --default-service="${BACKEND}"

# 6. Managed SSL certificate for the domain, then the target HTTPS proxy.
echo "==> [7/8] Creating managed SSL cert ${CERT} for ${DOMAIN} and HTTPS proxy ${HTTPS_PROXY}"
gcloud compute ssl-certificates create "${CERT}" \
  --project="${PROJECT}" \
  --global \
  --domains="${DOMAIN}"

gcloud compute target-https-proxies create "${HTTPS_PROXY}" \
  --project="${PROJECT}" \
  --global \
  --url-map="${URLMAP}" \
  --ssl-certificates="${CERT}"

# 7. Global forwarding rule on the reserved static IP, port 443.
echo "==> [8/8] Creating global forwarding rule ${FORWARDING_RULE} on ${STATIC_IP}:443"
gcloud compute forwarding-rules create "${FORWARDING_RULE}" \
  --project="${PROJECT}" \
  --global \
  --address="${STATIC_IP}" \
  --target-https-proxy="${HTTPS_PROXY}" \
  --ports=443 \
  --load-balancing-scheme=EXTERNAL_MANAGED

RESERVED_IP="$(gcloud compute addresses describe "${STATIC_IP}" \
  --project="${PROJECT}" --global --format='value(address)')"

echo ""
echo "==> Done. LB stack for cluster=${CLUSTER} created."
echo "    Reserved IP : ${RESERVED_IP}"
echo "    Domain      : ${DOMAIN}  (ensure a DNS A record -> ${RESERVED_IP})"
echo "    URL         : https://${DOMAIN}"
echo ""
echo "    The managed cert (${CERT}) provisions only after DNS resolves to"
echo "    the reserved IP; expect a few minutes to ACTIVE. Grant access with"
echo "    step 7 of './deploy.sh ${CLUSTER} --setup'."
