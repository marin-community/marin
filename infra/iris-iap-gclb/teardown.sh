#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Tear down the GCLB+IAP stack created by deploy.sh for a cluster.
#
# Deletes resources in dependency order:
#   forwarding rule -> target HTTPS proxy -> URL map -> backend service
#   -> NEG -> health check -> SSL cert
#
# Leaves the reserved static IP and the IAP OAuth client in place (so the
# DNS A record and consent screen survive a redeploy). A printed note at the
# end shows how to release them if you really want them gone.
#
# Usage:
#   ./teardown.sh marin
#   ./teardown.sh marin-dev

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="hai-gcp-models"
ZONE="us-central1-a"

CLUSTER="${1:-}"
if [[ -z "${CLUSTER}" ]]; then
    echo "Usage: $0 <cluster>" >&2
    echo "  e.g.  $0 marin" >&2
    exit 1
fi

NEG="iris-${CLUSTER}-neg"
HC="iris-${CLUSTER}-hc"
BACKEND="iris-${CLUSTER}-be"
URLMAP="iris-${CLUSTER}-urlmap"
HTTPS_PROXY="iris-${CLUSTER}-https-proxy"
FORWARDING_RULE="iris-${CLUSTER}-fr"
STATIC_IP="iris-${CLUSTER}-ip"
CERT="iris-${CLUSTER}-cert"

# Delete a resource, tolerating "already gone" so teardown is restartable.
# Args: <human label> <gcloud args...>
del() {
    local label="$1"
    shift
    echo "==> Deleting ${label}"
    if ! gcloud "$@" --project="${PROJECT}" --quiet; then
        echo "    (skip: ${label} missing or already deleted)"
    fi
}

del "forwarding rule ${FORWARDING_RULE}" \
    compute forwarding-rules delete "${FORWARDING_RULE}" --global

del "target HTTPS proxy ${HTTPS_PROXY}" \
    compute target-https-proxies delete "${HTTPS_PROXY}" --global

del "URL map ${URLMAP}" \
    compute url-maps delete "${URLMAP}" --global

del "backend service ${BACKEND}" \
    compute backend-services delete "${BACKEND}" --global

del "NEG ${NEG}" \
    compute network-endpoint-groups delete "${NEG}" --zone="${ZONE}"

del "health check ${HC}" \
    compute health-checks delete "${HC}" --global

del "SSL certificate ${CERT}" \
    compute ssl-certificates delete "${CERT}" --global

echo ""
echo "==> Stack for cluster=${CLUSTER} torn down."
echo "    Left in place (reusable across redeploys):"
echo "      - static IP ${STATIC_IP}  (keeps the DNS A record valid)"
echo "      - IAP OAuth client + brand (consent screen)"
echo ""
echo "    To release the static IP:"
echo "      gcloud compute addresses delete ${STATIC_IP} --project=${PROJECT} --global"
echo "    To delete the IAP OAuth client (list, then delete by full name):"
echo "      gcloud iap oauth-clients list projects/<PROJECT_NUMBER>/brands/<BRAND_ID> --project=${PROJECT}"
echo "      gcloud iap oauth-clients delete <CLIENT_NAME> --project=${PROJECT}"
