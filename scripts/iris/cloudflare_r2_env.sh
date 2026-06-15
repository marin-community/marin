#!/usr/bin/env bash
# Emit shell exports for Cloudflare R2 S3-compatible credentials.
#
# Usage:
#   eval "$(scripts/iris/cloudflare_r2_env.sh ~/.config/marin/marin-r2.env)"
#
# The env file may contain either:
#   R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT_URL
# or:
#   CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN
#
# For an R2-capable Cloudflare API token, the S3 Access Key ID is the token id
# and the Secret Access Key is sha256(token value), per Cloudflare's R2 docs.

set -euo pipefail

DEFAULT_ENV_FILE="$HOME/.config/marin/marin-r2.env"
if [ ! -f "$DEFAULT_ENV_FILE" ] && [ -f "$HOME/.config/marin/cloudflare-r2.env" ]; then
    DEFAULT_ENV_FILE="$HOME/.config/marin/cloudflare-r2.env"
fi
ENV_FILE="${1:-${MARIN_R2_ENV_FILE:-$DEFAULT_ENV_FILE}}"
API_BASE="https://api.cloudflare.com/client/v4"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

emit_export() {
    local name="$1"
    local value="$2"
    printf "export %s=%q\n" "$name" "$value"
}

load_env_file() {
    if [ -f "$ENV_FILE" ]; then
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
        return
    fi

    if [ "$#" -gt 0 ]; then
        die "env file not found: $ENV_FILE"
    fi
}

verify_token() {
    local endpoint="$1"
    curl -fsS \
        --header "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
        "$endpoint"
}

derive_access_key_id() {
    local response

    if response="$(verify_token "${API_BASE}/user/tokens/verify" 2>/dev/null)"; then
        jq -er '.result.id' <<<"$response"
        return
    fi

    [ -n "${CLOUDFLARE_ACCOUNT_ID:-}" ] || die "CLOUDFLARE_ACCOUNT_ID is required to verify an account API token"
    response="$(verify_token "${API_BASE}/accounts/${CLOUDFLARE_ACCOUNT_ID}/tokens/verify")"
    jq -er '.result.id' <<<"$response"
}

load_env_file "$@"

if [ -z "${R2_ACCESS_KEY_ID:-}" ] || [ -z "${R2_SECRET_ACCESS_KEY:-}" ]; then
    [ -n "${CLOUDFLARE_API_TOKEN:-}" ] || die "set R2_* credentials or CLOUDFLARE_API_TOKEN"
    require_command curl
    require_command jq
    require_command shasum

    R2_ACCESS_KEY_ID="$(derive_access_key_id)"
    R2_SECRET_ACCESS_KEY="$(printf "%s" "$CLOUDFLARE_API_TOKEN" | shasum -a 256 | awk '{print $1}')"
fi

if [ -z "${R2_ENDPOINT_URL:-}" ]; then
    [ -n "${CLOUDFLARE_ACCOUNT_ID:-}" ] || die "CLOUDFLARE_ACCOUNT_ID is required when R2_ENDPOINT_URL is unset"
    R2_ENDPOINT_URL="https://${CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com"
fi

emit_export R2_ACCESS_KEY_ID "$R2_ACCESS_KEY_ID"
emit_export R2_SECRET_ACCESS_KEY "$R2_SECRET_ACCESS_KEY"
emit_export R2_ENDPOINT_URL "$R2_ENDPOINT_URL"
emit_export AWS_ACCESS_KEY_ID "$R2_ACCESS_KEY_ID"
emit_export AWS_SECRET_ACCESS_KEY "$R2_SECRET_ACCESS_KEY"
emit_export AWS_ENDPOINT_URL "$R2_ENDPOINT_URL"
emit_export AWS_ENDPOINT_URL_S3 "$R2_ENDPOINT_URL"
