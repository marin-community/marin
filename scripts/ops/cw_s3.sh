#!/usr/bin/env bash
# Run the AWS CLI against CoreWeave AI Object Storage.
#
# CoreWeave's S3 endpoint needs three things plain AWS_* variables cannot express
# together: a custom endpoint, virtual-hosted addressing (path-style is rejected
# outright), and region-less signing. This wrapper supplies them and forwards every
# argument to `aws`, so anything the CLI can do works:
#
#   set -a; source ~/marin.env; set +a          # provides CW_KEY_ID / CW_KEY_SECRET
#   scripts/ops/cw_s3.sh s3 ls s3://marin-us-east-02a/MarinFold/
#   scripts/ops/cw_s3.sh s3 ls --recursive --human-readable --summarize s3://.../tokenized/
#   scripts/ops/cw_s3.sh s3 cp s3://.../.stats.json -
#
# Inside a CoreWeave pod none of this is needed: task pods already carry the endpoint,
# addressing style, and credentials. This is for reading the buckets from outside.
set -euo pipefail

: "${CW_KEY_ID:?set CW_KEY_ID (CoreWeave object-storage access key id)}"
: "${CW_KEY_SECRET:?set CW_KEY_SECRET (CoreWeave object-storage access key secret)}"

# Virtual-hosted addressing is only expressible through a config file, not an env var.
config_file="$(mktemp)"
trap 'rm -f "$config_file"' EXIT
cat >"$config_file" <<'EOF'
[default]
s3 =
    addressing_style = virtual
EOF

AWS_CONFIG_FILE="$config_file" \
AWS_ACCESS_KEY_ID="$CW_KEY_ID" \
AWS_SECRET_ACCESS_KEY="$CW_KEY_SECRET" \
AWS_REGION="${CW_S3_REGION:-US-EAST-02A}" \
AWS_ENDPOINT_URL_S3="${CW_S3_ENDPOINT:-https://cwobject.com}" \
AWS_PAGER="" \
exec uv run --with awscli aws "$@"
