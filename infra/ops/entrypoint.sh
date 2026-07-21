#!/bin/sh
set -eu

# An empty libpq URI intentionally picks up PGHOST/PGDATABASE/PGUSER/PGPASSWORD.
# This keeps the password in Secret Manager-backed environment, not argv.
set -- serve \
  --database-url postgresql:// \
  --migrations /app/migrations \
  --grafana-database-url-env GRAFANA_DATABASE_URL \
  --grafana-database-password-env GRAFANA_PGPASSWORD \
  --grafana-poll-interval "${GRAFANA_POLL_INTERVAL:-60}" \
  --public-url "${OPS_PUBLIC_URL}" \
  --slack-webhook-url-env OPS_SLACK_WEBHOOK \
  --auth-mode iap \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --static-dir /app/dashboard \
  --repo-revision "${OPS_REPO_REVISION}" \
  --skill-revision "${OPS_SKILL_REVISION}"

if [ "${OPS_AGENT_MODE}" = "stub" ]; then
  exec ops-workflow "$@" --agent-mode stub
fi

exec ops-workflow "$@" \
  --agent-mode loom \
  --loom-api-url "${LOOM_API_URL}" \
  --loom-token-env LOOM_TOKEN \
  --loom-agent "${LOOM_AGENT:-codex}" \
  --loom-effort "${LOOM_EFFORT:-low}" \
  --repo-root "${LOOM_REPO_ROOT}" \
  --loom-base "${LOOM_BASE}"
