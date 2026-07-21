#!/bin/sh
set -eu

# An empty libpq URI intentionally picks up PGHOST/PGDATABASE/PGUSER/PGPASSWORD.
# This keeps the password in Secret Manager-backed environment, not argv.
args="serve --database-url postgresql:// --migrations /app/migrations --surface ${OPS_SURFACE} --auth-mode iap --host 0.0.0.0 --port ${PORT:-8080} --static-dir /app/dashboard --repo-revision ${OPS_REPO_REVISION} --skill-revision ${OPS_SKILL_REVISION}"

if [ "${OPS_SURFACE}" = "ingest" ]; then
  exec ops-workflow $args --agent-mode stub --grafana-webhook-secret-env OPS_ALERT_WEBHOOK_SECRET
fi

exec ops-workflow $args --agent-mode loom --loom-api-url "${LOOM_API_URL}" --loom-token-env LOOM_TOKEN --loom-agent "${LOOM_AGENT:-codex}" --loom-effort "${LOOM_EFFORT:-low}" --repo-root "${LOOM_REPO_ROOT}" --loom-base "${LOOM_BASE}"
