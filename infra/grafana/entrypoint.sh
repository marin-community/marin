#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Run the finelog bridge alongside Grafana and tie their lifetimes together.
# bash, not sh: `wait -n` (exit when the *first* child dies) is a bash builtin.
#
# Grafana is useless without the bridge (every datasource points at it), so if
# either process exits, this exits — Cloud Run then replaces the whole instance.
# Silently serving a Grafana whose panels all error is the one outcome worth
# avoiding.
set -eu

/opt/bridge/venv/bin/grafana-bridge &
bridge_pid=$!

# Grafana reads its port from the config/env; Cloud Run tells us which one.
export GF_SERVER_HTTP_PORT="${PORT:-8080}"

# Grafana's host/port database settings reject Cloud SQL socket paths (the
# instance connection name's colons break host:port splitting), so the socket
# deployment hands us the socket directory separately and we compose the one
# database setting that accepts it: a URL with the directory as the `host`
# query parameter. The password must stay URL-safe (see infra/cloudsql/README.md).
if [ -n "${DATABASE_SOCKET_DIR:-}" ]; then
  export GF_DATABASE_URL="postgres://${GF_DATABASE_USER}:${GF_DATABASE_PASSWORD}@/${GF_DATABASE_NAME}?host=${DATABASE_SOCKET_DIR}"
fi

/run.sh "$@" &
grafana_pid=$!

# Wait for whichever dies first, then take the container down with it.
exit_code=0
wait -n "$bridge_pid" "$grafana_pid" || exit_code=$?
echo "entrypoint: a supervised process exited (status ${exit_code}); stopping container" >&2
kill "$bridge_pid" "$grafana_pid" 2>/dev/null || true
exit "${exit_code}"
