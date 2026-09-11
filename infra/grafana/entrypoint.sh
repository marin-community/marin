#!/bin/bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Run nginx, Grafana, and the finelog bridge and tie their lifetimes together.
# bash, not sh: `wait -n` (exit when the *first* child dies) is a bash builtin.
#
# Grafana is useless without the bridge (every datasource points at it), so if
# any process exits, this exits and Cloud Run replaces the whole instance.
set -eu

# Grafana stays on loopback. Nginx listens on Cloud Run's public container port
# and adds the fixed Grafana role after IAP has authenticated the request.
default_public_port=8080
grafana_port=3000
public_port="${PORT:-${default_public_port}}"
case "${public_port}" in
  "" | *[!0-9]*)
    echo "entrypoint: PORT must be numeric, got ${public_port}" >&2
    exit 1
    ;;
esac
export GF_SERVER_HTTP_PORT="${grafana_port}"

# The image creates these directories, but make them again in case the runtime
# mounts an empty tmpfs over /tmp.
mkdir -p /tmp/nginx/client-body /tmp/nginx/fastcgi /tmp/nginx/proxy /tmp/nginx/scgi /tmp/nginx/uwsgi
sed \
  -e "s/PUBLIC_PORT/${public_port}/" \
  -e "s/GRAFANA_PORT/${grafana_port}/" \
  /etc/nginx/marin.conf > /tmp/nginx.conf
nginx -t -c /tmp/nginx.conf

# Grafana's host/port database settings reject Cloud SQL socket paths (the
# instance connection name's colons break host:port splitting), so the socket
# deployment hands us the socket directory separately and we compose the one
# database setting that accepts it: a URL with the directory as the `host`
# query parameter. The password must stay URL-safe (see infra/cloudsql/README.md).
if [ -n "${DATABASE_SOCKET_DIR:-}" ]; then
  export GF_DATABASE_URL="postgres://${GF_DATABASE_USER}:${GF_DATABASE_PASSWORD}@/${GF_DATABASE_NAME}?host=${DATABASE_SOCKET_DIR}"
fi

# Apply Marin-owned database migrations before Grafana or its permission cache starts.
# The first migration upgrades IAP accounts created as Viewer by older revisions.
/opt/bridge/venv/bin/python -m grafana_migrations

/opt/bridge/venv/bin/grafana-bridge &
bridge_pid=$!

/run.sh "$@" &
grafana_pid=$!

nginx -c /tmp/nginx.conf -g "daemon off;" &
nginx_pid=$!

# Cloud Run signals shutdown with SIGTERM to PID 1 and SIGKILLs the container a few
# seconds later, and bash does not pass that on to background children. Grafana has
# to receive it and run its own shutdown: that is when it snapshots the Alertmanager
# notification log to the database. Killed outright, it leaves a stale log behind and
# the replacement instance re-notifies every alert still firing.
terminating=0
forward_term() {
  terminating=1
  kill -TERM "$nginx_pid" "$grafana_pid" "$bridge_pid" 2>/dev/null || true
}
trap forward_term TERM INT

# Wait for whichever dies first, then take the container down with it.
exit_code=0
wait -n "$nginx_pid" "$grafana_pid" "$bridge_pid" || exit_code=$?

if [ "${terminating}" -eq 1 ]; then
  # Let all three finish their shutdown; Cloud Run's SIGKILL is the real deadline.
  wait "$nginx_pid" 2>/dev/null || true
  wait "$grafana_pid" 2>/dev/null || true
  wait "$bridge_pid" 2>/dev/null || true
  exit 0
fi

echo "entrypoint: a supervised process exited (status ${exit_code}); stopping container" >&2
kill "$nginx_pid" "$grafana_pid" "$bridge_pid" 2>/dev/null || true
exit "${exit_code}"
