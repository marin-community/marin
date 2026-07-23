# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Grafana plus the finelog bridge, in one Cloud Run container.
#
# The -ubuntu variant, not the default Alpine one: the bridge needs pyarrow (via
# marin-finelog), which publishes manylinux wheels — on musl it would fall back to
# building Arrow from source.
#
# The Infinity plugin is installed at build time rather than through
# GF_INSTALL_PLUGINS, so a cold start never depends on grafana.com being reachable.
# The version is pinned: the provisioned alert rules evaluate through this plugin,
# so an untested plugin build must never arrive as a side effect of an image build.
FROM node:24.16.0-bookworm-slim AS infra-panel

WORKDIR /plugin
COPY marin-infra-panel/package.json marin-infra-panel/package-lock.json ./
RUN npm ci
COPY marin-infra-panel/.config .config
COPY marin-infra-panel/src src
COPY marin-infra-panel/CHANGELOG.md marin-infra-panel/LICENSE marin-infra-panel/README.md ./
COPY marin-infra-panel/tsconfig.json ./
RUN npm run build

FROM grafana/grafana:13.1.1-ubuntu

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN grafana cli --pluginsDir /var/lib/grafana/plugins plugins install yesoreyeram-infinity-datasource 3.10.1
COPY --from=infra-panel /plugin/dist /var/lib/grafana/plugins/marin-infra-panel

# The bridge lives in its own venv; nothing else in the image is Python.
#
# uv, not pip, and locked: marin-finelog depends on marin-rigging, which only ever
# publishes `0.2.x.devN` prereleases, so plain pip cannot resolve it and bare
# `pip --pre` would drag every dependency onto prereleases (notably httpx 1.0.dev,
# which breaks aiobotocore at import). uv's `prerelease = "if-necessary"` in
# pyproject.toml takes prereleases only for packages that have no stable release.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock /opt/bridge/
COPY src /opt/bridge/src
RUN cd /opt/bridge \
    && UV_PROJECT_ENVIRONMENT=/opt/bridge/venv uv sync --frozen --no-dev --compile-bytecode

COPY provisioning /etc/grafana/provisioning
# dashboards/*.json holds panelRef markers for panels shared across dashboards
# (see src/dashboard_stitch.py); this resolves them into the full panel bodies
# Grafana actually loads, the same way the infra-panel build above resolves
# TSX into JS. The source stays git-reviewable; only the resolved output ships.
COPY dashboards /tmp/dashboards-src
RUN /opt/bridge/venv/bin/python -m dashboard_stitch \
    --src-dir /tmp/dashboards-src --panels-dir /tmp/dashboards-src/panels --out-dir /etc/grafana/dashboards \
    && rm -rf /tmp/dashboards-src
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh && chown -R grafana:root /opt/bridge /etc/grafana/dashboards

USER grafana

# Cloud Run injects PORT and routes to it; Grafana serves the UI, the bridge sits
# on loopback beside it (Grafana's backend datasources fetch server-side, so
# nothing outside the container needs to reach it).
#
# Auth: IAP is the gate, and it stamps the caller's identity into
# X-Goog-Authenticated-User-Email (value "accounts.google.com:<email>", so the
# Grafana username carries that prefix). auth.proxy turns that into a real Grafana
# user — the dashboard browse, stars, and preferences endpoints all reject the
# anonymous user (userId 0), so a real user is what makes the list page work.
# Trusting the header is safe because the service is invokable only by the IAP
# service agent, so every request has already passed through IAP, which sets the
# header itself. Anonymous stays enabled as a fallback: a request without the
# header (a health check, a misconfig) degrades to read-only rather than 401.
#
# SMTP: plain Gmail submission (smtp.gmail.com:587, STARTTLS), authenticated as
# grafana@openathena.ai with an app password (GF_SMTP_PASSWORD from Secret Manager).
# The app sends mail itself — no Workspace relay, so deliverability rests on the
# sending account alone. GF_SMTP_ENABLED is deliberately absent here: the deploy sets
# it only when the credentials secret exists, so an image without credentials runs
# with SMTP off and alerts reach Slack only.
ENV GF_SMTP_HOST=smtp.gmail.com:587 \
    GF_SMTP_FROM_ADDRESS=grafana@openathena.ai \
    GF_SMTP_USER=grafana@openathena.ai \
    GF_SMTP_STARTTLS_POLICY=MandatoryStartTLS

ENV GF_PATHS_PROVISIONING=/etc/grafana/provisioning \
    GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/etc/grafana/dashboards/home.json \
    GF_PLUGINS_ALLOW_LOADING_UNSIGNED_PLUGINS=marin-infra-panel \
    GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    GF_AUTH_DISABLE_LOGIN_FORM=true \
    GF_AUTH_PROXY_ENABLED=true \
    GF_AUTH_PROXY_HEADER_NAME=X-Goog-Authenticated-User-Email \
    GF_AUTH_PROXY_HEADER_PROPERTY=username \
    GF_AUTH_PROXY_AUTO_SIGN_UP=true \
    GF_AUTH_PROXY_ENABLE_LOGIN_TOKEN=false \
    GF_AUTH_PROXY_SYNC_TTL=60 \
    GF_USERS_AUTO_ASSIGN_ORG=true \
    GF_USERS_AUTO_ASSIGN_ORG_ROLE=Viewer \
    GF_SECURITY_DISABLE_INITIAL_ADMIN_CREATION=true \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES=false

ENTRYPOINT ["/opt/entrypoint.sh"]
