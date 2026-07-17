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
FROM grafana/grafana:13.1.0-ubuntu

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN grafana cli --pluginsDir /var/lib/grafana/plugins plugins install yesoreyeram-infinity-datasource

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
COPY dashboards /etc/grafana/dashboards
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh && chown -R grafana:root /opt/bridge /etc/grafana/dashboards

USER grafana

# Cloud Run injects PORT and routes to it; Grafana serves the UI, the bridge sits
# on loopback beside it (Grafana's backend datasources fetch server-side, so
# nothing outside the container needs to reach it).
ENV GF_PATHS_PROVISIONING=/etc/grafana/provisioning \
    GF_AUTH_ANONYMOUS_ENABLED=true \
    GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    GF_AUTH_DISABLE_LOGIN_FORM=true \
    GF_SECURITY_DISABLE_INITIAL_ADMIN_CREATION=true \
    GF_ANALYTICS_REPORTING_ENABLED=false \
    GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES=false

ENTRYPOINT ["/opt/entrypoint.sh"]
