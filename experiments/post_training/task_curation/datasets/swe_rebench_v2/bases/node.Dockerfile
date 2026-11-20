# SWE-rebench V2.1 repair — node group base (js/ts rows).
# Covers node_16 (2,357) and node_20 (1,528) rows (4,885 tasks).
# Upstream anchor: base_dockerfiles/Dockerfile_node_16 / Dockerfile_node_20 (ubuntu:22.04 +
# NodeSource floating latest-16.x/20.x + corepack pnpm + Chrome for browser testing).
# Per-version toolchains: pinned official node tarballs side by side under
# /opt/toolchains/node-<major>; the task setup symlinks the row's declared major into
# /usr/local/node. Pinned minors (16.20.2 / 20.18.1) are the final releases of the majors
# upstream's floating NodeSource installs resolved to; authored choice, recorded here.
# Interpreter provisioning (repair, disclosed): apt python3 (jammy ships 3.10).
# NOTE: the upstream Chrome-for-testing layer is DROPPED from the repair base (disclosed):
# in-region its dl.google.com apt source fails and the tolerated-failure layer is the one
# common factor in every kaniko build that lost apt-installed binaries from the pushed
# image (v1, v9, v11 — go/julia controls built without it verify clean). Node rows whose
# tests require Chrome will fail their oracle honestly and go to the typed exclusion list.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update \
    && if [ -f /var/lib/dpkg/statoverride ]; then \
         while read -r mode owner group path; do \
           [ -n "$path" ] || continue; \
           if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then \
             dpkg-statoverride --remove "$path" || true; \
           fi; \
         done < /var/lib/dpkg/statoverride; \
       fi \
    && apt-get install -y --no-install-recommends \
    wget curl git build-essential jq gnupg ca-certificates apt-transport-https xz-utils python3 unzip \
    && rm -rf /var/lib/apt/lists/*

# Declared interpreter provisioning guard.
RUN test "$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "python3 >= 3.9 required on PATH" >&2; exit 1; }

# Chrome for browser-testing repos: DROPPED (see the disclosure above).

# Per-version node from pinned official tarballs.
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) suffix="x64" ;; \
      arm64) suffix="arm64" ;; \
      *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    for release in v16.20.2 v20.18.1; do \
      major="${release#v}"; major="${major%%.*}"; \
      url="https://nodejs.org/dist/${release}/node-${release}-linux-${suffix}.tar.xz"; \
      mkdir -p "/opt/toolchains/node-${major}"; \
      curl -fsSL "$url" | tar -xJ -C "/opt/toolchains/node-${major}" --strip-components=1; \
    done
RUN test -x /opt/toolchains/node-16/bin/node && test -x /opt/toolchains/node-20/bin/node \
    || { echo "node toolchains missing" >&2; exit 1; }

# Default to node 16 (the most common declared major); the task setup re-points per row.
RUN ln -sfn /opt/toolchains/node-16 /usr/local/node
ENV PATH="/usr/local/node/bin:${PATH}"

# corepack/pnpm available per toolchain (upstream node recipe installs corepack globally).
RUN corepack enable --install-directory /opt/toolchains/node-16/bin || true \
    && corepack enable --install-directory /opt/toolchains/node-20/bin || true

RUN node -v && npm -v

# Build-time guards: the apt binaries must survive the build (a kaniko snapshot loss dropped
# them from the first push).
RUN test -x /usr/bin/git && test -x /usr/bin/wget && test -x /usr/bin/curl \
    && test -x "$(command -v python3)" \
    || { echo "node base apt binaries missing" >&2; exit 1; }

RUN mkdir -p /logs /output && chmod 777 /logs /output
