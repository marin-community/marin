# SWE-rebench V2.1 repair — go group base.
# Covers go_1.18.10 / 1.19.13 / 1.20.14 / 1.21.13 / 1.22.12 / 1.23.8 rows (2,269 tasks).
# Upstream anchor: base_dockerfiles/Dockerfile_go_* (ubuntu:22.04 + official Go tarballs).
# Per-version toolchains: six Go distributions side by side under /opt/toolchains/go/<version>;
# the task setup symlinks the row's declared version into /usr/local/go.
# Interpreter provisioning (repair, disclosed): apt python3 (jammy ships 3.10).

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
    && apt-get update \
    && if [ -f /var/lib/dpkg/statoverride ]; then \
         while read -r mode owner group path; do \
           [ -n "$path" ] || continue; \
           if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then \
             dpkg-statoverride --remove "$path" || true; \
           fi; \
         done < /var/lib/dpkg/statoverride; \
       fi \
    && apt-get install -y --no-install-recommends \
    wget git curl build-essential python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Declared interpreter provisioning guard (the bootstrap needs python >= 3.9 on PATH).
RUN test "$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "python3 >= 3.9 required on PATH" >&2; exit 1; }

# Per-version Go toolchains, pinned official tarballs (upstream go recipe pattern).
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) suffix="linux-amd64" ;; \
      arm64) suffix="linux-arm64" ;; \
      *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    for version in 1.18.10 1.19.13 1.20.14 1.21.13 1.22.12 1.23.8; do \
      url="https://dl.google.com/go/go${version}.${suffix}.tar.gz"; \
      mkdir -p "/opt/toolchains/go/${version}"; \
      curl -fsSL "$url" | tar -xz -C "/opt/toolchains/go/${version}" --strip-components=1; \
    done
# Build-time guards: the controller bootstrap precondition, the declared toolchains, and the
# apt binaries (a kaniko snapshot loss dropped /usr/bin/{git,wget,curl} from the first push).
RUN test -x /usr/bin/git && test -x /usr/bin/wget && test -x /usr/bin/curl \
    && test -x "$(command -v python3)" \
    || { echo "go base apt binaries missing" >&2; exit 1; }
RUN test -x /opt/toolchains/go/1.19.13/bin/go \
    || { echo "go toolchains missing" >&2; exit 1; }

# The task setup re-points /usr/local/go per row; keep the default on the oldest declared
# version so the image is self-consistent without setup.
RUN ln -sfn /opt/toolchains/go/1.18.10 /usr/local/go
ENV PATH="/usr/local/go/bin:${PATH}"

RUN mkdir -p /logs /output && chmod 777 /logs /output
