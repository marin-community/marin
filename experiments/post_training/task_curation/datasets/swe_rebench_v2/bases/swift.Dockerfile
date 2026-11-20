# SWE-rebench V2.1 repair — swift group base.
# Covers swift_base rows (127 tasks).
# Upstream anchor: base_dockerfiles/Dockerfile_swift (swift:5.10-jammy; jammy ships python3.10).

FROM swift:5.10-jammy@sha256:ffc42e399ad0cad324f68d58f65e4d5f2c22eb9df73c4dfa249e898d497254b3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Disclosed repair normalization: apt runs non-interactively with --force-confdef/--force-confold
# (keeps the image's existing conffiles; an upgraded media-types prompted on mime.types and
# aborted non-interactive builds).

RUN apt-get update \
    && if [ -f /var/lib/dpkg/statoverride ]; then \
         while read -r mode owner group path; do \
           [ -n "$path" ] || continue; \
           if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then \
             dpkg-statoverride --remove "$path" || true; \
           fi; \
         done < /var/lib/dpkg/statoverride; \
       fi \
    && apt-get -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends \
    wget curl git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN test "$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "python3 >= 3.9 required on PATH" >&2; exit 1; }

RUN swift --version | grep -q '5.10' || { echo "swift 5.10 required" >&2; exit 1; }

RUN mkdir -p /logs /output && chmod 777 /logs /output
