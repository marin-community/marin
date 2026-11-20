# SWE-rebench V2.1 repair — julia group base.
# Covers julia_base rows (434 tasks).
# Upstream anchor: base_dockerfiles/Dockerfile_julia (verbatim recipe below).
# Interpreter provisioning (repair, disclosed): apt python3 (bookworm ships 3.11).

FROM julia:1.10-bookworm@sha256:46a7e2f54218c89aaadbe6cbcdbb2e68ced31f118247f6a1366d4a07a75b3ec2

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
    git curl wget unzip zip ca-certificates build-essential pkg-config gfortran cmake gnupg \
    libssl-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN test "$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "python3 >= 3.9 required on PATH" >&2; exit 1; }

RUN julia --version | grep -q '1.10' || { echo "julia 1.10 required" >&2; exit 1; }

RUN mkdir -p /logs /output && chmod 777 /logs /output
